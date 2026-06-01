#!/usr/bin/env python3
import argparse
import csv
import itertools
import os
import random
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import LandslideDataset, LandslideNet  # noqa: E402

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from cross_validate import analyze_cv_results, setup_fold_logger, train_model  # noqa: E402
from plotting_utils import add_panel_label, add_wgs84_axes, save_figure, setup_plot_style, to_wgs84  # noqa: E402


def normalize_path(path):
    if path is None:
        return None
    value = os.path.expanduser(str(path).strip())
    if os.name != "nt" and (value.startswith("/") or value.startswith(".")):
        value = value.replace("\\", "/")
    return value


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def read_xml_params(xml_path):
    root = ET.parse(xml_path).getroot()
    params = {}
    for param in root.findall("param"):
        name = param.find("name").text
        value = param.find("value").text
        params[name] = normalize_path(value)
    return params


def parse_patch_origin(file_name):
    stem = Path(file_name).stem
    parts = stem.split("_")
    if len(parts) < 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def get_label_subdir(labels_dir):
    subdirs = sorted(
        d for d in os.listdir(labels_dir)
        if os.path.isdir(os.path.join(labels_dir, d))
    )
    if not subdirs:
        raise FileNotFoundError(f"No label subdirectory found in {labels_dir}")
    return os.path.join(labels_dir, subdirs[0])


def read_factor_sample_means(factors_dir, factor_subdirs, label_file, sample_mask):
    values = []
    for factor_subdir in factor_subdirs:
        factor_dir = os.path.join(factors_dir, factor_subdir)
        factor_path = os.path.join(factor_dir, label_file)
        if not os.path.exists(factor_path):
            factor_files = sorted(f for f in os.listdir(factor_dir) if f.endswith(".tif"))
            try:
                factor_path = os.path.join(factor_dir, next(f for f in factor_files if f == label_file))
            except StopIteration:
                values.append(0.0)
                continue

        with rasterio.open(factor_path) as src:
            data = src.read(1)
            nodata_val = src.nodata

        valid_mask = np.isfinite(data)
        if nodata_val is not None:
            valid_mask &= ~np.isclose(data, nodata_val, equal_nan=True)

        factor_mask = valid_mask & sample_mask
        if not np.any(factor_mask):
            factor_mask = valid_mask

        if np.any(factor_mask):
            values.append(float(np.nanmean(np.clip(data[factor_mask], 0, 1))))
        else:
            values.append(0.0)
    return values


def build_patch_metadata(factors_dir, labels_dir, crop_size, include_feature_stats=False):
    dataset = LandslideDataset(
        factors_dir=factors_dir,
        labels_dir=labels_dir,
        crop_size=crop_size,
        train=False,
    )
    label_subdir = get_label_subdir(labels_dir)
    rows = []

    for file_name in dataset.valid_files:
        label_path = os.path.join(label_subdir, file_name)
        with rasterio.open(label_path) as src:
            label = src.read(1)
            height, width = label.shape
            x_center, y_center = src.xy(height // 2, width // 2)
            lon_center, lat_center = to_wgs84([x_center], [y_center], src.crs)
            transform = src.transform

        origin = parse_patch_origin(file_name)
        if origin is None:
            row_origin = len(rows)
            col_origin = 0
            parsed_origin = False
        else:
            row_origin, col_origin = origin
            parsed_origin = True

        landslide_pixels = int(np.sum(label == 1))
        non_landslide_pixels = int(np.sum(label == 2))
        valid_pixels = landslide_pixels + non_landslide_pixels
        landslide_ratio = landslide_pixels / valid_pixels if valid_pixels else 0.0
        sample_mask = (label == 1) | (label == 2)

        row = {
            "file": file_name,
            "row_origin": row_origin,
            "col_origin": col_origin,
            "center_row": row_origin + height / 2.0,
            "center_col": col_origin + width / 2.0,
            "x": float(x_center),
            "y": float(y_center),
            "lon": float(lon_center[0]),
            "lat": float(lat_center[0]),
            "pixel_width": abs(float(transform.a)),
            "pixel_height": abs(float(transform.e)),
            "landslide_pixels": landslide_pixels,
            "non_landslide_pixels": non_landslide_pixels,
            "valid_pixels": valid_pixels,
            "landslide_ratio": landslide_ratio,
            "strata": 1 if landslide_ratio > 0.05 else 0,
            "parsed_origin": parsed_origin,
        }
        if include_feature_stats:
            row["feature_values"] = read_factor_sample_means(
                factors_dir,
                dataset.factors_subdirs,
                file_name,
                sample_mask,
            )
        rows.append(row)

    if not rows:
        raise RuntimeError("No valid label patches found for spatial CV.")
    return rows


def assign_grid_blocks(rows, block_size_pixels=None, block_size_map_units=None):
    if block_size_map_units is not None:
        min_x = min(row["x"] for row in rows)
        min_y = min(row["y"] for row in rows)
        for row in rows:
            block_col = int(np.floor((row["x"] - min_x) / block_size_map_units))
            block_row = int(np.floor((row["y"] - min_y) / block_size_map_units))
            row["block_row"] = block_row
            row["block_col"] = block_col
            row["block_id"] = f"r{block_row}_c{block_col}"
        return

    if block_size_pixels is None:
        raise ValueError("Either block_size_pixels or block_size_map_units must be provided.")

    for row in rows:
        block_row = int(np.floor(row["center_row"] / block_size_pixels))
        block_col = int(np.floor(row["center_col"] / block_size_pixels))
        row["block_row"] = block_row
        row["block_col"] = block_col
        row["block_id"] = f"r{block_row}_c{block_col}"


def split_stats(rows):
    stats = {
        "patches": len(rows),
        "blocks": len({row["block_id"] for row in rows}),
        "landslide_pixels": int(sum(row["landslide_pixels"] for row in rows)),
        "non_landslide_pixels": int(sum(row["non_landslide_pixels"] for row in rows)),
    }
    valid_pixels = stats["landslide_pixels"] + stats["non_landslide_pixels"]
    stats["valid_pixels"] = valid_pixels
    stats["landslide_ratio"] = stats["landslide_pixels"] / valid_pixels if valid_pixels else 0.0
    return stats


def summarize_blocks(rows):
    blocks = {}
    for row in rows:
        block = blocks.setdefault(row["block_id"], {
            "block_id": row["block_id"],
            "block_row": row["block_row"],
            "block_col": row["block_col"],
            "rows": [],
            "patches": 0,
            "landslide_pixels": 0,
            "non_landslide_pixels": 0,
            "feature_sums": None,
            "feature_weight": 0,
        })
        block["rows"].append(row)
        block["patches"] += 1
        block["landslide_pixels"] += row["landslide_pixels"]
        block["non_landslide_pixels"] += row["non_landslide_pixels"]
        if "feature_values" in row:
            feature_values = np.asarray(row["feature_values"], dtype=np.float64)
            if block["feature_sums"] is None:
                block["feature_sums"] = np.zeros_like(feature_values)
            feature_weight = max(1, row["valid_pixels"])
            block["feature_sums"] += feature_values * feature_weight
            block["feature_weight"] += feature_weight
    return sorted(blocks.values(), key=lambda x: (x["block_row"], x["block_col"], x["block_id"]))


def empty_fold_stats():
    return {
        "patches": 0,
        "landslide_pixels": 0,
        "non_landslide_pixels": 0,
        "block_ids": [],
        "feature_sums": None,
        "feature_weight": 0,
    }


def add_block_to_stats(stats, block):
    stats["patches"] += block["patches"]
    stats["landslide_pixels"] += block["landslide_pixels"]
    stats["non_landslide_pixels"] += block["non_landslide_pixels"]
    stats["block_ids"].append(block["block_id"])
    if block.get("feature_sums") is not None:
        if stats["feature_sums"] is None:
            stats["feature_sums"] = np.zeros_like(block["feature_sums"])
        stats["feature_sums"] += block["feature_sums"]
        stats["feature_weight"] += block["feature_weight"]


def make_feature_targets(blocks):
    feature_blocks = [block for block in blocks if block.get("feature_sums") is not None and block["feature_weight"] > 0]
    if not feature_blocks:
        return None

    total_sums = np.sum([block["feature_sums"] for block in feature_blocks], axis=0)
    total_weight = sum(block["feature_weight"] for block in feature_blocks)
    means = total_sums / max(total_weight, 1)

    block_means = np.stack([
        block["feature_sums"] / max(block["feature_weight"], 1)
        for block in feature_blocks
    ])
    scales = np.std(block_means, axis=0)
    scales = np.where(scales < 1e-6, 1.0, scales)

    return {
        "means": means,
        "scales": scales,
    }


def fold_balance_score(folds, targets):
    score = 0.0
    target_ratio = targets["landslide_ratio"]
    for fold in folds:
        valid = fold["landslide_pixels"] + fold["non_landslide_pixels"]
        if fold["patches"] == 0 or valid == 0:
            score += 1_000_000.0
            continue

        ratio = fold["landslide_pixels"] / valid
        score += ((fold["patches"] - targets["patches"]) / max(targets["patches"], 1.0)) ** 2
        score += 1.8 * ((fold["landslide_pixels"] - targets["landslide_pixels"]) / max(targets["landslide_pixels"], 1.0)) ** 2
        score += 1.2 * ((fold["non_landslide_pixels"] - targets["non_landslide_pixels"]) / max(targets["non_landslide_pixels"], 1.0)) ** 2
        score += 0.6 * ((ratio - target_ratio) / max(target_ratio, 1e-6)) ** 2
        if fold["landslide_pixels"] == 0:
            score += 100.0
        if fold["non_landslide_pixels"] == 0:
            score += 100.0

        feature_targets = targets.get("feature_targets")
        if feature_targets is not None and fold.get("feature_sums") is not None and fold["feature_weight"] > 0:
            fold_means = fold["feature_sums"] / max(fold["feature_weight"], 1)
            standardized = (fold_means - feature_targets["means"]) / feature_targets["scales"]
            score += targets.get("feature_weight", 0.35) * float(np.mean(standardized ** 2))
    return score


def build_fold_stats(blocks, assignment, k_folds):
    folds = [empty_fold_stats() for _ in range(k_folds)]
    for block_idx, fold_idx in enumerate(assignment):
        add_block_to_stats(folds[fold_idx], blocks[block_idx])
    return folds


def greedy_initial_assignment(blocks, k_folds, targets, rng, shuffle):
    order = list(range(len(blocks)))
    if shuffle:
        rng.shuffle(order)
    else:
        order.sort(
            key=lambda idx: (
                blocks[idx]["landslide_pixels"] + blocks[idx]["non_landslide_pixels"],
                blocks[idx]["patches"],
            ),
            reverse=True,
        )

    folds = [empty_fold_stats() for _ in range(k_folds)]
    assignment = [-1] * len(blocks)
    empty_folds = list(range(k_folds))

    for block_idx in order:
        if empty_folds:
            fold_idx = empty_folds.pop(0)
        else:
            candidates = []
            for candidate_idx in range(k_folds):
                trial = [dict(fold, block_ids=list(fold["block_ids"])) for fold in folds]
                add_block_to_stats(trial[candidate_idx], blocks[block_idx])
                candidates.append((fold_balance_score(trial, targets), candidate_idx))
            fold_idx = min(candidates, key=lambda item: item[0])[1]

        assignment[block_idx] = fold_idx
        add_block_to_stats(folds[fold_idx], blocks[block_idx])

    return assignment


def improve_assignment(blocks, assignment, k_folds, targets, max_passes=8):
    best_assignment = list(assignment)
    best_score = fold_balance_score(build_fold_stats(blocks, best_assignment, k_folds), targets)

    for _ in range(max_passes):
        improved = False
        fold_counts = Counter(best_assignment)

        for block_idx, current_fold in enumerate(list(best_assignment)):
            if fold_counts[current_fold] <= 1:
                continue
            for candidate_fold in range(k_folds):
                if candidate_fold == current_fold:
                    continue
                trial = list(best_assignment)
                trial[block_idx] = candidate_fold
                score = fold_balance_score(build_fold_stats(blocks, trial, k_folds), targets)
                if score + 1e-12 < best_score:
                    best_assignment = trial
                    best_score = score
                    improved = True
                    fold_counts = Counter(best_assignment)

        if not improved:
            break

    return best_assignment, best_score


def make_balanced_spatial_folds(rows, k_folds, seed, search_restarts=40, feature_weight=2.0):
    blocks = summarize_blocks(rows)
    if len(blocks) < k_folds:
        raise RuntimeError("Spatial CV needs at least as many blocks as folds.")

    totals = split_stats(rows)
    targets = {
        "patches": totals["patches"] / k_folds,
        "landslide_pixels": totals["landslide_pixels"] / k_folds,
        "non_landslide_pixels": totals["non_landslide_pixels"] / k_folds,
        "landslide_ratio": totals["landslide_ratio"],
        "feature_targets": make_feature_targets(blocks),
        "feature_weight": feature_weight,
    }

    rng = random.Random(seed)
    best_assignment = None
    best_score = np.inf

    for restart in range(max(1, search_restarts)):
        assignment = greedy_initial_assignment(
            blocks,
            k_folds,
            targets,
            rng,
            shuffle=restart > 0,
        )
        assignment, score = improve_assignment(blocks, assignment, k_folds, targets)
        if score < best_score:
            best_assignment = assignment
            best_score = score

    fold_rows = [[] for _ in range(k_folds)]
    for block_idx, fold_idx in enumerate(best_assignment):
        fold_rows[fold_idx].extend(blocks[block_idx]["rows"])
    return fold_rows


def select_balanced_validation_blocks(blocks, val_fraction, seed, search_restarts=120, feature_weight=2.0):
    n_blocks = len(blocks)
    target_blocks = max(1, min(n_blocks - 1, int(round(n_blocks * val_fraction))))
    target_counts = sorted({
        count for count in (target_blocks - 1, target_blocks, target_blocks + 1)
        if 1 <= count < n_blocks
    })

    total_patches = sum(block["patches"] for block in blocks)
    total_pos = sum(block["landslide_pixels"] for block in blocks)
    total_neg = sum(block["non_landslide_pixels"] for block in blocks)
    target = {
        "patches": total_patches * val_fraction,
        "landslide_pixels": total_pos * val_fraction,
        "non_landslide_pixels": total_neg * val_fraction,
        "landslide_ratio": total_pos / (total_pos + total_neg) if total_pos + total_neg else 0.0,
    }
    feature_targets = make_feature_targets(blocks)

    def score(indices):
        val_blocks = [blocks[idx] for idx in indices]
        val_pos = sum(block["landslide_pixels"] for block in val_blocks)
        val_neg = sum(block["non_landslide_pixels"] for block in val_blocks)
        val_patches = sum(block["patches"] for block in val_blocks)
        val_ratio = val_pos / (val_pos + val_neg) if val_pos + val_neg else 0.0
        train_pos = total_pos - val_pos
        train_neg = total_neg - val_neg

        item_score = ((val_patches - target["patches"]) / max(target["patches"], 1.0)) ** 2
        item_score += 1.8 * ((val_pos - target["landslide_pixels"]) / max(target["landslide_pixels"], 1.0)) ** 2
        item_score += 1.2 * ((val_neg - target["non_landslide_pixels"]) / max(target["non_landslide_pixels"], 1.0)) ** 2
        item_score += 0.6 * ((val_ratio - target["landslide_ratio"]) / max(target["landslide_ratio"], 1e-6)) ** 2
        if val_pos == 0 or val_neg == 0 or train_pos == 0 or train_neg == 0:
            item_score += 100.0

        if feature_targets is not None:
            feature_blocks = [block for block in val_blocks if block.get("feature_sums") is not None]
            if feature_blocks:
                feature_sums = np.sum([block["feature_sums"] for block in feature_blocks], axis=0)
                feature_total_weight = sum(block["feature_weight"] for block in feature_blocks)
                val_feature_means = feature_sums / max(feature_total_weight, 1)
                standardized = (val_feature_means - feature_targets["means"]) / feature_targets["scales"]
                item_score += feature_weight * float(np.mean(standardized ** 2))
        return item_score

    best_indices = None
    best_score = np.inf

    if n_blocks <= 24:
        for count in target_counts:
            for indices in itertools.combinations(range(n_blocks), count):
                item_score = score(indices)
                if item_score < best_score:
                    best_indices = set(indices)
                    best_score = item_score
    else:
        rng = random.Random(seed)
        for _ in range(max(1, search_restarts)):
            count = rng.choice(target_counts)
            indices = set(rng.sample(range(n_blocks), count))
            improved = True
            while improved:
                improved = False
                current_score = score(indices)
                for remove_idx in list(indices):
                    if remove_idx not in indices:
                        continue
                    for add_idx in range(n_blocks):
                        if add_idx in indices:
                            continue
                        trial = set(indices)
                        trial.remove(remove_idx)
                        trial.add(add_idx)
                        trial_score = score(trial)
                        if trial_score + 1e-12 < current_score:
                            indices = trial
                            current_score = trial_score
                            improved = True
                            break
                    if improved:
                        break
            if current_score < best_score:
                best_indices = set(indices)
                best_score = current_score

    return {blocks[idx]["block_id"] for idx in best_indices}


def remove_buffered_train_files(train_rows, test_rows, buffer_pixels):
    if buffer_pixels <= 0:
        return train_rows

    test_centers = np.array(
        [[row["center_row"], row["center_col"]] for row in test_rows],
        dtype=np.float64,
    )
    if test_centers.size == 0:
        return train_rows

    kept = []
    for row in train_rows:
        center = np.array([row["center_row"], row["center_col"]], dtype=np.float64)
        chebyshev_dist = np.max(np.abs(test_centers - center), axis=1)
        if np.min(chebyshev_dist) > float(buffer_pixels):
            kept.append(row)
    return kept


def split_train_val_by_blocks(train_rows, val_fraction, seed, balanced=True,
                              search_restarts=120, feature_weight=2.0):
    block_ids = sorted({row["block_id"] for row in train_rows})

    if len(block_ids) <= 1 or val_fraction <= 0:
        return train_rows, []

    if balanced:
        blocks = summarize_blocks(train_rows)
        val_blocks = select_balanced_validation_blocks(
            blocks,
            val_fraction,
            seed,
            search_restarts=search_restarts,
            feature_weight=feature_weight,
        )
    else:
        rng = random.Random(seed)
        rng.shuffle(block_ids)
        n_val_blocks = max(1, int(round(len(block_ids) * val_fraction)))
        n_val_blocks = min(n_val_blocks, len(block_ids) - 1)
        val_blocks = set(block_ids[:n_val_blocks])

    final_train = [row for row in train_rows if row["block_id"] not in val_blocks]
    val_rows = [row for row in train_rows if row["block_id"] in val_blocks]
    return final_train, val_rows


def write_manifest(path, fold_records):
    fieldnames = [
        "fold", "split", "file", "block_id", "block_row", "block_col",
        "row_origin", "col_origin", "center_row", "center_col", "x", "y", "lon", "lat",
        "landslide_pixels", "non_landslide_pixels", "valid_pixels",
        "landslide_ratio",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in fold_records:
            writer.writerow({k: record.get(k, "") for k in fieldnames})


def write_block_summary(path, rows):
    summary = {}
    for row in rows:
        item = summary.setdefault(row["block_id"], {
            "block_id": row["block_id"],
            "block_row": row["block_row"],
            "block_col": row["block_col"],
            "patches": 0,
            "landslide_pixels": 0,
            "non_landslide_pixels": 0,
        })
        item["patches"] += 1
        item["landslide_pixels"] += row["landslide_pixels"]
        item["non_landslide_pixels"] += row["non_landslide_pixels"]

    fieldnames = [
        "block_id", "block_row", "block_col", "patches",
        "landslide_pixels", "non_landslide_pixels",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in sorted(summary.values(), key=lambda x: (x["block_row"], x["block_col"])):
            writer.writerow(item)


def write_fold_summary(path, records):
    fieldnames = [
        "fold", "split", "patches", "blocks", "landslide_pixels",
        "non_landslide_pixels", "valid_pixels", "landslide_ratio",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in fieldnames})


def plot_fold_layout(rows, fold_assignments, output_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping fold layout plot.")
        return

    setup_plot_style(plt)
    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    palette = [
        "#2166ac", "#b2182b", "#1b7837", "#762a83", "#f4a582",
        "#4393c3", "#d6604d", "#5aae61", "#9970ab", "#92c5de",
    ]
    for fold in sorted(set(fold_assignments.values())):
        fold_rows = [row for row in rows if fold_assignments[row["file"]] == fold]
        color = palette[(fold - 1) % len(palette)]
        ax.scatter(
            [row.get("lon", row["x"]) for row in fold_rows],
            [row.get("lat", row["y"]) for row in fold_rows],
            s=34,
            alpha=0.78,
            c=color,
            edgecolors="white",
            linewidths=0.35,
            label=f"Fold {fold}",
        )
    ax.set_aspect("equal", adjustable="box")
    add_wgs84_axes(ax, "Spatial block CV fold layout")
    add_panel_label(ax, "WGS84")
    ax.legend(markerscale=1.4, fontsize=8, ncol=2, loc="best", title="Test fold")
    fig.tight_layout()
    save_figure(fig, output_path)
    plt.close()


def make_loader(factors_dir, labels_dir, files, crop_size, batch_size, num_workers, train, seed):
    dataset = LandslideDataset(
        factors_dir=factors_dir,
        labels_dir=labels_dir,
        crop_size=crop_size,
        train=train,
        file_list=files,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)

    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        generator=generator,
        worker_init_fn=seed_worker,
    )


def run_spatial_cv(args):
    params = read_xml_params(args.xml)
    factors_dir = params["output_factors_dir"]
    labels_dir = params["output_labels_dir"]
    train_output = params["train_output"]
    crop_size = int(params["crop_size"])
    buffer_pixels = args.buffer_pixels
    batch_size = int(params["batch_size"])
    num_workers = int(params["num_workers"])
    num_epochs = int(params["num_epochs"])
    lr = float(params["lr"])
    patience = int(params["patience"])
    num_bands = int(params["num_bands"])
    weight_decay = float(params["weight_decay"])
    device_ids = list(map(int, params["device_ids"].strip("[]").split(",")))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = normalize_path(args.output_dir) if args.output_dir else os.path.join(
        train_output,
        f"Spatial_Block_CV_{timestamp}",
    )
    os.makedirs(output_dir, exist_ok=True)

    set_global_seed(args.seed)
    rows = build_patch_metadata(
        factors_dir,
        labels_dir,
        crop_size,
        include_feature_stats=args.balance_features,
    )
    block_size_pixels = args.block_size_pixels
    if block_size_pixels is None and args.block_size_map_units is None:
        block_size_pixels = int(round(crop_size * 1.5))

    assign_grid_blocks(
        rows,
        block_size_pixels=block_size_pixels,
        block_size_map_units=args.block_size_map_units,
    )

    groups = np.array([row["block_id"] for row in rows])
    y = np.array([row["strata"] for row in rows])
    n_groups = len(set(groups))
    k_folds = min(args.k_folds, n_groups)
    if k_folds < 2:
        raise RuntimeError("Spatial CV needs at least two spatial blocks.")

    if args.fold_strategy == "balanced":
        test_folds = make_balanced_spatial_folds(
            rows,
            k_folds,
            args.seed,
            search_restarts=args.balance_restarts,
            feature_weight=args.feature_balance_weight,
        )
    else:
        from sklearn.model_selection import GroupKFold

        gkf = GroupKFold(n_splits=k_folds)
        test_folds = [
            [rows[i] for i in test_idx]
            for _, test_idx in gkf.split(np.zeros(len(rows)), y, groups)
        ]

    all_metrics = []
    fold_records = []
    fold_summary_records = []
    fold_assignments = {
        row["file"]: fold_idx
        for fold_idx, test_rows in enumerate(test_folds, start=1)
        for row in test_rows
    }
    plot_fold_layout(rows, fold_assignments, os.path.join(output_dir, "spatial_cv_fold_layout.png"))

    print(
        f"Spatial blocks: {n_groups}; folds: {k_folds}; patches: {len(rows)}; "
        f"block_size_pixels={block_size_pixels}; buffer_pixels={buffer_pixels}; "
        f"strategy={args.fold_strategy}; balance_features={args.balance_features}; "
        f"selection_metric={args.selection_metric}; threshold_metric={args.threshold_metric}; "
        f"eval_tta={args.eval_tta}"
    )
    if len(rows) / k_folds < 50:
        print(
            "Warning: fewer than 50 patches per spatial test fold. "
            "F1 coefficient of variation can be inflated by small fold size; "
            "5-fold spatial CV is recommended for this dataset."
        )
    if args.dry_run:
        print("Dry run enabled: writing manifests only, no model training.")

    for fold_idx, test_rows in enumerate(test_folds, start=1):
        set_global_seed(args.seed + fold_idx)
        test_files = {row["file"] for row in test_rows}
        train_rows = [row for row in rows if row["file"] not in test_files]
        train_rows = remove_buffered_train_files(train_rows, test_rows, buffer_pixels)
        train_rows, val_rows = split_train_val_by_blocks(
            train_rows,
            args.val_fraction,
            args.seed + fold_idx,
            balanced=args.val_strategy == "balanced",
            search_restarts=args.balance_restarts,
            feature_weight=args.feature_balance_weight,
        )

        if not val_rows:
            val_rows = test_rows

        for split_name, split_rows in (
            ("train", train_rows),
            ("val", val_rows),
            ("test", test_rows),
        ):
            for row in split_rows:
                record = dict(row)
                record["fold"] = fold_idx
                record["split"] = split_name
                fold_records.append(record)

        split_summaries = {}
        for split_name, split_rows in (("train", train_rows), ("val", val_rows), ("test", test_rows)):
            stats = split_stats(split_rows)
            stats["fold"] = fold_idx
            stats["split"] = split_name
            fold_summary_records.append(stats)
            split_summaries[split_name] = stats

        print(f"Fold {fold_idx}:")
        for split_name in ("train", "val", "test"):
            stats = split_summaries[split_name]
            print(
                f"  {split_name:5s} patches={stats['patches']:3d} "
                f"blocks={stats['blocks']:2d} pos={stats['landslide_pixels']:5d} "
                f"neg={stats['non_landslide_pixels']:5d} "
                f"pos_ratio={stats['landslide_ratio']:.4f}"
            )

        if args.dry_run:
            continue

        fold_dir = os.path.join(output_dir, f"Fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)
        logger = setup_fold_logger(fold_dir, fold_idx)

        train_loader = make_loader(
            factors_dir, labels_dir, [row["file"] for row in train_rows],
            crop_size, batch_size, num_workers, train=True, seed=args.seed + fold_idx * 100 + 1,
        )
        val_loader = make_loader(
            factors_dir, labels_dir, [row["file"] for row in val_rows],
            crop_size, batch_size, num_workers, train=False, seed=args.seed + fold_idx * 100 + 2,
        )
        test_loader = make_loader(
            factors_dir, labels_dir, [row["file"] for row in test_rows],
            crop_size, batch_size, num_workers, train=False, seed=args.seed + fold_idx * 100 + 3,
        )

        model = LandslideNet(num_bands=num_bands)
        metrics = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            num_epochs=num_epochs,
            lr=lr,
            device_ids=device_ids,
            patience=patience,
            output_dir=fold_dir,
            weight_decay=weight_decay,
            logger=logger,
            selection_metric=args.selection_metric,
            threshold_metric=args.threshold_metric,
            use_tta=args.eval_tta,
        )
        if metrics:
            metrics["fold"] = fold_idx
            metrics["train_patches"] = len(train_rows)
            metrics["val_patches"] = len(val_rows)
            metrics["test_patches"] = len(test_rows)
            metrics["train_blocks"] = len({row["block_id"] for row in train_rows})
            metrics["val_blocks"] = len({row["block_id"] for row in val_rows})
            metrics["test_blocks"] = len({row["block_id"] for row in test_rows})
            all_metrics.append(metrics)

        del model, train_loader, val_loader, test_loader
        torch.cuda.empty_cache()

    write_manifest(os.path.join(output_dir, "spatial_cv_manifest.csv"), fold_records)
    write_block_summary(os.path.join(output_dir, "spatial_block_summary.csv"), rows)
    write_fold_summary(os.path.join(output_dir, "spatial_fold_summary.csv"), fold_summary_records)

    if all_metrics:
        analyze_cv_results(all_metrics, output_dir)

    print(f"Spatial CV outputs saved to: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Spatial block cross-validation for LandslideNet.")
    parser.add_argument("xml", help="Path to Landslide_susceptibility_mapping.xml")
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--block-size-pixels", type=int, default=None)
    parser.add_argument("--block-size-map-units", type=float, default=None)
    parser.add_argument(
        "--buffer-pixels",
        type=int,
        default=0,
        help="Remove training patches within this Chebyshev pixel distance from test patch centers. Use 512 for a conservative non-overlap buffer.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20250609)
    parser.add_argument("--fold-strategy", choices=("balanced", "group-kfold"), default="balanced")
    parser.add_argument("--val-strategy", choices=("balanced", "random"), default="balanced")
    parser.add_argument("--balance-restarts", type=int, default=40)
    parser.add_argument("--balance-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--feature-balance-weight", type=float, default=2.0)
    parser.add_argument("--selection-metric", choices=("f1", "kappa", "iou"), default="f1")
    parser.add_argument("--threshold-metric", choices=("f1", "kappa", "iou"), default="f1")
    parser.add_argument("--eval-tta", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Only write spatial fold manifests.")
    return parser.parse_args()


if __name__ == "__main__":
    run_spatial_cv(parse_args())
