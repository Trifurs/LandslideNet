#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from plotting_utils import add_panel_label, add_wgs84_axes, save_figure, setup_plot_style, to_wgs84


FACTOR_KEYWORDS = {
    "slope": ("slope",),
    "elevation": ("dem", "elev", "elevation"),
    "relief": ("relief",),
    "roughness": ("roughness",),
}

LABEL_DISPLAY = {
    "landslide": "Landslide",
    "non_landslide": "Non-landslide",
}


def normalize_path(path):
    if path is None:
        return None
    value = os.path.expanduser(str(path).strip())
    if os.name != "nt" and (value.startswith("/") or value.startswith(".")):
        value = value.replace("\\", "/")
    return value


def read_xml_params(xml_path):
    root = ET.parse(xml_path).getroot()
    params = {}
    for param in root.findall("param"):
        name = param.find("name").text
        value = param.find("value").text
        params[name] = normalize_path(value)
    return params


def find_first_tif(path):
    if os.path.isfile(path):
        return path

    direct = sorted(
        os.path.join(path, f)
        for f in os.listdir(path)
        if f.lower().endswith((".tif", ".tiff"))
    )
    if direct:
        return direct[0]

    for root, _, files in os.walk(path):
        tif_files = sorted(f for f in files if f.lower().endswith((".tif", ".tiff")))
        if tif_files:
            return os.path.join(root, tif_files[0])

    raise FileNotFoundError(f"No GeoTIFF found in {path}")


def list_factor_files(factors_dir):
    direct = sorted(
        os.path.join(factors_dir, f)
        for f in os.listdir(factors_dir)
        if f.lower().endswith((".tif", ".tiff"))
    )
    if direct:
        return direct

    files = []
    for subdir in sorted(os.listdir(factors_dir)):
        subdir_path = os.path.join(factors_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        tif_files = sorted(
            os.path.join(subdir_path, f)
            for f in os.listdir(subdir_path)
            if f.lower().endswith((".tif", ".tiff"))
        )
        if tif_files:
            files.append(tif_files[0])
    return files


def pick_factor_files(factors_dir):
    all_files = list_factor_files(factors_dir)
    selected = {}
    for canonical_name, keywords in FACTOR_KEYWORDS.items():
        for path in all_files:
            stem = Path(path).stem.lower()
            compact = stem.replace("_", "").replace("-", "").replace(" ", "")
            if any(keyword in stem or keyword in compact for keyword in keywords):
                selected[canonical_name] = path
                break
    return selected


def assert_factor_alignment(label_src, factor_sources):
    aligned = {}
    for name, src in factor_sources.items():
        same_shape = src.width == label_src.width and src.height == label_src.height
        same_transform = src.transform.almost_equals(label_src.transform)
        if same_shape and same_transform:
            aligned[name] = src
        else:
            print(f"Skipping {name}: raster grid does not match label raster.")
            src.close()
    return aligned


def iter_windows(width, height, block_size):
    for row in range(0, height, block_size):
        for col in range(0, width, block_size):
            yield Window(col, row, min(block_size, width - col), min(block_size, height - row))


def collect_sample_table(label_path, factor_paths, block_size):
    records = []
    factor_sources = {}
    with rasterio.open(label_path) as label_src:
        for name, path in factor_paths.items():
            factor_sources[name] = rasterio.open(path)
        factor_sources = assert_factor_alignment(label_src, factor_sources)

        for window in iter_windows(label_src.width, label_src.height, block_size):
            labels = label_src.read(1, window=window)
            sample_mask = (labels == 1) | (labels == 2)
            if not np.any(sample_mask):
                continue

            rel_rows, rel_cols = np.nonzero(sample_mask)
            abs_rows = rel_rows + int(window.row_off)
            abs_cols = rel_cols + int(window.col_off)
            xs, ys = rasterio.transform.xy(label_src.transform, abs_rows, abs_cols)
            lons, lats = to_wgs84(xs, ys, label_src.crs)
            values = labels[sample_mask]

            factor_values = {}
            for name, src in factor_sources.items():
                data = src.read(1, window=window)
                nodata = src.nodata
                selected = data[sample_mask].astype(np.float64)
                if nodata is not None:
                    selected[np.isclose(selected, nodata)] = np.nan
                factor_values[name] = selected

            for i, label_value in enumerate(values):
                record = {
                    "row": int(abs_rows[i]),
                    "col": int(abs_cols[i]),
                    "x": float(xs[i]),
                    "y": float(ys[i]),
                    "lon": float(lons[i]),
                    "lat": float(lats[i]),
                    "label": int(label_value),
                    "label_name": "landslide" if int(label_value) == 1 else "non_landslide",
                }
                for name, array in factor_values.items():
                    record[name] = float(array[i]) if np.isfinite(array[i]) else np.nan
                records.append(record)

        for src in factor_sources.values():
            src.close()

    if not records:
        raise RuntimeError("No DWSS sample pixels with labels 1 or 2 were found.")
    return pd.DataFrame.from_records(records)


def valid_raster_values(data, nodata):
    values = data.astype(np.float64, copy=False).ravel()
    mask = np.isfinite(values)
    if nodata is not None:
        mask &= ~np.isclose(values, nodata)
    return values[mask]


def update_reservoir(reservoir, seen, values, rng):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return seen

    capacity = len(reservoir)
    if capacity == 0:
        return seen + values.size

    filled = min(seen, capacity)
    if filled < capacity:
        n_fill = min(capacity - filled, values.size)
        reservoir[filled:filled + n_fill] = values[:n_fill]
        seen += n_fill
        values = values[n_fill:]

    if values.size:
        positions = np.arange(seen + 1, seen + values.size + 1, dtype=np.float64)
        replacement_indices = (rng.random(values.size) * positions).astype(np.int64)
        replace_mask = replacement_indices < capacity
        reservoir[replacement_indices[replace_mask]] = values[replace_mask]
        seen += values.size

    return seen


def collect_factor_background(factor_paths, block_size, max_points, seed):
    rng = np.random.default_rng(seed)
    samples = {}
    valid_counts = {}

    for name, path in factor_paths.items():
        with rasterio.open(path) as src:
            reservoir = np.empty(max_points, dtype=np.float64)
            seen = 0
            for window in iter_windows(src.width, src.height, block_size):
                data = src.read(1, window=window)
                values = valid_raster_values(data, src.nodata)
                seen = update_reservoir(reservoir, seen, values, rng)

        count = min(seen, max_points)
        samples[name] = reservoir[:count].copy()
        valid_counts[name] = int(seen)

    return samples, valid_counts


def add_region_classes(df):
    y_col = "lat" if "lat" in df.columns else "y"
    y_min = df[y_col].min()
    y_max = df[y_col].max()
    y_norm = (df[y_col] - y_min) / (y_max - y_min + 1e-12)
    df["north_south_band"] = np.select(
        [y_norm >= 2.0 / 3.0, y_norm <= 1.0 / 3.0],
        ["north", "south"],
        default="central",
    )

    thresholds = {}
    rugged_proxy = np.zeros(len(df), dtype=bool)
    flat_proxy = np.ones(len(df), dtype=bool)

    if "slope" in df.columns:
        slope = df["slope"].to_numpy(dtype=np.float64)
        slope_q25, slope_q75 = np.nanpercentile(slope, [25, 75])
        thresholds["slope_q25"] = float(slope_q25)
        thresholds["slope_q75"] = float(slope_q75)
        rugged_proxy |= slope >= slope_q75
        flat_proxy &= slope <= slope_q25

    if "elevation" in df.columns:
        elevation = df["elevation"].to_numpy(dtype=np.float64)
        elev_q25, elev_q75 = np.nanpercentile(elevation, [25, 75])
        thresholds["elevation_q25"] = float(elev_q25)
        thresholds["elevation_q75"] = float(elev_q75)
        rugged_proxy |= elevation >= elev_q75
        flat_proxy &= elevation <= elev_q25

    if not thresholds:
        df["terrain_proxy"] = "not_available"
    else:
        df["terrain_proxy"] = np.select(
            [rugged_proxy, flat_proxy],
            ["rugged_proxy", "flat_proxy"],
            default="intermediate_proxy",
        )

    df["macro_region"] = df["north_south_band"] + "_" + df["terrain_proxy"]
    return thresholds


def write_region_stats(df, output_dir):
    group_cols = ["label_name", "north_south_band", "terrain_proxy", "macro_region"]
    stats = (
        df.groupby(group_cols, dropna=False)
        .size()
        .reset_index(name="count")
    )
    totals = df.groupby("label_name").size().to_dict()
    stats["percentage_within_label"] = stats.apply(
        lambda row: row["count"] / totals[row["label_name"]] * 100.0,
        axis=1,
    )
    stats.to_csv(os.path.join(output_dir, "dwss_region_stats.csv"), index=False)
    return stats


def write_factor_stats(df, output_dir):
    factor_cols = [col for col in FACTOR_KEYWORDS if col in df.columns]
    rows = []
    for factor in factor_cols:
        for label_name, group in df.groupby("label_name"):
            values = group[factor].dropna().to_numpy(dtype=np.float64)
            if values.size == 0:
                continue
            rows.append({
                "factor": factor,
                "label_name": label_name,
                "count": int(values.size),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "q25": float(np.percentile(values, 25)),
                "median": float(np.percentile(values, 50)),
                "q75": float(np.percentile(values, 75)),
                "max": float(np.max(values)),
            })
    with open(os.path.join(output_dir, "dwss_factor_stats.csv"), "w", newline="", encoding="utf-8") as f:
        fieldnames = ["factor", "label_name", "count", "mean", "std", "min", "q25", "median", "q75", "max"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_background_factor_stats(background_values, valid_counts, output_dir):
    rows = []
    for factor, values in background_values.items():
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        rows.append({
            "factor": factor,
            "valid_pixel_count": int(valid_counts.get(factor, values.size)),
            "sampled_pixel_count": int(values.size),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "q25": float(np.percentile(values, 25)),
            "median": float(np.percentile(values, 50)),
            "q75": float(np.percentile(values, 75)),
            "max": float(np.max(values)),
        })

    with open(os.path.join(output_dir, "dwss_factor_background_stats.csv"), "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "factor", "valid_pixel_count", "sampled_pixel_count",
            "mean", "std", "min", "q25", "median", "q75", "max",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def ks_distance(values_a, values_b):
    values_a = np.sort(finite_series(values_a))
    values_b = np.sort(finite_series(values_b))
    if values_a.size == 0 or values_b.size == 0:
        return np.nan

    grid = np.sort(np.unique(np.concatenate([values_a, values_b])))
    cdf_a = np.searchsorted(values_a, grid, side="right") / values_a.size
    cdf_b = np.searchsorted(values_b, grid, side="right") / values_b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def write_factor_background_comparison(df, background_values, output_dir):
    rows = []
    for factor in FACTOR_KEYWORDS:
        if factor not in df.columns or factor not in background_values:
            continue
        background = finite_series(background_values[factor])
        if background.size == 0:
            continue
        background_quantiles = np.percentile(background, [5, 25, 50, 75, 95])
        for label_name, group in df.groupby("label_name"):
            sample = finite_series(group[factor].to_numpy(dtype=np.float64))
            if sample.size == 0:
                continue
            sample_quantiles = np.percentile(sample, [5, 25, 50, 75, 95])
            rows.append({
                "factor": factor,
                "sample_group": label_name,
                "sample_count": int(sample.size),
                "background_count": int(background.size),
                "sample_mean": float(np.mean(sample)),
                "background_mean": float(np.mean(background)),
                "mean_difference": float(np.mean(sample) - np.mean(background)),
                "sample_median": float(sample_quantiles[2]),
                "background_median": float(background_quantiles[2]),
                "median_difference": float(sample_quantiles[2] - background_quantiles[2]),
                "sample_q05": float(sample_quantiles[0]),
                "background_q05": float(background_quantiles[0]),
                "sample_q25": float(sample_quantiles[1]),
                "background_q25": float(background_quantiles[1]),
                "sample_q75": float(sample_quantiles[3]),
                "background_q75": float(background_quantiles[3]),
                "sample_q95": float(sample_quantiles[4]),
                "background_q95": float(background_quantiles[4]),
                "ks_distance": ks_distance(sample, background),
            })

    with open(os.path.join(output_dir, "dwss_factor_background_comparison.csv"), "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "factor", "sample_group", "sample_count", "background_count",
            "sample_mean", "background_mean", "mean_difference",
            "sample_median", "background_median", "median_difference",
            "sample_q05", "background_q05", "sample_q25", "background_q25",
            "sample_q75", "background_q75", "sample_q95", "background_q95",
            "ks_distance",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_summary(df, thresholds, output_dir):
    neg = df[df["label"] == 2]
    pos = df[df["label"] == 1]
    summary = {
        "total_landslide_samples": int(len(pos)),
        "total_non_landslide_samples": int(len(neg)),
        "terrain_thresholds": thresholds,
    }

    for label_name, group in (("landslide", pos), ("non_landslide", neg)):
        total = max(len(group), 1)
        region_rows = []
        grouped = group.groupby(["north_south_band", "terrain_proxy", "macro_region"], dropna=False)
        for (band, terrain, macro_region), region_group in grouped:
            region_rows.append({
                "north_south_band": band,
                "terrain_proxy": terrain,
                "macro_region": macro_region,
                "count": int(len(region_group)),
                "percentage_within_label": float(len(region_group) / total * 100.0),
            })
        region_rows.sort(key=lambda item: (item["north_south_band"], item["terrain_proxy"]))
        summary[f"{label_name}_region_distribution"] = region_rows

    with open(os.path.join(output_dir, "dwss_distribution_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def finite_series(values):
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def make_histogram_bins(arrays, n_bins=44):
    finite_arrays = [finite_series(values) for values in arrays if len(values) > 0]
    finite_arrays = [values for values in finite_arrays if values.size > 0]
    if not finite_arrays:
        return None
    combined = np.concatenate(finite_arrays)
    vmin = float(np.nanmin(combined))
    vmax = float(np.nanmax(combined))
    if np.isclose(vmin, vmax):
        pad = max(abs(vmin) * 0.05, 1e-6)
        vmin -= pad
        vmax += pad
    return np.linspace(vmin, vmax, n_bins + 1)


def plot_factor_distribution_axis(ax, factor, df, background_values, colors):
    arrays_for_bins = []
    background = finite_series(background_values.get(factor, np.array([], dtype=np.float64)))
    if background.size:
        arrays_for_bins.append(background)
    for _, group in df.groupby("label_name"):
        arrays_for_bins.append(group[factor].dropna().to_numpy(dtype=np.float64))
    bins = make_histogram_bins(arrays_for_bins)
    if bins is None:
        return False

    if background.size:
        ax.hist(
            background,
            bins=bins,
            density=True,
            color="#98a2b3",
            alpha=0.22,
            edgecolor="white",
            linewidth=0.25,
            label="Factor valid pixels (background)",
            zorder=1,
        )
        ax.hist(
            background,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.8,
            color="#101828",
            label="_nolegend_",
            zorder=3,
        )

    for label_name, group in df.groupby("label_name"):
        values = group[factor].dropna().to_numpy(dtype=np.float64)
        if values.size == 0:
            continue
        linestyle = "-" if label_name == "landslide" else (0, (4, 2))
        ax.hist(
            values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2.2,
            linestyle=linestyle,
            color=colors.get(label_name, "#344054"),
            label=f"{LABEL_DISPLAY.get(label_name, label_name)} samples",
            zorder=4 if label_name == "non_landslide" else 5,
        )

    if background.size:
        annotations = []
        for label_name in ("landslide", "non_landslide"):
            if label_name not in set(df["label_name"]):
                continue
            values = df.loc[df["label_name"] == label_name, factor].dropna().to_numpy(dtype=np.float64)
            if values.size:
                annotations.append(f"{LABEL_DISPLAY[label_name]} KS={ks_distance(values, background):.3f}")
        if annotations:
            ax.text(
                0.98,
                0.96,
                "\n".join(annotations),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="#344054",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#d0d5dd", "alpha": 0.88},
            )

    ax.set_xlabel(f"{factor.title()} value")
    ax.set_ylabel("Probability density")
    ax.set_title(f"{factor.title()} distribution", weight="bold", pad=8)
    return True


def plot_factor_background_overview(df, background_values, output_dir, plt):
    factor_cols = [factor for factor in FACTOR_KEYWORDS if factor in df.columns]
    if not factor_cols:
        return

    colors = {"landslide": "#d73027", "non_landslide": "#4575b4"}
    n_cols = 2
    n_rows = int(np.ceil(len(factor_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12.4, 4.2 * n_rows), squeeze=False)
    used_axes = []

    for ax, factor in zip(axes.ravel(), factor_cols):
        if plot_factor_distribution_axis(ax, factor, df, background_values, colors):
            used_axes.append(ax)

    for ax in axes.ravel()[len(factor_cols):]:
        ax.axis("off")

    if used_axes:
        handles, labels = used_axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, os.path.join(output_dir, "dwss_factor_background_vs_samples.png"))
    plt.close(fig)


def ecdf_xy(values, max_points=5000):
    values = np.sort(finite_series(values))
    if values.size == 0:
        return values, values
    if values.size > max_points:
        indices = np.unique(np.linspace(0, values.size - 1, max_points).astype(np.int64))
        return values[indices], (indices + 1) / values.size
    return values, np.arange(1, values.size + 1, dtype=np.float64) / values.size


def plot_factor_background_ecdf(df, background_values, output_dir, plt):
    factor_cols = [factor for factor in FACTOR_KEYWORDS if factor in df.columns]
    if not factor_cols:
        return

    colors = {"landslide": "#d73027", "non_landslide": "#4575b4"}
    n_cols = 2
    n_rows = int(np.ceil(len(factor_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12.4, 4.2 * n_rows), squeeze=False)
    used_axes = []

    for ax, factor in zip(axes.ravel(), factor_cols):
        background = background_values.get(factor, np.array([], dtype=np.float64))
        x_bg, y_bg = ecdf_xy(background)
        if x_bg.size:
            ax.plot(
                x_bg,
                y_bg,
                color="#101828",
                linewidth=1.9,
                label="Factor valid pixels (background)",
            )
        for label_name, group in df.groupby("label_name"):
            x_sample, y_sample = ecdf_xy(group[factor].dropna().to_numpy(dtype=np.float64))
            if not x_sample.size:
                continue
            linestyle = "-" if label_name == "landslide" else (0, (4, 2))
            ax.plot(
                x_sample,
                y_sample,
                color=colors.get(label_name, "#344054"),
                linestyle=linestyle,
                linewidth=2.0,
                label=f"{LABEL_DISPLAY.get(label_name, label_name)} samples",
            )
        ax.set_xlabel(f"{factor.title()} value")
        ax.set_ylabel("Empirical cumulative probability")
        ax.set_ylim(0, 1)
        ax.set_title(f"{factor.title()} ECDF", weight="bold", pad=8)
        used_axes.append(ax)

    for ax in axes.ravel()[len(factor_cols):]:
        ax.axis("off")

    if used_axes:
        handles, labels = used_axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, os.path.join(output_dir, "dwss_factor_background_vs_samples_ecdf.png"))
    plt.close(fig)


def plot_distribution(df, background_values, output_dir, max_points, seed):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots.")
        return

    setup_plot_style(plt)
    plot_df = df
    if len(plot_df) > max_points:
        plot_df = plot_df.sample(max_points, random_state=seed)

    colors = {"landslide": "#d73027", "non_landslide": "#4575b4"}
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    for label_name, group in plot_df.groupby("label_name"):
        ax.scatter(
            group["lon"],
            group["lat"],
            s=8 if label_name == "landslide" else 5,
            alpha=0.68 if label_name == "landslide" else 0.45,
            c=colors.get(label_name, "gray"),
            edgecolors="none",
            label=label_name.replace("_", " ").title(),
        )
    lat_min, lat_max = df["lat"].min(), df["lat"].max()
    for frac, label in ((1.0 / 3.0, "South/Central"), (2.0 / 3.0, "Central/North")):
        y = lat_min + (lat_max - lat_min) * frac
        ax.axhline(y, color="#667085", linewidth=0.9, linestyle="--", alpha=0.8)
        ax.text(df["lon"].min(), y, f" {label}", va="bottom", ha="left", fontsize=8, color="#344054")
    ax.set_aspect("equal", adjustable="box")
    add_wgs84_axes(ax, "DWSS sample locations")
    add_panel_label(ax, "WGS84")
    ax.legend(markerscale=2.2, loc="best")
    fig.tight_layout()
    save_figure(fig, os.path.join(output_dir, "dwss_sample_spatial_distribution.png"))
    plt.close()

    for factor in FACTOR_KEYWORDS:
        if factor not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(7.6, 4.8))
        plot_factor_distribution_axis(ax, factor, df, background_values, colors)
        ax.legend()
        fig.tight_layout()
        save_figure(fig, os.path.join(output_dir, f"dwss_{factor}_histogram.png"))
        plt.close()

    plot_factor_background_overview(df, background_values, output_dir, plt)
    plot_factor_background_ecdf(df, background_values, output_dir, plt)
    plot_non_landslide_region_distribution(df, output_dir, max_points, seed, plt)


def plot_non_landslide_region_distribution(df, output_dir, max_points, seed, plt):
    neg = df[df["label_name"] == "non_landslide"].copy()
    if neg.empty:
        return

    plot_neg = neg
    if len(plot_neg) > max_points:
        plot_neg = plot_neg.sample(max_points, random_state=seed)

    terrain_colors = {
        "flat_proxy": "#2b8cbe",
        "intermediate_proxy": "#a6bddb",
        "rugged_proxy": "#ef8a62",
        "not_available": "#98a2b3",
    }
    terrain_labels = {
        "flat_proxy": "Flat proxy",
        "intermediate_proxy": "Intermediate",
        "rugged_proxy": "Rugged proxy",
        "not_available": "Not available",
    }

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8), gridspec_kw={"width_ratios": [1.25, 1.0]})
    ax_map, ax_bar = axes

    for terrain, group in plot_neg.groupby("terrain_proxy"):
        ax_map.scatter(
            group["lon"],
            group["lat"],
            s=6,
            alpha=0.46,
            c=terrain_colors.get(terrain, "#98a2b3"),
            edgecolors="none",
            label=terrain_labels.get(terrain, terrain),
        )

    lat_min, lat_max = neg["lat"].min(), neg["lat"].max()
    for frac in (1.0 / 3.0, 2.0 / 3.0):
        ax_map.axhline(lat_min + (lat_max - lat_min) * frac, color="#667085", linewidth=0.9, linestyle="--")
    ax_map.set_aspect("equal", adjustable="box")
    add_wgs84_axes(ax_map, "Non-landslide sample locations")
    add_panel_label(ax_map, "A")
    ax_map.legend(markerscale=1.8, fontsize=8, loc="best")

    counts = (
        neg.groupby(["north_south_band", "terrain_proxy"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=["north", "central", "south"], fill_value=0)
    )
    order = ["flat_proxy", "intermediate_proxy", "rugged_proxy", "not_available"]
    bottom = np.zeros(len(counts), dtype=np.float64)
    x = np.arange(len(counts))
    total_negative = max(len(neg), 1)
    for terrain in order:
        if terrain not in counts.columns:
            continue
        pct = counts[terrain].to_numpy(dtype=np.float64) / total_negative * 100.0
        ax_bar.bar(
            x,
            pct,
            bottom=bottom,
            width=0.62,
            color=terrain_colors.get(terrain, "#98a2b3"),
            edgecolor="white",
            linewidth=0.6,
            label=terrain_labels.get(terrain, terrain),
        )
        bottom += np.nan_to_num(pct)

    band_totals = counts.sum(axis=1).to_numpy(dtype=np.float64)
    band_pct = band_totals / total_negative * 100.0
    y_limit = max(10.0, float(np.nanmax(band_pct)) * 1.24)
    for idx, (count, pct) in enumerate(zip(band_totals, band_pct)):
        ax_bar.text(
            idx,
            pct + y_limit * 0.025,
            f"{int(count):,}\n{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#344054",
        )
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(["North", "Central", "South"])
    ax_bar.set_ylim(0, y_limit)
    ax_bar.set_ylabel("Share of non-landslide samples (%)")
    ax_bar.set_title("Non-landslide north-south distribution", weight="bold", pad=10)
    add_panel_label(ax_bar, "B")
    ax_bar.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    save_figure(fig, os.path.join(output_dir, "dwss_non_landslide_north_south_distribution.png"))
    plt.close(fig)


def run_analysis(args):
    params = read_xml_params(args.xml)
    label_path = normalize_path(args.label_path) if args.label_path else find_first_tif(params["input_labels_dir"])
    factors_dir = normalize_path(args.factors_dir) if args.factors_dir else params["input_factors_dir"]
    train_output = params.get("train_output", ".")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = normalize_path(args.output_dir) if args.output_dir else os.path.join(
        train_output,
        f"DWSS_Diagnostics_{timestamp}",
    )
    os.makedirs(output_dir, exist_ok=True)

    factor_paths = pick_factor_files(factors_dir)
    print(f"Label raster: {label_path}")
    print(f"Detected diagnostic factors: {factor_paths}")

    df = collect_sample_table(label_path, factor_paths, args.block_size)
    background_values, background_counts = collect_factor_background(
        factor_paths,
        args.block_size,
        args.max_background_points,
        args.seed,
    )
    thresholds = add_region_classes(df)
    region_stats = write_region_stats(df, output_dir)
    factor_stats = write_factor_stats(df, output_dir)
    background_stats = write_background_factor_stats(background_values, background_counts, output_dir)
    comparison_stats = write_factor_background_comparison(df, background_values, output_dir)
    summary = write_summary(df, thresholds, output_dir)

    if args.write_points:
        df.to_csv(os.path.join(output_dir, "dwss_sample_points.csv"), index=False)

    plot_distribution(df, background_values, output_dir, args.max_plot_points, args.seed)

    print(region_stats)
    print(json.dumps(summary, indent=2))
    print(f"Factor stat rows: {len(factor_stats)}")
    print(f"Background factor stat rows: {len(background_stats)}")
    print(f"Background comparison rows: {len(comparison_stats)}")
    print(f"DWSS diagnostics saved to: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze DWSS positive/negative sample distribution.")
    parser.add_argument("xml", help="Path to Landslide_susceptibility_mapping.xml")
    parser.add_argument("--label-path", default=None, help="Override label raster path.")
    parser.add_argument("--factors-dir", default=None, help="Override factor raster directory.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--max-plot-points", type=int, default=200000)
    parser.add_argument(
        "--max-background-points",
        type=int,
        default=300000,
        help="Maximum valid pixels sampled from each factor raster for background distribution plots.",
    )
    parser.add_argument("--seed", type=int, default=20250609)
    parser.add_argument("--write-points", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_analysis(parse_args())
