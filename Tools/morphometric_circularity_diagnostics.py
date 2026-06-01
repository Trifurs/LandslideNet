#!/usr/bin/env python3
import argparse
import csv
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


def find_factor_by_keywords(factors_dir, keywords):
    for path in list_factor_files(factors_dir):
        stem = Path(path).stem.lower()
        compact = stem.replace("_", "").replace("-", "").replace(" ", "")
        if any(keyword in stem or keyword in compact for keyword in keywords):
            return path
    return None


def find_morphometric_map(params, cli_path):
    if cli_path:
        return normalize_path(cli_path)

    for key in ("morphometric_map", "morphometric_susceptibility_map"):
        if key in params and params[key]:
            return params[key]

    candidates = []
    mosaic = params.get("mosaic_map")
    train_output = params.get("train_output")
    if mosaic:
        mosaic_path = Path(mosaic)
        candidates.append(mosaic_path.parent / "Morphometric_Susceptibility.tif")
        for parent in mosaic_path.parents:
            candidates.append(parent / "Morphometric" / "Morphometric_Susceptibility.tif")
            candidates.append(parent / "All_LSMs" / "Morphometric" / "Morphometric_Susceptibility.tif")
    if train_output:
        train_path = Path(train_output)
        candidates.append(train_path.parent / "Morphometric" / "Morphometric_Susceptibility.tif")
        candidates.append(train_path.parent / "All_LSMs" / "Morphometric" / "Morphometric_Susceptibility.tif")

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        "Morphometric map was not found. Add <morphometric_map> to the XML "
        "or pass --morphometric-map."
    )


def check_alignment(base_src, named_sources):
    aligned = {}
    for name, src in named_sources.items():
        same_shape = src.width == base_src.width and src.height == base_src.height
        same_transform = src.transform.almost_equals(base_src.transform)
        if same_shape and same_transform:
            aligned[name] = src
        else:
            src.close()
            raise ValueError(f"Raster grid mismatch for {name}; please resample it to the LSM grid first.")
    return aligned


def iter_windows(width, height, block_size):
    for row in range(0, height, block_size):
        for col in range(0, width, block_size):
            yield Window(col, row, min(block_size, width - col), min(block_size, height - row))


def valid_mask_for(data, nodata):
    mask = np.isfinite(data)
    if nodata is not None:
        mask &= ~np.isclose(data, nodata)
    return mask


def sample_rasters(raster_paths, max_samples, block_size, seed):
    rng = np.random.default_rng(seed)
    names = list(raster_paths.keys())
    base_name = names[0]
    records = []

    with rasterio.open(raster_paths[base_name]) as base_src:
        sources = {}
        for name in names[1:]:
            sources[name] = rasterio.open(raster_paths[name])
        sources = check_alignment(base_src, sources)

        n_windows = int(np.ceil(base_src.width / block_size) * np.ceil(base_src.height / block_size))
        samples_per_window = max(1000, int(np.ceil(max_samples / max(n_windows, 1) * 2)))

        for window in iter_windows(base_src.width, base_src.height, block_size):
            arrays = {base_name: base_src.read(1, window=window).astype(np.float64)}
            nodata = {base_name: base_src.nodata}
            for name, src in sources.items():
                arrays[name] = src.read(1, window=window).astype(np.float64)
                nodata[name] = src.nodata

            valid = np.ones(arrays[base_name].shape, dtype=bool)
            for name, data in arrays.items():
                valid &= valid_mask_for(data, nodata[name])

            flat_idx = np.flatnonzero(valid.ravel())
            if flat_idx.size == 0:
                continue

            take = min(flat_idx.size, samples_per_window)
            chosen = rng.choice(flat_idx, size=take, replace=False)
            rel_rows, rel_cols = np.unravel_index(chosen, valid.shape)
            abs_rows = rel_rows + int(window.row_off)
            abs_cols = rel_cols + int(window.col_off)
            xs, ys = rasterio.transform.xy(base_src.transform, abs_rows, abs_cols)
            lons, lats = to_wgs84(xs, ys, base_src.crs)

            for i, flat_index in enumerate(chosen):
                row = {
                    "row": int(abs_rows[i]),
                    "col": int(abs_cols[i]),
                    "x": float(xs[i]),
                    "y": float(ys[i]),
                    "lon": float(lons[i]),
                    "lat": float(lats[i]),
                }
                for name, data in arrays.items():
                    row[name] = float(data.ravel()[flat_index])
                records.append(row)

        for src in sources.values():
            src.close()

    if not records:
        raise RuntimeError("No valid overlapping pixels were sampled.")

    df = pd.DataFrame.from_records(records)
    if len(df) > max_samples:
        df = df.sample(max_samples, random_state=seed).reset_index(drop=True)
    return df


def minmax(values):
    values = np.asarray(values, dtype=np.float64)
    vmin = np.nanmin(values)
    vmax = np.nanmax(values)
    return (values - vmin) / (vmax - vmin + 1e-12)


def linear_r2(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 3:
        return np.nan
    design = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ beta
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))


def pearson_corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid) < 3:
        return np.nan
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def residualize(y, controls):
    y = np.asarray(y, dtype=np.float64)
    controls = np.asarray(controls, dtype=np.float64)
    if controls.ndim == 1:
        controls = controls[:, None]
    valid = np.isfinite(y) & np.all(np.isfinite(controls), axis=1)
    residuals = np.full_like(y, np.nan, dtype=np.float64)
    design = np.column_stack([np.ones(np.sum(valid)), controls[valid]])
    beta, *_ = np.linalg.lstsq(design, y[valid], rcond=None)
    residuals[valid] = y[valid] - design @ beta
    return residuals


def spearman_corr(x, y):
    series_x = pd.Series(x)
    series_y = pd.Series(y)
    return float(series_x.corr(series_y, method="spearman"))


def compute_stats(df, control_names):
    lsm = minmax(df["landslidenet"].to_numpy())
    morph = minmax(df["morphometric"].to_numpy())

    stats = {
        "n_samples": len(df),
        "raw_pearson": pearson_corr(morph, lsm),
        "raw_r2_lsm_from_morphometric": linear_r2(morph, lsm),
        "raw_spearman": spearman_corr(morph, lsm),
    }

    if "slope" in df.columns:
        slope = minmax(df["slope"].to_numpy())
        stats["slope_only_r2_landslidenet"] = linear_r2(slope, lsm)
        stats["slope_only_r2_morphometric"] = linear_r2(slope, morph)
        lsm_resid = residualize(lsm, slope)
        morph_resid = residualize(morph, slope)
        partial = pearson_corr(morph_resid, lsm_resid)
        stats["partial_pearson_controlling_slope"] = partial
        stats["partial_r2_controlling_slope"] = partial ** 2 if np.isfinite(partial) else np.nan
        df["landslidenet_residual_slope"] = lsm_resid
        df["morphometric_residual_slope"] = morph_resid

    available_controls = [name for name in control_names if name in df.columns]
    if available_controls:
        controls = np.column_stack([minmax(df[name].to_numpy()) for name in available_controls])
        lsm_resid = residualize(lsm, controls)
        morph_resid = residualize(morph, controls)
        partial = pearson_corr(morph_resid, lsm_resid)
        suffix = "_".join(available_controls)
        stats[f"partial_pearson_controlling_{suffix}"] = partial
        stats[f"partial_r2_controlling_{suffix}"] = partial ** 2 if np.isfinite(partial) else np.nan

    return stats


def write_stats(stats, output_dir):
    with open(os.path.join(output_dir, "morphometric_circularity_stats.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in stats.items():
            writer.writerow([key, value])


def plot_results(df, output_dir, max_plot_points, seed):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots.")
        return

    setup_plot_style(plt)
    plot_df = df
    if len(plot_df) > max_plot_points:
        plot_df = plot_df.sample(max_plot_points, random_state=seed)

    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    x = minmax(plot_df["morphometric"])
    y = minmax(plot_df["landslidenet"])
    if "slope" in plot_df.columns:
        scatter = ax.scatter(
            x,
            y,
            c=minmax(plot_df["slope"]),
            s=6,
            alpha=0.35,
            cmap="viridis",
            edgecolors="none",
        )
        cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label("Normalized slope")
    else:
        ax.scatter(x, y, s=6, alpha=0.35, color="#2166ac", edgecolors="none")
    ax.plot([0, 1], [0, 1], color="#101828", linewidth=1.0, linestyle="--", label="1:1 line")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Morphometric susceptibility (normalized)")
    ax.set_ylabel("LandslideNet susceptibility (normalized)")
    ax.set_title("Susceptibility agreement", weight="bold", pad=10)
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_figure(fig, os.path.join(output_dir, "morphometric_vs_landslidenet_scatter.png"))
    plt.close()

    if {"landslidenet_residual_slope", "morphometric_residual_slope"}.issubset(plot_df.columns):
        fig, ax = plt.subplots(figsize=(6.2, 5.8))
        ax.scatter(
            plot_df["morphometric_residual_slope"],
            plot_df["landslidenet_residual_slope"],
            s=6,
            alpha=0.35,
            color="#2b8cbe",
            edgecolors="none",
        )
        ax.axhline(0, color="#667085", linewidth=0.8)
        ax.axvline(0, color="#667085", linewidth=0.8)
        ax.set_xlabel("Morphometric residual after slope control")
        ax.set_ylabel("LandslideNet residual after slope control")
        ax.set_title("Residual agreement after slope control", weight="bold", pad=10)
        fig.tight_layout()
        save_figure(fig, os.path.join(output_dir, "morphometric_slope_controlled_residuals.png"))
        plt.close()

    if {"lon", "lat"}.issubset(plot_df.columns):
        fig, ax = plt.subplots(figsize=(8.2, 6.4))
        delta = minmax(plot_df["landslidenet"]) - minmax(plot_df["morphometric"])
        vmax = np.nanpercentile(np.abs(delta), 98)
        vmax = max(float(vmax), 1e-6)
        scatter = ax.scatter(
            plot_df["lon"],
            plot_df["lat"],
            c=delta,
            s=7,
            alpha=0.72,
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            edgecolors="none",
        )
        ax.set_aspect("equal", adjustable="box")
        add_wgs84_axes(ax, "Spatial pattern of LandslideNet minus morphometric susceptibility")
        add_panel_label(ax, "WGS84")
        cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label("Normalized susceptibility difference")
        fig.tight_layout()
        save_figure(fig, os.path.join(output_dir, "morphometric_landslidenet_spatial_difference.png"))
        plt.close(fig)


def run_diagnostics(args):
    params = read_xml_params(args.xml)
    lsm_path = normalize_path(args.landslidenet_map) if args.landslidenet_map else params["mosaic_map"]
    morph_path = find_morphometric_map(params, args.morphometric_map)
    factors_dir = normalize_path(args.factors_dir) if args.factors_dir else params["input_factors_dir"]
    train_output = params.get("train_output", ".")

    slope_path = normalize_path(args.slope_raster) if args.slope_raster else find_factor_by_keywords(
        factors_dir,
        ("slope",),
    )

    raster_paths = {
        "landslidenet": lsm_path,
        "morphometric": morph_path,
    }
    if slope_path:
        raster_paths["slope"] = slope_path
    if args.flow_acc_raster:
        raster_paths["flow_acc"] = normalize_path(args.flow_acc_raster)
    if args.beta_raster:
        raster_paths["beta"] = normalize_path(args.beta_raster)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = normalize_path(args.output_dir) if args.output_dir else os.path.join(
        train_output,
        f"Morphometric_Circularity_{timestamp}",
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"Raster inputs: {raster_paths}")
    df = sample_rasters(raster_paths, args.max_samples, args.block_size, args.seed)
    control_names = ["slope"]
    if args.flow_acc_raster:
        control_names.append("flow_acc")
    if args.beta_raster:
        control_names.append("beta")

    stats = compute_stats(df, control_names)
    write_stats(stats, output_dir)
    if args.write_samples:
        df.to_csv(os.path.join(output_dir, "morphometric_circularity_samples.csv"), index=False)
    plot_results(df, output_dir, args.max_plot_points, args.seed)

    print(pd.DataFrame([{"metric": k, "value": v} for k, v in stats.items()]))
    print(f"Morphometric circularity diagnostics saved to: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose slope-driven circularity in morphometric validation.")
    parser.add_argument("xml", help="Path to Landslide_susceptibility_mapping.xml")
    parser.add_argument("--landslidenet-map", default=None)
    parser.add_argument("--morphometric-map", default=None)
    parser.add_argument("--factors-dir", default=None)
    parser.add_argument("--slope-raster", default=None)
    parser.add_argument("--flow-acc-raster", default=None)
    parser.add_argument("--beta-raster", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-samples", type=int, default=250000)
    parser.add_argument("--max-plot-points", type=int, default=60000)
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20250609)
    parser.add_argument("--write-samples", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_diagnostics(parse_args())
