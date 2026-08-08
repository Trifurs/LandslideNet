"""Manuscript-ready comparison tables and rank-capture curves."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from .progress import track


def rank_capture_curve(labels, probabilities, grid=None):
    """Cumulative positive capture versus inspected sample fraction."""
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if grid is None:
        grid = np.linspace(0.0, 1.0, 201)
    if len(labels) == 0 or np.count_nonzero(labels == 1) == 0:
        return np.asarray(grid), np.full(len(grid), np.nan), float("nan")
    order = np.argsort(-probabilities, kind="mergesort")
    cumulative = np.cumsum(labels[order] == 1) / np.count_nonzero(labels == 1)
    fraction = np.arange(1, len(labels) + 1, dtype=np.float64) / len(labels)
    source_x = np.concatenate(([0.0], fraction))
    source_y = np.concatenate(([0.0], cumulative))
    curve = np.interp(grid, source_x, source_y)
    area = np.trapezoid(curve, grid) if hasattr(np, "trapezoid") else np.trapz(curve, grid)
    return np.asarray(grid), curve, float(area)


def write_rate_curve_reports(metrics, output_dir):
    """Write equal-region SRC/PRC summaries and figures for every model/arm.

    SRC here is the inner-training success-rate rank-capture curve. PRC here is
    the held-out prediction-rate rank-capture curve used in susceptibility
    literature; it is deliberately named separately from standard precision-
    recall AUC (`pr_auc`) in the metric tables.
    """
    if not metrics:
        return
    output_dir = Path(output_dir)
    grid = np.linspace(0.0, 1.0, 201)
    detailed_rows = []
    curve_rows = []
    for row in track(
        metrics,
        total=len(metrics),
        desc="Generating SRC/PRC data",
        unit="result",
    ):
        test_path = output_dir / row["prediction_file"]
        paths = {
            "success_rate": test_path.with_name("success_predictions.npz"),
            "prediction_rate": test_path,
        }
        for curve_type, path in paths.items():
            if not path.exists():
                continue
            with np.load(path) as prediction:
                x, y, area = rank_capture_curve(
                    prediction["label"], prediction["probability"], grid
                )
            identity = {
                "fold": int(row["fold"]),
                "test_region": int(row["test_region"]),
                "model": row["model"],
                "model_display_name": row["model_display_name"],
                "sampling_method": row["sampling_method"],
                "curve_type": curve_type,
            }
            detailed_rows.append({**identity, "rank_capture_auc": area})
            curve_rows.extend(
                {**identity, "sample_fraction": float(xv), "capture_fraction": float(yv)}
                for xv, yv in zip(x, y)
            )
    if not detailed_rows:
        return

    detailed = pd.DataFrame(detailed_rows)
    curves = pd.DataFrame(curve_rows)
    detailed.to_csv(output_dir / "rate_curve_auc_detailed.csv", index=False)
    group_columns = [
        "model", "model_display_name", "sampling_method", "curve_type"
    ]
    summary = detailed.groupby(group_columns)["rank_capture_auc"].agg(
        ["mean", "std", "min", "max", "count"]
    ).reset_index()
    summary.rename(columns={"mean": "equal_region_mean"}, inplace=True)
    summary.to_csv(output_dir / "rate_curve_auc_region_summary.csv", index=False)
    mean_curves = curves.groupby(
        group_columns + ["sample_fraction"], as_index=False
    )["capture_fraction"].mean()
    mean_curves.to_csv(output_dir / "rate_curves_equal_region_mean.csv", index=False)

    manuscript = summary.pivot_table(
        index=["model", "model_display_name", "sampling_method"],
        columns="curve_type",
        values="equal_region_mean",
    ).reset_index().rename(columns={
        "success_rate": "SRC_AUC_equal_region_mean",
        "prediction_rate": "PRC_AUC_equal_region_mean",
    })
    manuscript.to_csv(output_dir / "table_src_prc_auc.csv", index=False)

    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plot_jobs = [
        (method, curve_type, title)
        for method in sorted(mean_curves["sampling_method"].unique())
        for curve_type, title in (
            ("success_rate", "Success-rate curves (inner training)"),
            ("prediction_rate", "Prediction-rate curves (held-out regions)"),
        )
    ]
    for method, curve_type, title in track(
        plot_jobs,
        total=len(plot_jobs),
        desc="Plotting SRC/PRC curves",
        unit="figure",
    ):
        selected = mean_curves[
            (mean_curves["sampling_method"] == method)
            & (mean_curves["curve_type"] == curve_type)
        ]
        if selected.empty:
            continue
        fig, ax = plt.subplots(figsize=(8.2, 6.2))
        for (_name, display_name), group in selected.groupby(
            ["model", "model_display_name"]
        ):
            ax.plot(
                group["sample_fraction"], group["capture_fraction"],
                linewidth=1.6, label=display_name,
            )
        ax.plot([0, 1], [0, 1], linestyle="--", color="0.5", linewidth=1)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Inspected sample fraction",
               ylabel="Captured landslide fraction", title=f"{title} — {method.upper()}")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(
            output_dir / f"{curve_type}_curves_{method}.png",
            dpi=300, bbox_inches="tight",
        )
        plt.close(fig)
