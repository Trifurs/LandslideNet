"""Classical comparison models under the same regional protocol as LandslideNet."""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np

from Tools.model_comparisons.machine_learning import (
    build_classical_model as _build_classical_model,
    dependency_status as _dependency_status,
    require_dependencies as _require_dependencies,
)

from .model_registry import MODEL_SPECS, canonical_model_name
from .progress import console, metric_line, timed_task, track
from .training import build_metrics, find_best_threshold, stratified_bootstrap_intervals


def dependency_status(model_names) -> dict:
    names = [canonical_model_name(name) for name in model_names]
    return _dependency_status(names)


def require_dependencies(model_names):
    names = [canonical_model_name(name) for name in model_names]
    return _require_dependencies(names)


def build_classical_model(name: str, seed: int, n_jobs: int, iterations: int):
    name = canonical_model_name(name)
    if MODEL_SPECS[name].family != "classical":
        raise ValueError(f"{name} is not a classical baseline.")
    return _build_classical_model(name, seed, n_jobs, iterations)


def _fit(model, name, train_x, train_y, train_weight, val_x, val_y, iterations):
    if name == "catboost":
        model.fit(
            train_x, train_y, sample_weight=train_weight,
            eval_set=(val_x, val_y), use_best_model=True,
            early_stopping_rounds=50,
            verbose=max(1, int(iterations) // 20),
        )
    elif name == "lightgbm":
        import lightgbm as lgb
        model.fit(
            train_x, train_y, sample_weight=train_weight,
            eval_set=[(val_x, val_y)], eval_metric="auc",
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(period=max(1, int(iterations) // 20)),
            ],
        )
    else:
        model.fit(train_x, train_y, sample_weight=train_weight)


def _probabilities(model, features):
    probabilities = model.predict_proba(features)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise RuntimeError("Binary classifier did not return two probability columns.")
    return np.asarray(probabilities[:, 1], dtype=np.float64)


def spatial_tile_ids(points: np.ndarray, tile_size: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.int64)
    tile_rows = points[:, 0] // int(tile_size)
    tile_cols = points[:, 1] // int(tile_size)
    pairs = np.column_stack((tile_rows, tile_cols))
    _unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
    return inverse.astype(np.int64, copy=False)


def train_classical_model(
    name: str,
    arrays: dict,
    points: dict,
    train_weights: np.ndarray,
    output_dir: str,
    seed: int,
    threshold_metric: str,
    crop_size: int,
    bootstrap_replicates: int,
    bootstrap_confidence: float,
    bootstrap_seed: int,
    n_jobs: int = 1,
    iterations: int = 500,
) -> dict:
    """Fit on train, select threshold on validation, touch test exactly once."""
    name = canonical_model_name(name)
    os.makedirs(output_dir, exist_ok=True)
    train_x, train_y = arrays["train"]
    val_x, val_y = arrays["val"]
    spec = MODEL_SPECS[name]
    progress = track(
        total=6,
        desc=f"{spec.display_name} 训练流程",
        unit="stage",
        leave=True,
    )
    progress.set_postfix_str("构建模型", refresh=False)
    model = build_classical_model(name, seed, n_jobs, iterations)
    progress.update(1)
    progress.set_postfix_str("拟合训练集", refresh=True)
    with timed_task(
        f"{spec.display_name} 拟合（train={len(train_y):,}, val={len(val_y):,}）"
    ):
        _fit(
            model, name, train_x, train_y, train_weights, val_x, val_y, iterations
        )
    progress.update(1)

    # Every selection decision is complete before test features are evaluated.
    progress.set_postfix_str("验证集阈值选择", refresh=False)
    val_probability = _probabilities(model, val_x)
    threshold = find_best_threshold(val_y, val_probability, metric=threshold_metric)
    val_prediction = (val_probability >= threshold).astype(np.int64)
    _val_text, val_metrics = build_metrics(
        val_y, val_prediction, val_probability, float("nan"),
        "Validation", threshold,
    )
    progress.update(1)
    console(
        f"{spec.display_name} 验证结果 | {metric_line(val_metrics)} | "
        f"threshold={threshold:.3f}"
    )

    progress.set_postfix_str("训练集成功率评估", refresh=False)
    train_probability = _probabilities(model, train_x)
    np.savez_compressed(
        os.path.join(output_dir, "success_predictions.npz"),
        row=points["train"][:, 0],
        col=points["train"][:, 1],
        label=train_y,
        probability=train_probability,
    )
    _success_text, success_metrics = build_metrics(
        train_y,
        (train_probability >= threshold).astype(np.int64),
        train_probability,
        float("nan"),
        "Inner-training success-rate evaluation",
        threshold,
    )
    progress.update(1)

    progress.set_postfix_str("测试区单次评估", refresh=False)
    test_x, test_y = arrays["test"]
    test_probability = _probabilities(model, test_x)
    test_prediction = (test_probability >= threshold).astype(np.int64)
    _test_text, metrics = build_metrics(
        test_y, test_prediction, test_probability, float("nan"),
        "Test", threshold,
    )
    metrics.update(stratified_bootstrap_intervals(
        test_y,
        test_probability,
        threshold,
        replicates=bootstrap_replicates,
        confidence=bootstrap_confidence,
        seed=bootstrap_seed,
        block_ids=spatial_tile_ids(points["test"], crop_size),
        progress_desc=f"{spec.display_name} 空间块 Bootstrap",
    ))
    metrics.update({
        "best_epoch": None,
        "validation_selection_metric": threshold_metric,
        "validation_selection_score": float(val_metrics[threshold_metric]),
        "threshold_metric": threshold_metric,
        "threshold_selected_on_validation": float(threshold),
        "test_evaluations": 1,
        "success_rate_auc": float(success_metrics["auc"]),
        "success_rate_pr_auc": float(success_metrics["pr_auc"]),
    })
    progress.update(1)
    console(
        f"{spec.display_name} 测试结果 | {metric_line(metrics, ('auc', 'pr_auc', 'f1', 'kappa', 'precision', 'recall'))}"
    )

    progress.set_postfix_str("保存模型与预测", refresh=False)
    joblib.dump(model, os.path.join(output_dir, "best_model.joblib"))
    metadata_path = Path(output_dir, "model_metadata.json")
    existing_metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists() else {}
    )
    metadata = {
        **existing_metadata,
        "model": MODEL_SPECS[name].to_dict(),
        "seed": int(seed),
        "training_samples": int(len(train_y)),
        "validation_samples": int(len(val_y)),
        "test_samples": int(len(test_y)),
        "threshold_selected_on_validation": float(threshold),
        "test_evaluations": 1,
        "estimator_parameters": model.get_params(deep=False),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    np.savez_compressed(
        os.path.join(output_dir, "test_predictions.npz"),
        row=points["test"][:, 0],
        col=points["test"][:, 1],
        label=test_y,
        probability=test_probability,
    )
    progress.update(1)
    progress.set_postfix_str("完成", refresh=False)
    progress.close()
    return metrics
