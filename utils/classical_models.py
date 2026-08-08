"""Classical comparison models under the same regional protocol as LandslideNet."""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import log_loss

from models.classical import (
    build_classical_model as _build_classical_model,
    dependency_status as _dependency_status,
    require_dependencies as _require_dependencies,
)

from .model_registry import MODEL_SPECS, canonical_model_name
from .progress import console, metric_line, timed_task, track
from .training import (
    append_training_history,
    build_metrics,
    find_best_threshold,
    initialize_training_history,
    stratified_bootstrap_intervals,
)


def dependency_status(model_names) -> dict:
    names = [canonical_model_name(name) for name in model_names]
    return _dependency_status(names)


def require_dependencies(model_names):
    names = [canonical_model_name(name) for name in model_names]
    if not names:
        return
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
            eval_set=(val_x, val_y), use_best_model=False,
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


def _positive_column(probabilities, classes=(0, 1)):
    probabilities = np.asarray(probabilities)
    classes = np.asarray(classes)
    matches = np.flatnonzero(classes == 1)
    if probabilities.ndim != 2 or len(matches) != 1:
        raise RuntimeError("A training round did not expose a positive-class probability.")
    return np.asarray(probabilities[:, matches[0]], dtype=np.float64)


def _classical_round_count(model, name):
    if name in {"extra_trees", "random_forest"}:
        return len(model.estimators_)
    if name == "catboost":
        return int(model.tree_count_)
    if name == "lightgbm":
        return int(model.booster_.num_trees())
    return 1


def _iter_round_probabilities(model, name, train_x, val_x):
    """Yield cumulative train/validation probabilities for every fit round."""
    if name in {"extra_trees", "random_forest"}:
        train_sum = np.zeros(len(train_x), dtype=np.float64)
        val_sum = np.zeros(len(val_x), dtype=np.float64)
        for round_index, estimator in enumerate(model.estimators_, start=1):
            train_sum += _positive_column(
                estimator.predict_proba(train_x),
                estimator.classes_,
            )
            val_sum += _positive_column(
                estimator.predict_proba(val_x),
                estimator.classes_,
            )
            yield round_index, train_sum / round_index, val_sum / round_index
        return
    if name == "catboost":
        train_stages = model.staged_predict_proba(train_x)
        val_stages = model.staged_predict_proba(val_x)
        for round_index, (train_probability, val_probability) in enumerate(
            zip(train_stages, val_stages),
            start=1,
        ):
            yield (
                round_index,
                _positive_column(train_probability),
                _positive_column(val_probability),
            )
        return
    if name == "lightgbm":
        for round_index in range(1, _classical_round_count(model, name) + 1):
            yield (
                round_index,
                _positive_column(
                    model.predict_proba(train_x, num_iteration=round_index),
                    model.classes_,
                ),
                _positive_column(
                    model.predict_proba(val_x, num_iteration=round_index),
                    model.classes_,
                ),
            )
        return
    yield 1, _probabilities(model, train_x), _probabilities(model, val_x)


def write_classical_training_history(
    model,
    name,
    train_x,
    train_y,
    val_x,
    val_y,
    output_dir,
    selection_metric,
    threshold_metric,
    threshold_score_tolerance,
):
    """Write loss/F1/etc. for every meaningful estimator fitting round."""
    history_path = initialize_training_history(output_dir)
    total_rounds = _classical_round_count(model, name)
    learning_rate = model.get_params(deep=False).get("learning_rate", "")
    best_score = -np.inf
    best_round = 0
    round_iterator = track(
        _iter_round_probabilities(model, name, train_x, val_x),
        total=total_rounds,
        desc=f"{MODEL_SPECS[name].display_name} per-round history",
        unit="round",
    )
    for round_index, train_probability, val_probability in round_iterator:
        threshold_details = find_best_threshold(
            val_y,
            val_probability,
            metric=threshold_metric,
            score_tolerance=threshold_score_tolerance,
            return_details=True,
        )
        threshold = threshold_details["threshold"]
        _train_text, train_metrics = build_metrics(
            train_y,
            (train_probability >= threshold).astype(np.int64),
            train_probability,
            log_loss(train_y, train_probability, labels=[0, 1]),
            "Training",
            threshold,
        )
        _val_text, val_metrics = build_metrics(
            val_y,
            (val_probability >= threshold).astype(np.int64),
            val_probability,
            log_loss(val_y, val_probability, labels=[0, 1]),
            "Validation",
            threshold,
        )
        selection_score = float(val_metrics[selection_metric])
        is_best = selection_score > best_score
        if is_best:
            best_score = selection_score
            best_round = round_index
        append_training_history(
            history_path,
            {
                "epoch": round_index,
                "total_epochs": total_rounds,
                "learning_rate": learning_rate,
                "train_loss": train_metrics["loss"],
                "train_f1": train_metrics["f1"],
                "train_precision": train_metrics["precision"],
                "train_recall": train_metrics["recall"],
                "train_oa": train_metrics["oa"],
                "train_kappa": train_metrics["kappa"],
                "val_loss": val_metrics["loss"],
                "val_f1": val_metrics["f1"],
                "val_auc": val_metrics["auc"],
                "val_pr_auc": val_metrics["pr_auc"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_oa": val_metrics["oa"],
                "val_kappa": val_metrics["kappa"],
                "val_threshold": threshold,
                "val_threshold_max_score": threshold_details["maximum_score"],
                "val_threshold_selected_score": threshold_details["selected_score"],
                "val_precision_recall_gap": val_metrics["precision_recall_gap"],
                "selection_metric": selection_metric,
                "selection_score": selection_score,
                "is_best": int(is_best),
                "best_epoch": best_round,
                "best_selection_score": best_score,
                "early_stop_counter": "",
                "early_stopping_eligible": "",
                "domain_risk_penalty": "",
                "domain_risk_weight": "",
            },
        )

    # Mirror deep-model epoch selection: retain the cumulative estimator round
    # selected solely on validation data, never on the outer test region.
    if name in {"extra_trees", "random_forest"} and best_round < len(
        model.estimators_
    ):
        model.estimators_ = model.estimators_[:best_round]
        model.set_params(n_estimators=best_round)
    elif name == "catboost" and best_round < int(model.tree_count_):
        model.shrink(ntree_end=best_round)
    elif name == "lightgbm":
        model.booster_.best_iteration = int(best_round)
    return {
        "history_round_unit": (
            "complete_solver_fit"
            if name == "logistic_regression"
            else "cumulative_tree_or_boosting_iteration"
        ),
        "history_rounds": int(total_rounds),
        "history_best_validation_round": int(best_round),
        "history_best_validation_score": float(best_score),
        "final_estimator_round_selected_on_validation": int(best_round),
    }


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
    selection_metric: str = "auc",
    threshold_score_tolerance: float = 0.0,
) -> dict:
    """Fit on train, select threshold on validation, touch test exactly once."""
    name = canonical_model_name(name)
    os.makedirs(output_dir, exist_ok=True)
    train_x, train_y = arrays["train"]
    val_x, val_y = arrays["val"]
    spec = MODEL_SPECS[name]
    progress = track(
        total=6,
        desc=f"{spec.display_name} training workflow",
        unit="stage",
        leave=True,
    )
    progress.set_postfix_str("Building model", refresh=False)
    model = build_classical_model(name, seed, n_jobs, iterations)
    progress.update(1)
    progress.set_postfix_str("Fitting training set", refresh=True)
    with timed_task(
        f"{spec.display_name} fitting (train={len(train_y):,}, val={len(val_y):,})"
    ):
        _fit(
            model, name, train_x, train_y, train_weights, val_x, val_y, iterations
        )
    progress.update(1)

    history_metadata = write_classical_training_history(
        model,
        name,
        train_x,
        train_y,
        val_x,
        val_y,
        output_dir,
        selection_metric,
        threshold_metric,
        threshold_score_tolerance,
    )

    # Every selection decision is complete before test features are evaluated.
    progress.set_postfix_str("Selecting validation threshold", refresh=False)
    val_probability = _probabilities(model, val_x)
    threshold_details = find_best_threshold(
        val_y,
        val_probability,
        metric=threshold_metric,
        score_tolerance=threshold_score_tolerance,
        return_details=True,
    )
    threshold = threshold_details["threshold"]
    val_prediction = (val_probability >= threshold).astype(np.int64)
    _val_text, val_metrics = build_metrics(
        val_y, val_prediction, val_probability, float("nan"),
        "Validation", threshold,
    )
    progress.update(1)
    console(
        f"{spec.display_name} validation results | {metric_line(val_metrics)} | "
        f"threshold={threshold:.3f}"
    )

    progress.set_postfix_str("Evaluating training success rate", refresh=False)
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

    progress.set_postfix_str("Single-pass test-region evaluation", refresh=False)
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
        progress_desc=f"{spec.display_name} spatial-block bootstrap",
    ))
    metrics.update({
        "best_epoch": None,
        "validation_selection_metric": selection_metric,
        "validation_selection_score": float(val_metrics[selection_metric]),
        "threshold_metric": threshold_metric,
        "threshold_selected_on_validation": float(threshold),
        "validation_threshold_max_score": threshold_details["maximum_score"],
        "validation_threshold_selected_score": threshold_details["selected_score"],
        "validation_threshold_score_tolerance": threshold_details[
            "score_tolerance"
        ],
        "validation_threshold_precision_recall_gap": threshold_details[
            "precision_recall_gap"
        ],
        "validation_threshold_candidate_count": threshold_details[
            "candidate_count"
        ],
        "test_evaluations": 1,
        "success_rate_auc": float(success_metrics["auc"]),
        "success_rate_pr_auc": float(success_metrics["pr_auc"]),
    })
    progress.update(1)
    console(
        f"{spec.display_name} test results | {metric_line(metrics, ('auc', 'pr_auc', 'f1', 'kappa', 'precision', 'recall'))}"
    )

    progress.set_postfix_str("Saving model and predictions", refresh=False)
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
        "threshold_selection": threshold_details,
        "test_evaluations": 1,
        "estimator_parameters": model.get_params(deep=False),
        **history_metadata,
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
    progress.set_postfix_str("Completed", refresh=False)
    progress.close()
    return metrics
