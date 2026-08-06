"""Construct the classical machine-learning comparison estimators."""

from __future__ import annotations

import importlib.util

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


OPTIONAL_DEPENDENCIES = {
    "catboost": "catboost",
    "lightgbm": "lightgbm",
}

CLASSICAL_MODELS = {
    "logistic_regression",
    "catboost",
    "extra_trees",
    "lightgbm",
    "random_forest",
}


def dependency_status(model_names) -> dict:
    result = {}
    for name in model_names:
        package = OPTIONAL_DEPENDENCIES.get(name)
        result[name] = {
            "required_package": package,
            "available": package is None or importlib.util.find_spec(package) is not None,
        }
    return result


def require_dependencies(model_names):
    status = dependency_status(model_names)
    missing = [
        f"{name} (pip/conda package: {row['required_package']})"
        for name, row in status.items()
        if not row["available"]
    ]
    if missing:
        raise RuntimeError(
            "Missing optional dependencies for requested comparison models: "
            + ", ".join(missing)
            + ". Install environment.yml before running these models."
        )
    return status


def build_classical_model(name: str, seed: int, n_jobs: int, iterations: int):
    if name not in CLASSICAL_MODELS:
        raise ValueError(f"Not a registered classical comparison model: {name}")
    if name == "logistic_regression":
        return LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=max(1000, iterations),
            random_state=seed,
            n_jobs=n_jobs,
        )
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=iterations,
            max_features="sqrt",
            min_samples_leaf=1,
            bootstrap=False,
            n_jobs=n_jobs,
            random_state=seed,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=iterations,
            max_features="sqrt",
            min_samples_leaf=1,
            bootstrap=True,
            n_jobs=n_jobs,
            random_state=seed,
        )
    if name == "catboost":
        require_dependencies([name])
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            iterations=iterations,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=seed,
            thread_count=n_jobs,
            verbose=False,
            allow_writing_files=False,
        )
    if name == "lightgbm":
        require_dependencies([name])
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=iterations,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=n_jobs,
            verbosity=-1,
        )
    raise AssertionError(name)


__all__ = [
    "CLASSICAL_MODELS",
    "OPTIONAL_DEPENDENCIES",
    "build_classical_model",
    "dependency_status",
    "require_dependencies",
]
