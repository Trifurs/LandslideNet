#!/usr/bin/env python3
"""Nested continuous macro-region hold-out and DWSS/random control experiment.

This is the strict validation entry point requested in the second revision.  An
outer fold withholds one complete, scientifically defined macro-region for the
single final test.  A second complete region is withheld from the remaining
regions for model/epoch/threshold selection.  Normalization and DWSS are fitted
only on the regions left for inner training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import warnings
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.enums import Resampling
from scipy.stats import t as student_t

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from .classical_models import (
    dependency_status,
    require_dependencies,
    train_classical_model,
)
from .model_registry import (
    MODEL_GROUPS,
    MODEL_SPECS,
    build_deep_model,
    canonical_model_name,
    expand_model_selection,
    model_specs,
)
from .progress import configure_progress, console, metric_line, track
from .training import setup_fold_logger, train_model
from .reporting import write_rate_curve_reports
from .data import (
    FrozenDWSS,
    FrozenMinMaxNormalizer,
    audit_region_connectivity,
    build_fold_feature_transformer,
    choose_random_points,
    collect_positive_points,
    compute_region_factor_ranges,
    compute_training_category_counts,
    is_vector_inventory,
    list_factor_paths,
    make_sparse_loader,
    parse_frequency_ratio_specs,
    read_point_features,
    region_class_balanced_weights,
    sample_background_points,
    validate_aligned_rasters,
)


REQUIRED_METRICS = ["auc", "pr_auc", "f1", "kappa", "precision", "recall"]

PAPER_MODEL_SUITES = {
    "reviewer": ["landslidenet"],
    "machine_learning": [*MODEL_GROUPS["machine_learning"], "landslidenet"],
    "deep_learning": [*MODEL_GROUPS["deep_learning"], "landslidenet"],
    "ablation": [*MODEL_GROUPS["ablation"], "landslidenet"],
    "paper": MODEL_GROUPS["all"],
}


def normalize_path(value):
    value = os.path.expanduser(str(value).strip())
    if os.name != "nt":
        value = value.replace("\\", "/")
    return value


def read_xml_params(xml_path):
    root = ET.parse(xml_path).getroot()
    params = {}
    for param in root.findall("param"):
        name_node = param.find("name")
        value_node = param.find("value")
        if name_node is not None and value_node is not None:
            params[name_node.text] = normalize_path(value_node.text)
    return params


def parse_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def configured_region_path(params):
    template = params.get("macro_region_output")
    if not template:
        return None
    count = params.get("macro_region_count", "5")
    return normalize_path(template).replace(
        "{macro_region_count}", str(count)
    ).replace("{count}", str(count))


def load_region_provenance(regions_path):
    path = Path(regions_path)
    diagnostics_path = path.with_name(f"{path.stem}_diagnostics.json")
    if not diagnostics_path.exists():
        return {
            "source": "external_or_legacy_region_raster",
            "diagnostics_path": None,
            "warning": (
                "No terrain-region diagnostics sidecar was found. Continuity is still "
                "validated, but scientific provenance must be documented separately."
            ),
        }
    with open(diagnostics_path, encoding="utf-8") as handle:
        diagnostics = json.load(handle)
    if diagnostics.get("landslide_inventory_used") is True:
        raise ValueError(
            "The supplied region diagnostics state that the landslide inventory was used "
            "to construct boundaries; this would leak outcome information into validation."
        )
    if diagnostics.get("model_output_used") is True:
        raise ValueError(
            "The supplied region diagnostics state that model output was used to construct "
            "boundaries; this would invalidate the regional hold-out."
        )
    return {
        "source": diagnostics.get("scientific_interpretation", "generated_region_raster"),
        "diagnostics_path": str(diagnostics_path),
        "method": diagnostics.get("method"),
        "configuration_sha256": diagnostics.get("configuration", {}).get("sha256"),
        "landslide_inventory_used": diagnostics.get("landslide_inventory_used"),
        "model_output_used": diagnostics.get("model_output_used"),
        "preregistration_note": diagnostics.get("preregistration_note"),
    }


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def software_environment():
    packages = {}
    for package in (
        "numpy", "scipy", "pandas", "scikit-learn", "rasterio", "torch",
        "torchvision", "jenkspy", "catboost", "lightgbm", "geopandas",
        "pyogrio", "shapely",
    ):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cuda_available = bool(torch.cuda.is_available())
        cuda_device_count = int(torch.cuda.device_count())
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "torch_cuda_version": torch.version.cuda,
    }


def parse_device_ids(value):
    return [int(item.strip()) for item in str(value).strip("[]").split(",") if item.strip()]


def resolve_requested_models(args, params):
    explicit_models = getattr(args, "models", None)
    if explicit_models:
        requested = expand_model_selection(explicit_models)
        selection_source = "cli_models"
    elif getattr(args, "suite", None):
        suite = args.suite
        requested = expand_model_selection(PAPER_MODEL_SUITES[suite])
        selection_source = f"cli_suite:{suite}"
    else:
        xml_models = params.get("models") or params.get("model")
        if xml_models:
            requested = expand_model_selection(_parse_tokens(xml_models))
            selection_source = "xml_models"
        else:
            suite = params.get("experiment_suite", "reviewer")
            if suite not in PAPER_MODEL_SUITES:
                raise ValueError(
                    f"Unknown experiment suite {suite!r}; choose from "
                    f"{sorted(PAPER_MODEL_SUITES)}."
                )
            requested = expand_model_selection(PAPER_MODEL_SUITES[suite])
            selection_source = f"suite:{suite}"
    if not requested:
        raise ValueError("Model selection resolved to an empty list.")
    return selection_source, requested


def _parse_tokens(value):
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [
        token.strip().strip("'\"")
        for token in str(value).strip("[]").replace(",", " ").split()
        if token.strip().strip("'\"")
    ]


def _array_sha256(values, dtype) -> str:
    """Return a stable content fingerprint for fold-pairing audits."""
    array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    digest = hashlib.sha256()
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _json_sha256(value) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def configure_experiment_args(args, params):
    """Resolve CLI > XML > audited default for experiment controls."""
    scalar_fields = (
        ("positive_value", "positive_value", int, 1),
        ("min_region_positives", "regional_min_region_positives", int, 1),
        ("negative_ratio", "negative_ratio", float, 1.0),
        ("candidate_multiplier", "dwss_candidate_multiplier", float, 10.0),
        ("candidate_minimum", "dwss_candidate_minimum", int, 10000),
        ("candidate_maximum", "dwss_candidate_maximum", int, 100000),
        (
            "training_candidate_minimum",
            "dwss_training_candidate_minimum",
            int,
            200000,
        ),
        (
            "training_candidate_maximum",
            "dwss_training_candidate_maximum",
            int,
            200000,
        ),
        (
            "adaptive_candidate_maximum",
            "dwss_adaptive_candidate_maximum",
            int,
            15000000,
        ),
        ("adaptive_batch_size", "dwss_adaptive_batch_size", int, 250000),
        ("screening_neighbors", "dwss_screening_neighbors", int, 64),
        ("theta_min", "dwss_theta_min", float, 0.55),
        ("n_strata", "dwss_n_strata", int, 3),
        ("weight_power", "dwss_weight_power", float, 1.0),
        ("min_per_stratum", "dwss_min_per_stratum", int, 0),
        ("max_prototypes", "dwss_max_prototypes", int, 0),
        ("kde_chunk_size", "dwss_kde_chunk_size", int, 2048),
        ("raster_chunk_size", "raster_chunk_size", int, 1024),
        ("feature_tile_size", "feature_tile_size", int, 1024),
        ("seed", "experiment_seed", int, 20250609),
        ("selection_metric", "selection_metric", str, "auc"),
        ("threshold_metric", "threshold_metric", str, "f1"),
        (
            "threshold_score_tolerance",
            "threshold_score_tolerance",
            float,
            0.0,
        ),
        ("selection_min_delta", "selection_min_delta", float, 0.0),
        ("minimum_epochs", "minimum_epochs", int, 1),
        ("lr_warmup_epochs", "lr_warmup_epochs", int, 0),
        ("lr_plateau_patience", "lr_plateau_patience", int, 10),
        ("lr_plateau_factor", "lr_plateau_factor", float, 0.5),
        ("minimum_lr", "minimum_lr", float, 1e-6),
        (
            "domain_risk_variance_weight",
            "domain_risk_variance_weight",
            float,
            0.0,
        ),
        (
            "domain_risk_warmup_epochs",
            "domain_risk_warmup_epochs",
            int,
            0,
        ),
        ("training_context_views", "training_context_views", int, 2),
        (
            "max_supervised_points_per_training_tile",
            "max_supervised_points_per_training_tile",
            int,
            512,
        ),
        ("ema_decay", "ema_decay", float, 0.99),
        ("ema_start_epoch", "ema_start_epoch", int, 1),
        ("augmentation_mode", "augmentation_mode", str, "aspect_safe_d4"),
        ("aspect_period", "aspect_period", float, 1.0),
        ("classical_iterations", "classical_iterations", int, 500),
        ("classical_n_jobs", "classical_n_jobs", int, 8),
    )
    for attribute, xml_name, cast, default in scalar_fields:
        current = getattr(args, attribute, None)
        if current is None:
            setattr(args, attribute, cast(params.get(xml_name, default)))
    if getattr(args, "sampling_methods", None) is None:
        args.sampling_methods = _parse_tokens(
            params.get("experiment_sampling_methods", "dwss,random")
        )
    invalid_sampling = sorted(set(args.sampling_methods) - {"dwss", "random"})
    if invalid_sampling or not args.sampling_methods:
        raise ValueError(f"Invalid sampling methods: {invalid_sampling}")
    if getattr(args, "eval_tta", None) is None:
        args.eval_tta = parse_bool(params.get("evaluation_tta"), False)
    if args.negative_ratio <= 0:
        raise ValueError("negative_ratio must be positive.")
    if args.candidate_multiplier < 1 or args.candidate_minimum < 1:
        raise ValueError("DWSS candidate multiplier/minimum are invalid.")
    if args.candidate_maximum and args.candidate_maximum < 1:
        raise ValueError("DWSS candidate maximum must be 0 (unbounded) or positive.")
    if args.training_candidate_minimum < 1:
        raise ValueError("DWSS training-candidate minimum must be positive.")
    if (
        args.training_candidate_maximum
        and args.training_candidate_maximum < args.training_candidate_minimum
    ):
        raise ValueError(
            "DWSS training-candidate maximum cannot be smaller than its minimum."
        )
    if (
        args.adaptive_candidate_maximum < 1
        or args.adaptive_batch_size < 1
        or args.screening_neighbors < 1
    ):
        raise ValueError(
            "DWSS adaptive candidate maximum, batch size, and screening-neighbor "
            "count must be positive."
        )
    if args.max_prototypes < 0 or args.classical_iterations < 1:
        raise ValueError("Prototype cap/estimator iteration configuration is invalid.")
    if not 0 <= args.threshold_score_tolerance < 1:
        raise ValueError("threshold_score_tolerance must be in [0, 1).")
    if args.selection_min_delta < 0:
        raise ValueError("selection_min_delta cannot be negative.")
    if args.minimum_epochs < 1 or args.lr_plateau_patience < 1:
        raise ValueError("minimum_epochs and lr_plateau_patience must be positive.")
    if args.lr_warmup_epochs < 0 or args.domain_risk_warmup_epochs < 0:
        raise ValueError("Warm-up epoch counts cannot be negative.")
    if not 0 < args.lr_plateau_factor < 1 or args.minimum_lr <= 0:
        raise ValueError("Invalid plateau LR factor/minimum.")
    if args.domain_risk_variance_weight < 0:
        raise ValueError("domain_risk_variance_weight cannot be negative.")
    if args.training_context_views not in {1, 2, 4}:
        raise ValueError("training_context_views must be one of 1, 2, or 4.")
    if args.max_supervised_points_per_training_tile < 0:
        raise ValueError(
            "max_supervised_points_per_training_tile cannot be negative."
        )
    if not 0 <= args.ema_decay < 1 or args.ema_start_epoch < 1:
        raise ValueError("EMA decay must be in [0, 1), and its start epoch positive.")
    if args.augmentation_mode not in {"none", "aspect_safe_d4"}:
        raise ValueError("augmentation_mode must be none or aspect_safe_d4.")
    if args.aspect_period <= 0:
        raise ValueError("aspect_period must be positive.")
    if not np.isclose(args.negative_ratio, 1.0):
        raise ValueError(
            "The manuscript uses a 1:1 positive/negative design; negative_ratio must be 1."
        )
    if not np.isclose(args.theta_min, 0.55):
        raise ValueError(
            "The manuscript DWSS threshold is theta_min=0.55."
        )
    if args.n_strata != 3:
        raise ValueError(
            "The manuscript DWSS uses exactly three natural-break strata."
        )
    if not np.isclose(args.weight_power, 1.0):
        raise ValueError(
            "The manuscript DWSS allocation uses unpowered stratum means."
        )
    if args.min_per_stratum != 0:
        raise ValueError(
            "The manuscript DWSS allocation has no per-stratum minimum."
        )


def load_region_names(path):
    names = {}
    if not path:
        return names
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"region_id", "region_name"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Region-name CSV must contain {sorted(required)}.")
        for row in reader:
            names[int(row["region_id"])] = row["region_name"].strip()
    return names


def build_validation_map(region_ids, path=None, strategy="cyclic", positive_counts=None,
                         minimum_training_positives=1,
                         preferred_validation_positives=1):
    if not path and strategy == "cyclic":
        return {
            test_region: region_ids[(index + 1) % len(region_ids)]
            for index, test_region in enumerate(region_ids)
        }
    if not path and strategy == "support_aware":
        if positive_counts is None:
            raise ValueError("support_aware validation mapping needs positive counts.")
        counts = {region_id: int(positive_counts.get(region_id, 0)) for region_id in region_ids}
        if int(preferred_validation_positives) < 1:
            raise ValueError("preferred_validation_positives must be positive.")
        mapping = {}
        for test_region in region_ids:
            # The held-out test-region label count is never consulted when choosing
            # the inner validation region. Only the current development-region
            # support is used.
            development_regions = [
                region_id for region_id in region_ids if region_id != test_region
            ]
            candidates = []
            for validation_region in development_regions:
                training_count = int(sum(
                    counts[region_id]
                    for region_id in development_regions
                    if region_id != validation_region
                ))
                if training_count >= int(minimum_training_positives):
                    candidates.append({
                        "validation_region": validation_region,
                        "validation_count": counts[validation_region],
                        "training_count": training_count,
                    })
            if not candidates:
                raise ValueError(
                    f"No validation region leaves at least {minimum_training_positives} "
                    f"inner-training positives when region {test_region} is the test region. "
                    "Lower regional_min_inner_training_positives only with an explicit "
                    "scientific justification."
                )
            supported = [
                candidate for candidate in candidates
                if candidate["validation_count"] >= int(preferred_validation_positives)
            ]
            if supported:
                # Once validation support is sufficient, retain as much information as
                # possible for model fitting by choosing the smallest sufficient region.
                chosen = min(
                    supported,
                    key=lambda candidate: (
                        candidate["validation_count"],
                        candidate["validation_region"],
                    ),
                )
            else:
                # A whole-region constraint can make the preferred support impossible.
                # In that case choose the strongest feasible validation region.
                chosen = max(
                    candidates,
                    key=lambda candidate: (
                        candidate["validation_count"],
                        -candidate["validation_region"],
                    ),
                )
            mapping[test_region] = chosen["validation_region"]
        return mapping
    if not path:
        raise ValueError(f"Unknown validation strategy: {strategy}")
    mapping = {}
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"test_region", "validation_region"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Validation-map CSV must contain {sorted(required)}.")
        for row in reader:
            mapping[int(row["test_region"])] = int(row["validation_region"])
    missing = set(region_ids) - set(mapping)
    if missing:
        raise ValueError(f"Validation map is missing test regions: {sorted(missing)}")
    return mapping


def candidate_pool_target(positive_count, negative_ratio, multiplier, minimum, maximum):
    required = int(round(positive_count * negative_ratio))
    target = max(required, int(np.ceil(required * multiplier)), minimum)
    if maximum > 0:
        target = min(target, maximum)
    if target < required:
        raise ValueError("Candidate pool maximum is smaller than the required negatives.")
    return required, target


def select_region_points(points, features, region_ids):
    mask = np.isin(points[:, 2], np.asarray(region_ids, dtype=np.int64))
    return points[mask], features[mask]


def sample_candidate_pool(
    args,
    factor_paths,
    allowed_regions,
    positive_count,
    seed,
    split,
):
    use_training_pool = split == "train" and "dwss" in args.sampling_methods
    minimum = (
        args.training_candidate_minimum
        if use_training_pool
        else args.candidate_minimum
    )
    maximum = (
        args.training_candidate_maximum
        if use_training_pool
        else args.candidate_maximum
    )
    required, requested = candidate_pool_target(
        positive_count,
        args.negative_ratio,
        args.candidate_multiplier,
        minimum,
        maximum,
    )
    points, total_background = sample_background_points(
        args.inventory,
        args.regions,
        allowed_regions,
        requested,
        seed,
        positive_value=args.positive_value,
        chunk_size=args.raster_chunk_size,
    )
    drawn_count = len(points)
    features, valid = read_point_features(
        factor_paths,
        points,
        tile_size=args.feature_tile_size,
    )
    points = points[valid]
    features = features[valid]
    if len(points) < required:
        raise RuntimeError(
            f"Only {len(points)} factor-valid candidates remain for {required} required "
            f"negatives in regions {list(allowed_regions)}. Increase --candidate-multiplier."
        )
    audit = {
        "allowed_regions": list(map(int, allowed_regions)),
        "total_background_cells": int(total_background),
        "candidate_pool_requested": int(requested),
        "candidate_pool_drawn": int(drawn_count),
        "candidate_pool_factor_valid": int(len(points)),
        "required_negative_samples": int(required),
        "split": split,
        "training_dwss_initial_pool": bool(use_training_pool),
    }
    return points, features, required, audit


def augment_dwss_training_candidates(
    args,
    factor_paths,
    allowed_regions,
    train_data,
    dwss,
    seed,
):
    """Fill strict DWSS stratum quotas with uniform proposals and safe pruning.

    The initial candidate pool fits the fold-specific natural breaks. If a
    stratum cannot meet Eq. (1), additional background pixels are proposed
    uniformly. A Gaussian-kernel density upper bound rejects only pixels that
    provably cannot enter any currently deficient stratum; all survivors use
    the original exact KDE before being retained.
    """
    initial_points = np.asarray(train_data["candidate_points"], dtype=np.int64)
    initial_raw = np.asarray(train_data["candidate_raw"], dtype=np.float32)
    initial_divergence = np.asarray(
        dwss.training_candidate_divergence,
        dtype=np.float64,
    )
    if len(initial_points) != len(initial_divergence):
        raise RuntimeError("Initial DWSS points and divergence values are misaligned.")

    total = int(train_data["required_negatives"])
    status = dwss.allocation_status(initial_divergence, total)
    base_audit = {
        "initial_candidate_count": int(len(initial_points)),
        "initial_eligible_count": int(status["eligible_count"]),
        "initial_target_counts": status["desired"].tolist(),
        "initial_available_counts": status["available"].tolist(),
        "initial_deficit_counts": status["deficit"].tolist(),
        "adaptive_search_used": bool(np.any(status["deficit"])),
        "proposal_rule": (
            "uniform_without_replacement_within_the_adaptive_draw_from_the_same "
            "inner-training background; overlaps with the initial draw are removed"
        ),
        "screening_rule": (
            "k-nearest Gaussian-kernel upper bound; every retained candidate "
            "receives exact joint-KDE evaluation"
        ),
        "screening_is_false_negative_safe": True,
        "screening_neighbors": int(args.screening_neighbors),
        "processing_batch_size": int(args.adaptive_batch_size),
        "cross_stratum_quota_redistribution": False,
    }
    train_data["base_uniform_candidate_count"] = int(len(initial_points))
    if not np.any(status["deficit"]):
        train_data["dwss_candidate_divergence"] = initial_divergence
        dwss.selection_candidate_count = int(len(initial_divergence))
        base_audit.update({
            "adaptive_candidates_requested": 0,
            "adaptive_candidates_processed": 0,
            "adaptive_candidates_factor_valid": 0,
            "adaptive_candidates_passing_safe_screen": 0,
            "adaptive_candidates_retained": 0,
            "final_target_counts": status["desired"].tolist(),
            "final_available_counts": status["available"].tolist(),
            "strict_manuscript_allocation_satisfied": True,
        })
        train_data["candidate_audit"]["dwss_adaptive_search"] = base_audit
        return

    maximum = min(
        int(args.adaptive_candidate_maximum),
        int(train_data["candidate_audit"]["total_background_cells"]),
    )
    initial_count = max(len(initial_divergence), 1)
    estimated = []
    for deficit, available in zip(status["deficit"], status["available"]):
        if deficit <= 0:
            continue
        if available <= 0:
            estimated.append(maximum)
        else:
            observed_rate = float(available) / initial_count
            # A 2x reserve absorbs Monte-Carlo variation and small shifts in
            # conditional stratum means after additional uniform draws.
            estimated.append(int(np.ceil(2.0 * float(deficit) / observed_rate)))
    requested = min(
        maximum,
        max(int(args.adaptive_batch_size), max(estimated, default=maximum)),
    )
    if requested < 1:
        raise RuntimeError(
            "DWSS strata are deficient but the adaptive candidate budget is empty."
        )

    proposal_points, total_background = sample_background_points(
        args.inventory,
        args.regions,
        allowed_regions,
        requested,
        seed,
        positive_value=args.positive_value,
        chunk_size=args.raster_chunk_size,
    )
    proposal_rng = np.random.default_rng(seed + 1)
    proposal_rng.shuffle(proposal_points)

    retained_points = []
    retained_raw = []
    retained_divergence = []
    known_coordinates = {
        (int(point[0]), int(point[1]))
        for point in initial_points
    }
    processed = 0
    factor_valid_count = 0
    screened_count = 0
    exact_count = 0

    for start in range(0, len(proposal_points), int(args.adaptive_batch_size)):
        combined_divergence = np.concatenate(
            [initial_divergence, *retained_divergence]
        )
        dwss.refresh_stratum_statistics(combined_divergence)
        status = dwss.allocation_status(combined_divergence, total)
        deficient = np.flatnonzero(status["deficit"] > 0)
        if not len(deficient):
            break

        stop = min(start + int(args.adaptive_batch_size), len(proposal_points))
        point_batch = proposal_points[start:stop]
        processed += len(point_batch)
        raw_batch, factor_valid = read_point_features(
            factor_paths,
            point_batch,
            tile_size=args.feature_tile_size,
        )
        point_batch = point_batch[factor_valid]
        raw_batch = raw_batch[factor_valid]
        factor_valid_count += len(point_batch)
        if not len(point_batch):
            continue

        screen = dwss.screen_for_strata(
            raw_batch,
            int(deficient.max()),
            neighbors=args.screening_neighbors,
        )
        screened_count += int(np.count_nonzero(screen))
        if not np.any(screen):
            continue

        screened_points = point_batch[screen]
        screened_raw = raw_batch[screen]
        screened_divergence = dwss.divergence(screened_raw)
        exact_count += len(screened_divergence)
        eligible = screened_divergence >= dwss.theta_min
        screened_strata = dwss.assign_strata(screened_divergence)
        retain = eligible & np.isin(screened_strata, deficient)
        if not np.any(retain):
            continue

        candidate_points = screened_points[retain]
        candidate_raw = screened_raw[retain]
        candidate_divergence = screened_divergence[retain]
        novel = np.asarray(
            [
                (int(point[0]), int(point[1])) not in known_coordinates
                for point in candidate_points
            ],
            dtype=bool,
        )
        if not np.any(novel):
            continue
        candidate_points = candidate_points[novel]
        candidate_raw = candidate_raw[novel]
        candidate_divergence = candidate_divergence[novel]
        known_coordinates.update(
            (int(point[0]), int(point[1]))
            for point in candidate_points
        )
        retained_points.append(candidate_points)
        retained_raw.append(candidate_raw)
        retained_divergence.append(candidate_divergence)

    combined_divergence = np.concatenate(
        [initial_divergence, *retained_divergence]
    )
    dwss.refresh_stratum_statistics(combined_divergence)
    final_status = dwss.allocation_status(combined_divergence, total)
    if np.any(final_status["deficit"]):
        raise RuntimeError(
            "The configured adaptive DWSS budget cannot satisfy the strict "
            "manuscript allocation. "
            f"targets={final_status['desired'].tolist()}, "
            f"available={final_status['available'].tolist()}, "
            f"deficits={final_status['deficit'].tolist()}, "
            f"processed={processed:,}, maximum={args.adaptive_candidate_maximum:,}. "
            "Increase dwss_adaptive_candidate_maximum in the XML; quotas were not "
            "redistributed."
        )

    if retained_points:
        train_data["candidate_points"] = np.concatenate(
            [initial_points, *retained_points],
            axis=0,
        )
        train_data["candidate_raw"] = np.concatenate(
            [initial_raw, *retained_raw],
            axis=0,
        )
    train_data["dwss_candidate_divergence"] = combined_divergence
    dwss.selection_candidate_count = int(len(combined_divergence))
    base_audit.update({
        "total_background_cells": int(total_background),
        "adaptive_candidates_requested": int(requested),
        "adaptive_candidates_processed": int(processed),
        "adaptive_candidates_factor_valid": int(factor_valid_count),
        "adaptive_candidates_passing_safe_screen": int(screened_count),
        "adaptive_candidates_exact_kde": int(exact_count),
        "adaptive_candidates_retained": int(
            sum(len(values) for values in retained_points)
        ),
        "final_target_counts": final_status["desired"].tolist(),
        "final_available_counts": final_status["available"].tolist(),
        "final_deficit_counts": final_status["deficit"].tolist(),
        "strict_manuscript_allocation_satisfied": True,
    })
    train_data["candidate_audit"]["dwss_adaptive_search"] = base_audit


def method_negative_samples(method, split_data, dwss, args, seed,
                            shared_evaluation_negatives):
    selected = {}
    diagnostics = {}
    offsets = {"train": 11, "val": 23, "test": 37}
    for split, data in split_data.items():
        points = data["candidate_points"]
        total = data["required_negatives"]
        split_seed = seed + offsets[split]
        if split in {"val", "test"}:
            if split not in shared_evaluation_negatives:
                raise RuntimeError(f"Missing shared {split} evaluation negatives.")
            selected[split] = np.asarray(
                shared_evaluation_negatives[split], dtype=np.int64
            ).copy()
            if len(selected[split]) != total:
                raise RuntimeError(
                    f"Shared {split} negatives contain {len(selected[split])} samples; "
                    f"expected {total}."
                )
            diagnostics[split] = {
                "candidate_pool": int(len(points)),
                "selected": int(len(selected[split])),
                "selection": "fixed_uniform_shared_across_sampling_arms",
                "sampling_method_does_not_change_evaluation_set": True,
            }
        elif method == "dwss":
            selected[split], diagnostics[split] = dwss.select(
                points,
                data["candidate_raw"],
                total,
                split_seed,
                minimum_per_stratum=args.min_per_stratum,
                precomputed_divergence=data.get(
                    "dwss_candidate_divergence",
                    dwss.training_candidate_divergence,
                ),
            )
        else:
            base_count = int(data.get("base_uniform_candidate_count", len(points)))
            selected[split] = choose_random_points(
                points[:base_count],
                total,
                split_seed,
            )
            diagnostics[split] = {
                "candidate_pool": base_count,
                "selected": int(len(selected[split])),
                "selection": "uniform_without_replacement",
                "adaptive_dwss_search_points_excluded": int(len(points) - base_count),
            }
    return selected, diagnostics


def selected_candidate_features(candidate_points, candidate_features, selected_points):
    """Return frozen transformed features in exactly the selected point order."""
    candidate_points = np.asarray(candidate_points, dtype=np.int64)
    selected_points = np.asarray(selected_points, dtype=np.int64)
    lookup = {
        (int(point[0]), int(point[1])): index
        for index, point in enumerate(candidate_points)
    }
    try:
        indices = np.asarray([
            lookup[(int(point[0]), int(point[1]))]
            for point in selected_points
        ], dtype=np.int64)
    except KeyError as error:
        raise RuntimeError(
            f"Selected negative coordinate {error.args[0]} is absent from its frozen "
            "candidate pool."
        ) from error
    return np.asarray(candidate_features, dtype=np.float32)[indices]


def make_classical_arrays(split_data, negatives):
    arrays = {}
    points = {}
    for split in ("train", "val", "test"):
        negative_features = selected_candidate_features(
            split_data[split]["candidate_points"],
            split_data[split]["candidate_normalized"],
            negatives[split],
        )
        positive_features = split_data[split]["positive_normalized"]
        arrays[split] = (
            np.concatenate((positive_features, negative_features), axis=0),
            np.concatenate((
                np.ones(len(positive_features), dtype=np.int64),
                np.zeros(len(negative_features), dtype=np.int64),
            )),
        )
        points[split] = np.concatenate((
            split_data[split]["positive_points"], negatives[split]
        ), axis=0)
    return arrays, points


def write_sample_manifest(path, transform, method, fold_index, region_names,
                          split_data, negatives):
    fieldnames = [
        "fold", "sampling_method", "split", "class", "label", "row", "col",
        "x", "y", "region_id", "region_name",
    ]
    total_rows = sum(
        len(split_data[split]["positive_points"]) + len(negatives[split])
        for split in ("train", "val", "test")
    )
    manifest_progress = track(
        total=total_rows,
        desc=f"Fold {fold_index} {method} sample inventory",
        unit="sample",
    )
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for split in ("train", "val", "test"):
            groups = (
                ("landslide", 1, split_data[split]["positive_points"]),
                ("non_landslide", 0, negatives[split]),
            )
            for class_name, label, points in groups:
                if len(points) == 0:
                    continue
                xs, ys = rasterio.transform.xy(
                    transform,
                    points[:, 0],
                    points[:, 1],
                    offset="center",
                )
                for point, x, y in zip(points, xs, ys):
                    region_id = int(point[2])
                    writer.writerow({
                        "fold": fold_index,
                        "sampling_method": method,
                        "split": split,
                        "class": class_name,
                        "label": label,
                        "row": int(point[0]),
                        "col": int(point[1]),
                        "x": float(x),
                        "y": float(y),
                        "region_id": region_id,
                        "region_name": region_names.get(region_id, f"region_{region_id}"),
                    })
                    manifest_progress.update(1)
    manifest_progress.close()


def write_region_audit(path, region_ids, region_names, factor_valid_positive_counts,
                       inventory_positive_counts, background_counts, connectivity_audit,
                       low_support_threshold=30, adequate_support_threshold=100):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "region_id", "region_name", "region_cells", "component_count",
                "positive_pixels", "positive_fraction", "inventory_positive_pixels",
                "factor_invalid_positive_pixels", "background_pixels", "support_tier",
                "reporting_guidance",
            ],
        )
        writer.writeheader()
        total_positives = max(
            sum(factor_valid_positive_counts.get(region_id, 0) for region_id in region_ids),
            1,
        )
        for region_id in region_ids:
            positive_count = factor_valid_positive_counts.get(region_id, 0)
            inventory_positive_count = inventory_positive_counts.get(region_id, 0)
            if positive_count < low_support_threshold:
                support_tier = "very_low"
                guidance = "report_bootstrap_ci_and_interpret_with_high_caution"
            elif positive_count < adequate_support_threshold:
                support_tier = "limited"
                guidance = "report_bootstrap_ci_and_avoid_rank_claims"
            else:
                support_tier = "adequate"
                guidance = "report_point_estimate_and_bootstrap_ci"
            writer.writerow({
                "region_id": region_id,
                "region_name": region_names.get(region_id, f"region_{region_id}"),
                "region_cells": connectivity_audit["regions"][region_id]["cell_count"],
                "component_count": connectivity_audit["regions"][region_id]["component_count"],
                "positive_pixels": positive_count,
                "positive_fraction": positive_count / total_positives,
                "inventory_positive_pixels": inventory_positive_count,
                "factor_invalid_positive_pixels": inventory_positive_count - positive_count,
                "background_pixels": background_counts.get(region_id, 0),
                "support_tier": support_tier,
                "reporting_guidance": guidance,
            })


def write_validation_support_audit(path, region_ids, mapping, positive_counts,
                                   preferred_validation_positives=1):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "test_region", "test_positive_pixels", "validation_region",
            "validation_positive_pixels", "preferred_validation_positive_pixels",
            "preferred_validation_support_met", "inner_training_regions",
            "inner_training_positive_pixels",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for test_region in region_ids:
            validation_region = mapping[test_region]
            training_regions = [
                region_id for region_id in region_ids
                if region_id not in {test_region, validation_region}
            ]
            writer.writerow({
                "test_region": test_region,
                "test_positive_pixels": positive_counts.get(test_region, 0),
                "validation_region": validation_region,
                "validation_positive_pixels": positive_counts.get(validation_region, 0),
                "preferred_validation_positive_pixels": preferred_validation_positives,
                "preferred_validation_support_met": (
                    positive_counts.get(validation_region, 0)
                    >= int(preferred_validation_positives)
                ),
                "inner_training_regions": "|".join(map(str, training_regions)),
                "inner_training_positive_pixels": int(sum(
                    positive_counts.get(region_id, 0) for region_id in training_regions
                )),
            })


def write_frequency_ratio_mapping(path, feature_transformer):
    fieldnames = [
        "factor_name", "factor_index", "category_code", "training_area_pixels",
        "training_positive_pixels", "expected_positive_pixels_under_area_null",
        "smoothed_frequency_ratio", "encoded_value",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for encoder in feature_transformer.frequency_ratio_encoders:
            for row in encoder.to_dict()["categories"]:
                writer.writerow({
                    "factor_name": encoder.spec.factor_name,
                    "factor_index": encoder.spec.factor_index,
                    **row,
                })


def plot_macro_region_layout(path, regions_path, positives, region_ids, region_names,
                             max_dimension=1800):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        console("matplotlib is unavailable; skipping the macro-region layout figure.", level="WARNING")
        return

    with rasterio.open(regions_path) as source:
        scale = max(1.0, max(source.height, source.width) / float(max_dimension))
        out_height = max(1, int(round(source.height / scale)))
        out_width = max(1, int(round(source.width / scale)))
        region_data = source.read(
            1,
            out_shape=(out_height, out_width),
            resampling=Resampling.nearest,
        )
        bounds = source.bounds
        transform = source.transform

    indexed = np.zeros(region_data.shape, dtype=np.int16)
    for display_index, region_id in enumerate(region_ids, start=1):
        indexed[np.isclose(region_data, region_id)] = display_index
    indexed = np.ma.masked_equal(indexed, 0)
    cmap = plt.get_cmap("tab20", len(region_ids))

    fig, ax = plt.subplots(figsize=(9.0, 8.0))
    ax.imshow(
        indexed,
        extent=(bounds.left, bounds.right, bounds.bottom, bounds.top),
        origin="upper",
        interpolation="nearest",
        cmap=cmap,
        vmin=1,
        vmax=max(1, len(region_ids)),
        alpha=0.82,
    )
    selected = positives[np.isin(positives[:, 2], region_ids)]
    if len(selected):
        xs, ys = rasterio.transform.xy(
            transform,
            selected[:, 0],
            selected[:, 1],
            offset="center",
        )
        ax.scatter(xs, ys, s=2.5, c="black", alpha=0.45, linewidths=0, label="Landslide")
    handles = [
        Patch(
            facecolor=cmap((index - 1) / max(len(region_ids) - 1, 1)),
            label=region_names.get(region_id, f"region_{region_id}"),
        )
        for index, region_id in enumerate(region_ids, start=1)
    ]
    handles.append(Patch(facecolor="black", label="Landslide inventory"))
    ax.legend(handles=handles, loc="best", fontsize=8, frameon=True)
    ax.set_title("Continuous macro-regions used for nested regional hold-out")
    ax.set_xlabel("Easting (map CRS)")
    ax.set_ylabel("Northing (map CRS)")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_unified_five_fold_table(frame, output_dir):
    """Write fold rows and per-model/per-arm five-fold summaries in one CSV."""
    identity = ["model", "sampling_method", "fold", "test_region"]
    duplicates = frame.duplicated(identity, keep=False)
    if np.any(duplicates):
        rows = frame.loc[duplicates, identity].to_dict(orient="records")
        raise RuntimeError(f"Duplicate fold metric rows would corrupt summaries: {rows}")

    metric_candidates = [
        "loss",
        "threshold",
        "auc",
        "pr_auc",
        "oa",
        "kappa",
        "precision",
        "recall",
        "precision_recall_gap",
        "specificity",
        "balanced_accuracy",
        "mcc",
        "brier",
        "ece",
        "f1",
        "iou",
        "tn",
        "fp",
        "fn",
        "tp",
        "auc_ci_low",
        "auc_ci_high",
        "pr_auc_ci_low",
        "pr_auc_ci_high",
        "f1_ci_low",
        "f1_ci_high",
        "kappa_ci_low",
        "kappa_ci_high",
        "precision_ci_low",
        "precision_ci_high",
        "recall_ci_low",
        "recall_ci_high",
        "validation_selection_score",
        "best_epoch",
        "epochs_completed",
        "early_stopped",
        "threshold_selected_on_validation",
        "validation_threshold_max_score",
        "validation_threshold_selected_score",
        "validation_threshold_score_tolerance",
        "validation_threshold_precision_recall_gap",
        "validation_threshold_candidate_count",
        "success_rate_auc",
        "success_rate_pr_auc",
        "training_unique_real_samples",
        "training_supervision_instances",
        "training_context_view_count",
        "training_optimizer_items",
        "training_observed_max_supervised_points_per_item",
        "training_batch_weight_mass_coefficient_of_variation",
        "training_batch_weight_mass_max_to_mean_ratio",
        "training_configured_negative_to_positive_weight_ratio",
        "training_observed_negative_to_positive_weight_ratio",
        "ema_decay",
        "ema_start_epoch",
        "ema_updates",
        "train_positive_samples",
        "validation_positive_samples",
        "test_positive_samples",
    ]
    metric_columns = [
        column for column in metric_candidates if column in frame.columns
    ]
    metadata_columns = [
        "model",
        "model_display_name",
        "model_family",
        "implementation",
        "paper_role",
        "sampling_method",
        "fold",
        "test_region",
        "validation_region",
        "train_regions",
    ]
    metadata_columns = [
        column for column in metadata_columns if column in frame.columns
    ]
    fold_rows = frame[metadata_columns + metric_columns].copy()
    fold_rows.insert(0, "row_type", "fold")
    fold_rows.insert(1, "completed_folds", 1)
    fold_rows.insert(2, "expected_folds", 5)

    summary_rows = []
    groups = ["model", "model_display_name", "sampling_method"]
    optional_group_columns = [
        column
        for column in ("model_family", "implementation", "paper_role")
        if column in frame.columns
    ]
    for key, group in frame.groupby(groups, sort=False):
        model_name, display_name, sampling_method = key
        fold_count = int(group["test_region"].nunique())
        base = {
            "model": model_name,
            "model_display_name": display_name,
            "sampling_method": sampling_method,
            "fold": "all",
            "test_region": "all",
            "validation_region": "varies",
            "train_regions": "varies",
        }
        for column in optional_group_columns:
            base[column] = group[column].iloc[0]
        for statistic, row_type in (
            ("mean", "five_fold_mean" if fold_count == 5 else "available_fold_mean"),
            ("std", "five_fold_std" if fold_count == 5 else "available_fold_std"),
        ):
            row = {
                "row_type": row_type,
                "completed_folds": fold_count,
                "expected_folds": 5,
                **base,
            }
            for column in metric_columns:
                values = pd.to_numeric(group[column], errors="coerce")
                if statistic == "mean":
                    row[column] = float(values.mean())
                else:
                    row[column] = (
                        float(values.std(ddof=0)) if values.notna().sum() > 1 else 0.0
                    )
            summary_rows.append(row)

    unified = pd.concat(
        [fold_rows, pd.DataFrame(summary_rows)],
        ignore_index=True,
        sort=False,
    )
    model_order = {name: index for index, name in enumerate(MODEL_SPECS)}
    sampling_order = {"dwss": 0, "random": 1}
    row_order = {"fold": 0, "five_fold_mean": 1, "available_fold_mean": 1,
                 "five_fold_std": 2, "available_fold_std": 2}
    unified["_model_order"] = unified["model"].map(model_order).fillna(len(model_order))
    unified["_sampling_order"] = (
        unified["sampling_method"].map(sampling_order).fillna(len(sampling_order))
    )
    unified["_row_order"] = unified["row_type"].map(row_order).fillna(9)
    unified["_fold_order"] = pd.to_numeric(unified["fold"], errors="coerce").fillna(999)
    unified = unified.sort_values(
        ["_sampling_order", "_model_order", "_row_order", "_fold_order"],
        kind="stable",
    ).drop(
        columns=["_model_order", "_sampling_order", "_row_order", "_fold_order"]
    )
    unified.to_csv(
        os.path.join(output_dir, "all_models_5fold_metrics.csv"),
        index=False,
    )


def write_unified_training_history(frame, output_dir):
    """Collect every per-model epoch/round history into one long-form CSV."""
    history_frames = []
    identity = [
        "fold",
        "test_region",
        "validation_region",
        "sampling_method",
        "model",
        "model_display_name",
        "model_family",
    ]
    available_identity = [column for column in identity if column in frame.columns]
    for _, row in frame.drop_duplicates(
        ["fold", "test_region", "sampling_method", "model"]
    ).iterrows():
        model_dir = Path(
            output_dir,
            f"Fold_{int(row['fold'])}_TestRegion_{int(row['test_region'])}",
            str(row["sampling_method"]).upper(),
            f"Model_{row['model']}",
        )
        history_path = model_dir / "training_history.csv"
        if not history_path.is_file():
            continue
        history = pd.read_csv(history_path)
        for column in reversed(available_identity):
            history.insert(0, column, row[column])
        history.insert(
            len(available_identity),
            "round_unit",
            (
                "epoch"
                if row.get("model_family") == "deep"
                else "fit_or_cumulative_estimator_round"
            ),
        )
        history.insert(
            len(available_identity) + 1,
            "history_file",
            os.path.relpath(history_path, output_dir),
        )
        history_frames.append(history)
    if history_frames:
        pd.concat(history_frames, ignore_index=True, sort=False).to_csv(
            os.path.join(output_dir, "all_models_training_history.csv"),
            index=False,
        )


def write_comparison_tables(metrics, output_dir):
    if not metrics:
        return
    frame = pd.DataFrame(metrics)
    if "model" not in frame:
        frame["model"] = "landslidenet"
    if "model_display_name" not in frame:
        frame["model_display_name"] = frame["model"].map(
            lambda name: MODEL_SPECS.get(name, MODEL_SPECS["landslidenet"]).display_name
        )
    detailed_path = os.path.join(output_dir, "sampling_comparison_detailed.csv")
    frame.to_csv(detailed_path, index=False)
    frame.to_csv(os.path.join(output_dir, "experiment_metrics_detailed.csv"), index=False)
    write_unified_five_fold_table(frame, output_dir)
    write_unified_training_history(frame, output_dir)

    available = [metric for metric in REQUIRED_METRICS if metric in frame.columns]
    groups = ["model", "model_display_name", "sampling_method"]
    summary = frame.groupby(groups)[available].agg(["mean", "std", "min", "max"])
    summary.to_csv(os.path.join(output_dir, "sampling_comparison_summary.csv"))

    balanced_rows = []
    for (model_name, display_name, method), group in frame.groupby(groups):
        support = group["test_positive_samples"].to_numpy(dtype=np.float64)
        for metric in available:
            values = group[metric].to_numpy(dtype=np.float64)
            finite = np.isfinite(values)
            if not np.any(finite):
                continue
            usable_values = values[finite]
            usable_support = support[finite]
            macro_mean = float(np.mean(usable_values))
            macro_std = float(np.std(usable_values, ddof=1)) if len(usable_values) > 1 else 0.0
            if len(usable_values) > 1:
                half_width = float(
                    student_t.ppf(0.975, df=len(usable_values) - 1)
                    * macro_std / np.sqrt(len(usable_values))
                )
            else:
                half_width = float("nan")
            metric_lower = -1.0 if metric == "kappa" else 0.0
            if np.isfinite(half_width):
                macro_ci_low = max(metric_lower, macro_mean - half_width)
                macro_ci_high = min(1.0, macro_mean + half_width)
            else:
                macro_ci_low = macro_ci_high = float("nan")
            balanced_rows.append({
                "model": model_name,
                "model_display_name": display_name,
                "sampling_method": method,
                "metric": metric,
                "region_macro_mean": macro_mean,
                "region_macro_std": macro_std,
                "region_macro_mean_ci_low": macro_ci_low,
                "region_macro_mean_ci_high": macro_ci_high,
                "region_macro_median": float(np.median(usable_values)),
                "region_macro_q25": float(np.quantile(usable_values, 0.25)),
                "region_macro_q75": float(np.quantile(usable_values, 0.75)),
                "positive_support_weighted_mean": float(np.average(
                    usable_values,
                    weights=usable_support if usable_support.sum() > 0 else None,
                )),
                "regions": int(len(usable_values)),
                "total_test_positives": int(usable_support.sum()),
                "primary_reporting": "region_macro_mean_with_per_region_ci",
            })
    pd.DataFrame(balanced_rows).to_csv(
        os.path.join(output_dir, "sampling_comparison_region_balanced_summary.csv"),
        index=False,
    )

    pooled_rows = []
    if {"tn", "fp", "fn", "tp"}.issubset(frame.columns):
        for (model_name, display_name, method), group in frame.groupby(groups):
            tn, fp, fn, tp = (
                int(group[column].sum()) for column in ("tn", "fp", "fn", "tp")
            )
            total = tn + fp + fn + tp
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            observed = (tp + tn) / total if total else 0.0
            expected = (
                ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / (total ** 2)
                if total else 0.0
            )
            kappa = (observed - expected) / (1.0 - expected) if expected < 1.0 else 0.0
            pooled_rows.append({
                "model": model_name,
                "model_display_name": display_name,
                "sampling_method": method,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "pooled_precision": precision,
                "pooled_recall": recall,
                "pooled_f1": f1,
                "pooled_kappa": kappa,
                "note": "secondary_support_dominated_summary_auc_not_poolable_without_predictions",
            })
    pd.DataFrame(pooled_rows).to_csv(
        os.path.join(output_dir, "sampling_comparison_pooled_confusion.csv"),
        index=False,
    )

    paired_rows = []
    for metric in available:
        pivot = frame.pivot(
            index=["model", "test_region"], columns="sampling_method", values=metric
        )
        if {"dwss", "random"}.issubset(pivot.columns):
            for (model_name, test_region), row in pivot.iterrows():
                paired_rows.append({
                    "model": model_name,
                    "test_region": int(test_region),
                    "metric": metric,
                    "dwss": float(row["dwss"]),
                    "random": float(row["random"]),
                    "dwss_minus_random": float(row["dwss"] - row["random"]),
                })
    pd.DataFrame(paired_rows).to_csv(
        os.path.join(output_dir, "sampling_comparison_paired_differences.csv"),
        index=False,
    )
    if paired_rows:
        paired_frame = pd.DataFrame(paired_rows)
        paired_groups = ["model", "metric"]
        grouped_differences = paired_frame.groupby(paired_groups)["dwss_minus_random"]
        paired_summary = grouped_differences.agg(
            ["mean", "std", "median", "min", "max"]
        )
        paired_summary["dwss_better_regions"] = paired_frame.groupby(paired_groups)[
            "dwss_minus_random"
        ].apply(lambda values: int(np.count_nonzero(values > 0)))
        paired_summary["ties"] = paired_frame.groupby(paired_groups)[
            "dwss_minus_random"
        ].apply(lambda values: int(np.count_nonzero(np.isclose(values, 0))))
        def exact_sign_flip_pvalue(values):
            values = np.asarray(values, dtype=np.float64)
            values = values[np.isfinite(values)]
            if not len(values):
                return float("nan")
            observed = abs(float(np.mean(values)))
            combinations = 1 << len(values)
            exceed = 0
            for mask in range(combinations):
                signs = np.asarray([
                    1.0 if mask & (1 << index) else -1.0
                    for index in range(len(values))
                ])
                exceed += abs(float(np.mean(values * signs))) >= observed - 1e-15
            return exceed / combinations

        def paired_mean_ci(values, side):
            values = np.asarray(values, dtype=np.float64)
            values = values[np.isfinite(values)]
            if len(values) < 2:
                return float("nan")
            mean = float(np.mean(values))
            half = float(
                student_t.ppf(0.975, len(values) - 1)
                * np.std(values, ddof=1) / np.sqrt(len(values))
            )
            return mean - half if side == "low" else mean + half

        paired_summary["exact_two_sided_sign_flip_p"] = grouped_differences.apply(
            exact_sign_flip_pvalue
        )
        paired_summary["mean_difference_ci_low"] = grouped_differences.apply(
            lambda values: paired_mean_ci(values, "low")
        )
        paired_summary["mean_difference_ci_high"] = grouped_differences.apply(
            lambda values: paired_mean_ci(values, "high")
        )
        paired_summary["inference_note"] = (
            "exact paired region-level sign-flip test; n_regions is small; "
            "unadjusted exploratory p-value"
        )
        paired_summary.to_csv(
            os.path.join(output_dir, "sampling_comparison_paired_summary.csv")
        )

    # Manuscript tables use equal-region means as the primary estimate.  Each
    # table is emitted for every sampling arm so the DWSS/random control cannot
    # be hidden by selecting only the favourable arm.
    primary = frame.groupby(groups, as_index=False)[available].mean()
    for role, filename in (
        ("machine_learning_comparison", "table_machine_learning.csv"),
        ("deep_learning_comparison", "table_deep_learning.csv"),
        ("ablation", "table_ablation.csv"),
    ):
        selected_names = {
            name for name, spec in MODEL_SPECS.items()
            if spec.paper_role == role
        }
        # LandslideNet is the reference row in all three manuscript tables.
        selected_names.add("landslidenet")
        primary[primary["model"].isin(selected_names)].to_csv(
            os.path.join(output_dir, filename), index=False
        )


def make_loaders(args, factor_paths, feature_transformer, split_regions, split_data,
                 negatives, seed):
    loaders = {}
    for index, split in enumerate(("train", "val", "test"), start=1):
        loaders[split] = make_sparse_loader(
            factor_paths,
            args.regions,
            split_data[split]["positive_points"],
            negatives[split],
            split_regions[split],
            feature_transformer,
            crop_size=args.crop_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            train=split == "train",
            seed=seed + index,
            region_balance=bool(args.region_balance_training and split == "train"),
            region_balance_power=args.region_balance_power,
            augmentation_mode=args.augmentation_mode,
            aspect_period=args.aspect_period,
            training_context_views=(
                args.training_context_views if split == "train" else 1
            ),
            max_supervised_points_per_training_tile=(
                args.max_supervised_points_per_training_tile
                if split == "train" else 0
            ),
        )
    loaders["train_eval"] = make_sparse_loader(
        factor_paths,
        args.regions,
        split_data["train"]["positive_points"],
        negatives["train"],
        split_regions["train"],
        feature_transformer,
        crop_size=args.crop_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train=False,
        seed=seed + 101,
        region_balance=False,
        augmentation_mode="none",
        aspect_period=args.aspect_period,
        training_context_views=1,
        max_supervised_points_per_training_tile=0,
    )
    return loaders


def load_completed_fold_metrics(fold_dir, sampling_methods, model_names, test_region):
    """Return all verified rows, or None when any requested artifact is incomplete."""
    rows = []
    for method in sampling_methods:
        for model_name in model_names:
            spec = MODEL_SPECS[model_name]
            model_dir = Path(fold_dir, method.upper(), f"Model_{model_name}")
            metrics_path = model_dir / "test_metrics.json"
            artifact = model_dir / (
                "best_model_weight.pth" if spec.family == "deep" else "best_model.joblib"
            )
            required = (
                metrics_path,
                artifact,
                model_dir / "test_predictions.npz",
                model_dir / "success_predictions.npz",
                model_dir / "training_history.csv",
            )
            if not all(path.exists() for path in required):
                return None
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if (
                metrics.get("model") != model_name
                or metrics.get("sampling_method") != method
                or int(metrics.get("test_region", -1)) != int(test_region)
                or int(metrics.get("test_evaluations", -1)) != 1
            ):
                raise RuntimeError(
                    f"Completed artifact metadata is inconsistent: {metrics_path}"
                )
            rows.append(metrics)
    return rows


def run(args):
    if hasattr(args, "progress"):
        configure_progress(bool(args.progress))
    params = read_xml_params(args.xml)
    configure_experiment_args(args, params)
    model_selection_source, requested_models = resolve_requested_models(args, params)
    requested_classical = [
        name for name in requested_models if MODEL_SPECS[name].family == "classical"
    ]
    model_dependency_audit = dependency_status(requested_classical)
    console(
        "Step 2/3: contiguous regional holdout experiment | "
        f"model_selection={model_selection_source} | models={len(requested_models)} | "
        f"sampling={args.sampling_methods} | mode="
        f"{'audit' if args.audit_only else 'prepare' if args.prepare_only else 'train'}"
    )
    missing_dependencies = [
        row["required_package"]
        for row in model_dependency_audit.values()
        if not row["available"]
    ]
    if missing_dependencies:
        console(
            "Optional model dependencies not installed: " + ", ".join(missing_dependencies),
            level="WARNING",
        )
    if not (args.audit_only or args.prepare_only):
        require_dependencies(requested_classical)
    inventory = args.inventory or params.get("landslide_inventory")
    regions = args.regions or configured_region_path(params)
    if not inventory:
        raise ValueError(
            "No inventory was provided. Set <landslide_inventory> in XML or use --inventory."
        )
    if not regions:
        raise ValueError(
            "No macro-region raster was provided. Set <macro_region_output> in XML or use --regions."
        )
    args.inventory = normalize_path(inventory)
    args.regions = normalize_path(regions)
    if not os.path.exists(args.inventory):
        raise FileNotFoundError(f"Landslide inventory does not exist: {args.inventory}")
    if not os.path.exists(args.regions):
        raise FileNotFoundError(
            f"Macro-region raster does not exist: {args.regions}. Build and inspect it first with "
            f"1_data_processing.py {args.xml} --regions-only."
        )
    if args.region_names is None:
        region_path = Path(args.regions)
        generated_names = region_path.with_name(f"{region_path.stem}_region_names.csv")
        if generated_names.exists():
            args.region_names = str(generated_names)
    factors_dir = normalize_path(args.factors_dir or params["input_factors_dir"])
    factor_paths = list_factor_paths(factors_dir)
    console(
        f"Inputs: inventory={args.inventory} | regions={args.regions} | "
        f"factors={len(factor_paths)}"
    )
    frequency_ratio_value = getattr(args, "frequency_ratio_factors", None)
    if frequency_ratio_value is None:
        frequency_ratio_value = params.get("frequency_ratio_factors", "")
    frequency_ratio_specs = parse_frequency_ratio_specs(
        frequency_ratio_value,
        factor_paths,
    )
    frequency_ratio_smoothing = getattr(args, "frequency_ratio_smoothing", None)
    if frequency_ratio_smoothing is None:
        frequency_ratio_smoothing = float(params.get("frequency_ratio_smoothing", 0.5))
    frequency_ratio_log2_clip = getattr(args, "frequency_ratio_log2_clip", None)
    if frequency_ratio_log2_clip is None:
        frequency_ratio_log2_clip = float(params.get("frequency_ratio_log2_clip", 4.0))

    if args.num_bands is None:
        args.num_bands = int(params["num_bands"])
    if len(factor_paths) != args.num_bands:
        raise ValueError(
            f"Found {len(factor_paths)} factor rasters but num_bands={args.num_bands}."
        )
    args.crop_size = args.crop_size or int(params["crop_size"])
    args.batch_size = args.batch_size or int(params["batch_size"])
    args.num_workers = args.num_workers if args.num_workers is not None else int(params["num_workers"])
    args.num_epochs = args.num_epochs or int(params["num_epochs"])
    args.lr = args.lr or float(params["lr"])
    args.patience = args.patience or int(params["patience"])
    args.weight_decay = args.weight_decay or float(params["weight_decay"])
    args.device_ids = parse_device_ids(args.device_ids or params["device_ids"])
    if args.minimum_epochs > args.num_epochs:
        raise ValueError("minimum_epochs cannot exceed num_epochs.")
    if args.minimum_lr >= args.lr:
        raise ValueError("minimum_lr must be smaller than the initial learning rate.")
    low_support_warning = getattr(args, "low_support_warning", None)
    if low_support_warning is None:
        low_support_warning = int(params.get("regional_low_support_warning", 30))
    adequate_support = getattr(args, "adequate_support", None)
    if adequate_support is None:
        adequate_support = int(params.get("regional_adequate_support", 100))
    if not 1 <= low_support_warning <= adequate_support:
        raise ValueError(
            "Require 1 <= regional_low_support_warning <= regional_adequate_support."
        )
    preferred_validation_positives = getattr(
        args, "preferred_validation_positives", None
    )
    if preferred_validation_positives is None:
        preferred_validation_positives = int(
            params.get("regional_preferred_validation_positives", adequate_support)
        )
    if preferred_validation_positives < 1:
        raise ValueError("regional_preferred_validation_positives must be positive.")
    bootstrap_replicates = getattr(args, "bootstrap_replicates", None)
    if bootstrap_replicates is None:
        bootstrap_replicates = int(params.get("regional_bootstrap_replicates", 1000))
    bootstrap_confidence = getattr(args, "bootstrap_confidence", None)
    if bootstrap_confidence is None:
        bootstrap_confidence = float(params.get("regional_bootstrap_confidence", 0.95))
    if bootstrap_replicates < 0 or not 0 < bootstrap_confidence < 1:
        raise ValueError("Invalid regional bootstrap configuration.")
    region_balance_value = getattr(args, "region_balance_training", None)
    if region_balance_value is None:
        region_balance_value = params.get("regional_balance_training_regions", "true")
    args.region_balance_training = parse_bool(region_balance_value, True)
    region_balance_power = getattr(args, "region_balance_power", None)
    if region_balance_power is None:
        region_balance_power = float(params.get("regional_balance_power", 0.5))
    if not 0 <= region_balance_power <= 1:
        raise ValueError("regional_balance_power must be in [0, 1].")
    args.region_balance_power = region_balance_power

    grid = validate_aligned_rasters(args.regions, factor_paths)
    if not is_vector_inventory(args.inventory):
        validate_aligned_rasters(args.regions, [args.inventory])
    (
        inventory_positives,
        inventory_positive_counts,
        background_counts,
        inventory_audit,
    ) = collect_positive_points(
        args.inventory,
        args.regions,
        positive_value=args.positive_value,
        chunk_size=args.raster_chunk_size,
        return_audit=True,
    )
    region_names = load_region_names(args.region_names)
    available_regions = sorted(set(inventory_positive_counts) | set(background_counts))
    region_ids = sorted(set(args.region_ids or available_regions))
    if len(region_ids) < 3:
        raise ValueError("Nested regional hold-out requires at least three macro-regions.")
    connectivity_audit = audit_region_connectivity(args.regions, region_ids)
    region_provenance = load_region_provenance(args.regions)
    validation_strategy = getattr(args, "validation_strategy", None)
    if validation_strategy is None:
        validation_strategy = params.get("regional_validation_strategy", "cyclic")
    minimum_training_positives = getattr(args, "min_inner_training_positives", None)
    if minimum_training_positives is None:
        minimum_training_positives = int(
            params.get("regional_min_inner_training_positives", 1)
        )
    positive_features, positive_valid = read_point_features(
        factor_paths,
        inventory_positives,
        tile_size=args.feature_tile_size,
    )
    positives = inventory_positives[positive_valid]
    positive_features = positive_features[positive_valid]
    positives_selected = np.isin(positives[:, 2], region_ids)
    positives = positives[positives_selected]
    positive_features = positive_features[positives_selected]
    valid_region_values, valid_region_counts = np.unique(positives[:, 2], return_counts=True)
    factor_valid_positive_counts = {
        int(region_id): int(count)
        for region_id, count in zip(valid_region_values, valid_region_counts)
    }
    console(
        "Valid landslide pixels by region: "
        + ", ".join(
            f"R{region_id}={factor_valid_positive_counts.get(region_id, 0):,}"
            for region_id in region_ids
        )
    )
    validation_map = build_validation_map(
        region_ids,
        args.validation_map,
        strategy=validation_strategy,
        positive_counts=factor_valid_positive_counts,
        minimum_training_positives=minimum_training_positives,
        preferred_validation_positives=preferred_validation_positives,
    )
    missing_background = [
        region_id for region_id in region_ids if background_counts.get(region_id, 0) == 0
    ]
    insufficient_positive = [
        region_id for region_id in region_ids
        if factor_valid_positive_counts.get(region_id, 0) < args.min_region_positives
    ]
    if missing_background:
        raise ValueError(f"Selected regions lack background cells: {missing_background}")
    if insufficient_positive:
        raise ValueError(
            f"Selected regions have fewer than {args.min_region_positives} "
            "factor-valid positives: "
            f"{insufficient_positive}. Do not silently remove them; revise and justify the "
            "scientific region definition or the preregistered --region-ids list."
        )
    for region_id in region_ids:
        count = factor_valid_positive_counts.get(region_id, 0)
        if count < adequate_support:
            tier = "very low" if count < low_support_warning else "limited"
            console(
                f"Macro-region {region_id} has only {count} valid landslide pixels ({tier} support); "
                "confidence intervals must be reported, and regions must not be ranked by point estimates.",
                level="WARNING",
            )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = normalize_path(args.output_dir) if args.output_dir else os.path.join(
        normalize_path(params["train_output"]),
        f"Continuous_Regional_Holdout_{timestamp}",
    )
    output_path = Path(output_dir)
    resume = bool(getattr(args, "resume", False))
    existing_entries = list(output_path.iterdir()) if output_path.exists() else []
    snapshot_path = output_path / "configuration_snapshot.xml"
    current_configuration = Path(args.xml).expanduser().read_bytes()
    if resume:
        if not snapshot_path.exists():
            raise RuntimeError(
                "--resume requires an existing configuration_snapshot.xml in the output."
            )
        if snapshot_path.read_bytes() != current_configuration:
            raise RuntimeError(
                "The XML differs from the existing run snapshot; refusing an unsafe resume."
            )
        registry_path = output_path / "model_registry.json"
        if registry_path.exists():
            previous_registry = json.loads(registry_path.read_text(encoding="utf-8"))
            previous_models = [
                row["name"] for row in previous_registry.get("requested_models", [])
            ]
            if previous_models != requested_models:
                raise RuntimeError(
                    f"Requested models {requested_models} differ from resumed run "
                    f"models {previous_models}."
                )
    elif existing_entries:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Choose a new directory or "
            "use --resume with the identical XML/model suite."
        )
    os.makedirs(output_dir, exist_ok=True)
    if not snapshot_path.exists():
        snapshot_path.write_bytes(current_configuration)
    software_path = Path(output_dir, "software_environment.json")
    if resume and software_path.exists():
        software_path = Path(output_dir, f"software_environment_resume_{timestamp}.json")
    environment = software_environment()
    with open(software_path, "w", encoding="utf-8") as handle:
        json.dump(environment, handle, indent=2, ensure_ascii=False)
    console(
        f"Runtime device: CUDA={environment['cuda_available']} | "
        f"GPU count={environment['cuda_device_count']} | "
        f"PyTorch={environment['packages'].get('torch')} | output={output_dir}"
    )
    with open(os.path.join(output_dir, "model_registry.json"), "w", encoding="utf-8") as handle:
        json.dump({
            "model_selection_source": model_selection_source,
            "requested_models": model_specs(requested_models),
            "optional_dependency_status": model_dependency_audit,
            "reporting_rule": (
                "task_adapter and paper_guided_task_adapter implementations must be "
                "reported as adaptations, never as official author-code reproductions"
            ),
        }, handle, indent=2, ensure_ascii=False)
    write_region_audit(
        os.path.join(output_dir, "macro_region_inventory_audit.csv"),
        region_ids,
        region_names,
        factor_valid_positive_counts,
        inventory_positive_counts,
        background_counts,
        connectivity_audit,
        low_support_warning,
        adequate_support,
    )
    with open(os.path.join(output_dir, "macro_region_connectivity_audit.json"), "w",
              encoding="utf-8") as handle:
        json.dump(connectivity_audit, handle, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "macro_region_source_audit.json"), "w",
              encoding="utf-8") as handle:
        json.dump(region_provenance, handle, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "landslide_inventory_audit.json"), "w",
              encoding="utf-8") as handle:
        json.dump(inventory_audit, handle, indent=2, ensure_ascii=False)
    write_validation_support_audit(
        os.path.join(output_dir, "regional_split_support_audit.csv"),
        region_ids,
        validation_map,
        factor_valid_positive_counts,
        preferred_validation_positives,
    )
    if not getattr(args, "no_plots", False):
        plot_macro_region_layout(
            os.path.join(output_dir, "macro_region_layout.png"),
            args.regions,
            positives,
            region_ids,
            region_names,
        )
    validation_mapping_support = {
        test_region: {
            "test_positive_pixels": factor_valid_positive_counts.get(test_region, 0),
            "validation_region": validation_region,
            "validation_positive_pixels": factor_valid_positive_counts.get(validation_region, 0),
            "inner_training_positive_pixels": int(sum(
                factor_valid_positive_counts.get(region_id, 0)
                for region_id in region_ids
                if region_id not in {test_region, validation_region}
            )),
        }
        for test_region, validation_region in validation_map.items()
    }
    outer_test_regions = list(getattr(args, "test_region_ids", None) or region_ids)
    invalid_outer_tests = sorted(set(outer_test_regions) - set(region_ids))
    if invalid_outer_tests:
        raise ValueError(f"Unknown --test-region-ids: {invalid_outer_tests}")
    if len(set(outer_test_regions)) != len(outer_test_regions):
        raise ValueError("--test-region-ids must not contain duplicates.")
    for test_region, validation_region in validation_map.items():
        if test_region == validation_region or validation_region not in region_ids:
            raise ValueError(
                f"Invalid validation mapping {test_region} -> {validation_region}."
            )

    protocol = {
        "protocol": "nested_leave_one_continuous_macro_region_out",
        "inventory": args.inventory,
        "inventory_audit": inventory_audit,
        "macro_regions": args.regions,
        "macro_region_provenance": region_provenance,
        "macro_region_connectivity": connectivity_audit,
        "factor_paths": factor_paths,
        "frequency_ratio_factors": [
            {
                "factor_name": spec.factor_name,
                "factor_index": spec.factor_index,
                "normalized_levels": spec.normalized_levels,
            }
            for spec in frequency_ratio_specs
        ],
        "frequency_ratio_fit_scope": "inner_training_regions_only",
        "frequency_ratio_unknown_category_policy": "neutral_log_frequency_ratio",
        "frequency_ratio_configuration": {
            "smoothing": frequency_ratio_smoothing,
            "log2_clip": frequency_ratio_log2_clip,
        },
        "raster_grid": {
            "height": grid["height"],
            "width": grid["width"],
            "crs": str(grid["crs"]),
            "transform": list(grid["transform"]),
        },
        "region_ids": region_ids,
        "outer_test_regions_executed": outer_test_regions,
        "factor_valid_positive_counts": factor_valid_positive_counts,
        "regional_support_reporting_thresholds": {
            "very_low_below": low_support_warning,
            "adequate_at_or_above": adequate_support,
            "policy": "never redraw frozen boundaries from these counts",
        },
        "validation_map": validation_map,
        "validation_mapping_strategy": (
            "explicit_csv" if args.validation_map else validation_strategy
        ),
        "validation_mapping_uses_only_pre_model_class_support": bool(
            not args.validation_map and validation_strategy == "support_aware"
        ),
        "validation_mapping_excludes_test_region_support": bool(
            not args.validation_map and validation_strategy == "support_aware"
        ),
        "minimum_inner_training_positives": minimum_training_positives,
        "preferred_validation_positives": preferred_validation_positives,
        "support_aware_validation_policy": (
            "smallest whole region meeting preferred validation support while "
            "preserving the minimum training support; otherwise strongest feasible region"
            if not args.validation_map and validation_strategy == "support_aware"
            else None
        ),
        "validation_mapping_support": validation_mapping_support,
        "normalization_fit_scope": "all factor-valid pixels in inner-training regions only",
        "dwss_fit_scope": "inner_training_regions_only",
        "dwss_computational_policy": {
            "theta_min": args.theta_min,
            "n_strata": args.n_strata,
            "stratum_weight_formula": "mean_zeta_k / sum_j(mean_zeta_j)",
            "minimum_per_stratum": 0,
            "maximum_kde_prototypes": args.max_prototypes,
            "kde_chunk_size": args.kde_chunk_size,
            "candidate_multiplier": args.candidate_multiplier,
            "candidate_minimum": args.candidate_minimum,
            "candidate_maximum": args.candidate_maximum,
            "training_candidate_minimum": args.training_candidate_minimum,
            "training_candidate_maximum": args.training_candidate_maximum,
            "adaptive_candidate_maximum": args.adaptive_candidate_maximum,
            "adaptive_batch_size": args.adaptive_batch_size,
            "screening_neighbors": args.screening_neighbors,
            "strict_stratum_capacity_policy": (
                "uniform adaptive proposals with safe KDE upper-bound pruning; "
                "exact KDE for every screen survivor; never redistribute a "
                "deficient stratum quota"
            ),
            "prototype_approximation_disclosure": (
                "A positive prototype cap is a predeclared uniform training-only "
                "approximation and is identical for all model comparisons."
                if args.max_prototypes else "all inner-training positives used"
            ),
            "background_computation_disclosure": (
                "Natural breaks are fitted on the predeclared uniform initial "
                "inner-training background pool. If a strict stratum quota is "
                "short, additional background pixels are proposed uniformly; "
                "conditional stratum means are refined from those exact-KDE "
                "draws. This is a reproducible Monte-Carlo implementation of the "
                "full-pixel rule, not an exhaustive KDE evaluation of every "
                "national background pixel."
            ),
            "rbs_environmental_similarity": {
                "prototype": (
                    "joint multivariate Gaussian KDE in the complete fold-fitted "
                    "min-max-scaled factor space"
                ),
                "bandwidth": "Scott",
                "similarity": "joint density divided by its training-fitted maximum",
                "divergence": "one minus normalized joint density",
            },
        },
        "candidate_pool_policy": (
            "base pools are uniform without replacement within each split and "
            "selected before reading factor values; DWSS may issue audited "
            "additional uniform inner-training proposals solely to meet strict "
            "stratum capacities"
        ),
        "random_negative_policy": (
            "uniform without replacement from the base uniform training pool"
        ),
        "held_out_negative_policy": (
            "one fixed uniform validation/test negative set per fold, shared unchanged by "
            "DWSS and random training-sampling arms"
        ),
        "sampling_method_varies_only_training_negatives": True,
        "model_selection_scope": "whole validation macro-region only",
        "threshold_selection_scope": "whole validation macro-region only",
        "test_policy": "single evaluation after all model and threshold selection",
        "evaluation_tta": bool(args.eval_tta),
        "test_uncertainty_policy": {
            "method": "class_stratified_spatial_tile_block_bootstrap",
            "spatial_block_pixels": args.crop_size,
            "reason": "avoid treating adjacent supervised pixels as independent draws",
            "replicates": bootstrap_replicates,
            "confidence": bootstrap_confidence,
            "applied_only_after_final_test_prediction": True,
        },
        "training_region_balance": {
            "enabled": args.region_balance_training,
            "scope": "inner_training_samples_only",
            "policy": (
                "class_prior_preserving_inverse_region_frequency_power"
            ),
            "normalization": "within_class_mean_weight_equals_one",
            "preserves_designed_one_to_one_class_prior": True,
            "balance_power": args.region_balance_power,
        },
        "training_supervision_expansion": {
            "scope": "inner_training_samples_only",
            "method": "label_preserving_shifted_raster_context_views",
            "context_view_count": args.training_context_views,
            "max_supervised_points_per_optimizer_item": (
                args.max_supervised_points_per_training_tile
            ),
            "unique_coordinates_and_labels_unchanged": True,
            "positive_and_negative_views_expanded_equally": True,
            "validation_and_test_expansion": False,
            "optimizer_batching": (
                "supervision_weight_mass_balanced_without_resampling"
            ),
            "every_optimizer_item_once_per_epoch": True,
        },
        "sampling_methods": args.sampling_methods,
        "sampling_configuration": {
            "positive_inventory_value": args.positive_value,
            "negative_to_positive_ratio": args.negative_ratio,
            "minimum_positive_pixels_per_region": args.min_region_positives,
        },
        "experiment_seed": args.seed,
        "model_selection_source": model_selection_source,
        "models": model_specs(requested_models),
        "model_dependency_audit": model_dependency_audit,
        "comparison_fairness": (
            "Within each fold and sampling arm every model receives the same positives, "
            "selected negatives, frozen feature transform, validation/test samples, and "
            "metric implementation. Model-specific randomness is fixed across sampling arms."
        ),
        "hyperparameters_fixed_before_outer_test": {
            "num_epochs": args.num_epochs,
            "minimum_epochs": args.minimum_epochs,
            "learning_rate": args.lr,
            "learning_rate_scheduler": (
                "linear_warmup_then_validation_reduce_on_plateau"
            ),
            "learning_rate_warmup_epochs": args.lr_warmup_epochs,
            "learning_rate_plateau_patience": args.lr_plateau_patience,
            "learning_rate_plateau_factor": args.lr_plateau_factor,
            "minimum_learning_rate": args.minimum_lr,
            "patience": args.patience,
            "selection_min_delta": args.selection_min_delta,
            "weight_decay": args.weight_decay,
            "crop_size": args.crop_size,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "device_ids": args.device_ids,
            "classical_iterations": args.classical_iterations,
            "classical_n_jobs": args.classical_n_jobs,
            "selection_metric": args.selection_metric,
            "threshold_metric": args.threshold_metric,
            "threshold_selection": (
                "exact_validation_scores_with_precision_recall_balance_tiebreak"
            ),
            "threshold_score_tolerance": args.threshold_score_tolerance,
            "sparse_objective_normalization": (
                "dataset_global_weight_mass_preserving_across_uneven_batches"
            ),
            "optimizer_batching": (
                "supervision_weight_mass_balanced_without_resampling"
            ),
            "domain_risk_regularization": (
                "class_conditional_V-REx_source_macro_region_risk_variance"
            ),
            "domain_risk_variance_weight": args.domain_risk_variance_weight,
            "domain_risk_warmup_epochs": args.domain_risk_warmup_epochs,
            "training_context_views": args.training_context_views,
            "max_supervised_points_per_training_tile": (
                args.max_supervised_points_per_training_tile
            ),
            "ema_decay": args.ema_decay,
            "ema_start_epoch": args.ema_start_epoch,
            "augmentation_mode": args.augmentation_mode,
            "aspect_period": args.aspect_period,
            "evaluation_tta": bool(args.eval_tta),
        },
    }
    protocol_path = Path(output_dir, "validation_protocol.json")
    resume_contract_keys = (
        "inventory", "macro_regions", "factor_paths", "frequency_ratio_factors",
        "frequency_ratio_configuration",
        "region_ids", "outer_test_regions_executed", "validation_map",
        "minimum_inner_training_positives", "preferred_validation_positives",
        "normalization_fit_scope", "dwss_computational_policy",
        "held_out_negative_policy", "training_region_balance", "sampling_methods",
        "sampling_configuration", "experiment_seed", "test_uncertainty_policy",
        "hyperparameters_fixed_before_outer_test",
    )
    if resume:
        if not protocol_path.exists():
            raise RuntimeError("--resume requires validation_protocol.json.")
        previous_protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        previous_contract = {
            key: previous_protocol.get(key) for key in resume_contract_keys
        }
        current_contract = json.loads(json.dumps({
            key: protocol.get(key) for key in resume_contract_keys
        }, sort_keys=True))
        if previous_contract != current_contract:
            raise RuntimeError(
                "Resolved split/sampling/training settings differ from the existing "
                "validation protocol; refusing an unsafe resume."
            )
    else:
        protocol_path.write_text(
            json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    if args.audit_only:
        console(f"Audit completed; results directory: {output_dir}")
        return

    all_metrics = []
    jobs_per_fold = (
        len(args.sampling_methods)
        if args.prepare_only
        else len(args.sampling_methods) * len(requested_models)
    )
    overall_progress = track(
        total=len(outer_test_regions) * jobs_per_fold,
        desc="Overall regional-experiment progress",
        unit="job",
        leave=True,
    )
    for test_region in outer_test_regions:
        fold_index = region_ids.index(test_region) + 1
        validation_region = validation_map[test_region]
        training_regions = [
            region_id for region_id in region_ids
            if region_id not in {test_region, validation_region}
        ]
        split_regions = {
            "train": training_regions,
            "val": [validation_region],
            "test": [test_region],
        }
        split_region_sets = {name: set(values) for name, values in split_regions.items()}
        splits_are_region_disjoint = all(
            split_region_sets[left].isdisjoint(split_region_sets[right])
            for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
        )
        if not splits_are_region_disjoint:
            raise RuntimeError(f"Region leakage detected before fold {fold_index} preparation.")
        console(
            f"Starting Fold {fold_index}/{len(region_ids)} | test={test_region} | "
            f"val={validation_region} | train={training_regions}"
        )
        fold_dir = os.path.join(output_dir, f"Fold_{fold_index}_TestRegion_{test_region}")
        os.makedirs(fold_dir, exist_ok=True)
        if resume and not args.prepare_only:
            completed_fold = load_completed_fold_metrics(
                fold_dir, args.sampling_methods, requested_models, test_region
            )
            if completed_fold is not None:
                console(
                    f"Resume: fold {fold_index} test region {test_region} is complete; "
                    "verified artifacts and skipped fold preparation/training."
                )
                all_metrics.extend(completed_fold)
                write_comparison_tables(all_metrics, output_dir)
                overall_progress.update(jobs_per_fold)
                continue

        fold_progress = track(
            total=8,
            desc=f"Fold {fold_index} data preparation",
            unit="stage",
        )

        # Fit-scope is explicit here: factor ranges are accumulated with an
        # inner-training-region mask for this fold. Validation and test regions
        # never contribute a value to these statistics.
        training_factor_ranges = compute_region_factor_ranges(
            factor_paths,
            args.regions,
            training_regions,
            chunk_size=args.raster_chunk_size,
        )
        fold_progress.update(1)
        fold_progress.set_postfix_str("Candidate pool and point factors", refresh=False)

        split_data = {}
        for split_index, split in enumerate(("train", "val", "test"), start=1):
            positive_points, positive_raw = select_region_points(
                positives,
                positive_features,
                split_regions[split],
            )
            if len(positive_points) == 0:
                raise RuntimeError(f"No factor-valid positives remain for {split} in fold {fold_index}.")
            candidate_points, candidate_raw, required, candidate_audit = sample_candidate_pool(
                args,
                factor_paths,
                split_regions[split],
                len(positive_points),
                args.seed + fold_index * 1000 + split_index * 100,
                split,
            )
            split_data[split] = {
                "positive_points": positive_points,
                "positive_raw": positive_raw,
                "candidate_points": candidate_points,
                "candidate_raw": candidate_raw,
                "required_negatives": required,
                "candidate_audit": candidate_audit,
            }
            fold_progress.update(1)

        fold_progress.set_postfix_str("Training-region FR statistics", refresh=False)
        normalizer = FrozenMinMaxNormalizer.from_region_ranges(
            factor_paths,
            training_factor_ranges,
            training_regions,
        )
        category_area_counts, category_count_diagnostics = compute_training_category_counts(
            factor_paths,
            args.regions,
            training_regions,
            frequency_ratio_specs,
            chunk_size=args.raster_chunk_size,
        )
        feature_transformer = build_fold_feature_transformer(
            normalizer,
            frequency_ratio_specs,
            category_area_counts,
            split_data["train"]["positive_raw"],
            smoothing=frequency_ratio_smoothing,
            log2_clip=frequency_ratio_log2_clip,
        )
        fold_progress.update(1)
        write_frequency_ratio_mapping(
            os.path.join(fold_dir, "frequency_ratio_mapping.csv"),
            feature_transformer,
        )

        dwss = None
        fold_progress.set_postfix_str("Within-fold DWSS fitting", refresh=False)
        if "dwss" in args.sampling_methods:
            dwss = FrozenDWSS.fit(
                split_data["train"]["positive_raw"],
                split_data["train"]["candidate_raw"],
                feature_transformer,
                theta_min=args.theta_min,
                n_strata=args.n_strata,
                weight_power=args.weight_power,
                seed=args.seed + fold_index * 10000 + 7,
                max_prototypes=args.max_prototypes,
                kde_chunk_size=args.kde_chunk_size,
            )
            augment_dwss_training_candidates(
                args,
                factor_paths,
                training_regions,
                split_data["train"],
                dwss,
                args.seed + fold_index * 10000 + 17,
            )
        fold_progress.update(1)

        for split in split_data.values():
            split["positive_normalized"] = feature_transformer.transform(
                split["positive_raw"]
            )
            split["candidate_normalized"] = feature_transformer.transform(
                split["candidate_raw"]
            )
        fold_progress.update(1)

        shared_evaluation_negatives = {}
        shared_evaluation_audit = {}
        for split, offset in (("val", 701), ("test", 907)):
            evaluation_seed = args.seed + fold_index * 10000 + offset
            shared_evaluation_negatives[split] = choose_random_points(
                split_data[split]["candidate_points"],
                split_data[split]["required_negatives"],
                evaluation_seed,
            )
            shared_evaluation_audit[split] = {
                "selection": "uniform_without_replacement",
                "seed": evaluation_seed,
                "candidate_pool": int(len(split_data[split]["candidate_points"])),
                "selected": int(len(shared_evaluation_negatives[split])),
                "shared_across_sampling_arms": True,
            }

        fold_audit = {
            "fold": fold_index,
            "test_region": test_region,
            "test_region_name": region_names.get(test_region, f"region_{test_region}"),
            "validation_region": validation_region,
            "validation_region_name": region_names.get(validation_region, f"region_{validation_region}"),
            "split_positive_support": validation_mapping_support[test_region],
            "inner_training_regions": training_regions,
            "normalization_fit_regions": training_regions,
            "normalization_fit_valid_pixel_counts": {
                Path(path).stem: int(sum(
                    training_factor_ranges[region_id]["counts"][factor_index]
                    for region_id in training_regions
                ))
                for factor_index, path in enumerate(factor_paths)
            },
            "dwss_fit_regions": training_regions,
            "splits_are_region_disjoint": splits_are_region_disjoint,
            "candidate_audit": {
                split: split_data[split]["candidate_audit"] for split in split_data
            },
            "feature_transformer": feature_transformer.to_dict(),
            "frequency_ratio_category_audit": category_count_diagnostics,
            "dwss": dwss.to_dict() if dwss is not None else None,
            "shared_evaluation_negatives": shared_evaluation_audit,
            "paired_sampling_control_audit": "paired_sampling_control_audit.json",
        }
        with open(os.path.join(fold_dir, "fold_leakage_audit.json"), "w", encoding="utf-8") as handle:
            json.dump(fold_audit, handle, indent=2, ensure_ascii=False)
        fold_progress.update(1)
        fold_progress.set_postfix_str("Completed", refresh=False)
        fold_progress.close()

        method_seed_offsets = {"dwss": 101, "random": 211}
        fold_seed = args.seed + fold_index * 10000
        prepared_method_negatives = {}
        prepared_sampling_diagnostics = {}
        for method in args.sampling_methods:
            negatives, sampling_diagnostics = method_negative_samples(
                method,
                split_data,
                dwss,
                args,
                fold_seed + method_seed_offsets[method],
                shared_evaluation_negatives,
            )
            prepared_method_negatives[method] = negatives
            prepared_sampling_diagnostics[method] = sampling_diagnostics

        model_seeds = {
            model_name: int(
                fold_seed
                + (list(MODEL_SPECS).index(model_name) + 1) * 100
            )
            for model_name in requested_models
        }
        pairing_audit = {
            "protocol": "paired_dwss_vs_uniform_random_training_negative_control",
            "fold": int(fold_index),
            "test_region": int(test_region),
            "validation_region": int(validation_region),
            "inner_training_regions": list(map(int, training_regions)),
            "sampling_arms": list(args.sampling_methods),
            "paired_control_complete": {
                "dwss",
                "random",
            }.issubset(args.sampling_methods),
            "only_manipulated_variable": (
                "inner_training_negative_selection_rule"
            ),
            "shared_inputs_sha256": {
                "feature_transformer": _json_sha256(
                    feature_transformer.to_dict()
                ),
                "positive_coordinates_by_split": {
                    split: _array_sha256(
                        split_data[split]["positive_points"], "<i8"
                    )
                    for split in ("train", "val", "test")
                },
                "candidate_pool_coordinates_by_split": {
                    split: _array_sha256(
                        split_data[split]["candidate_points"], "<i8"
                    )
                    for split in ("train", "val", "test")
                },
                "base_uniform_candidate_coordinates_by_split": {
                    split: _array_sha256(
                        split_data[split]["candidate_points"][
                            :int(
                                split_data[split].get(
                                    "base_uniform_candidate_count",
                                    len(split_data[split]["candidate_points"]),
                                )
                            )
                        ],
                        "<i8",
                    )
                    for split in ("train", "val", "test")
                },
                "dwss_adaptive_retained_training_coordinates": _array_sha256(
                    split_data["train"]["candidate_points"][
                        int(
                            split_data["train"].get(
                                "base_uniform_candidate_count",
                                len(split_data["train"]["candidate_points"]),
                            )
                        ):
                    ],
                    "<i8",
                ),
                "candidate_pool_transformed_features_by_split": {
                    split: _array_sha256(
                        split_data[split]["candidate_normalized"], "<f8"
                    )
                    for split in ("train", "val", "test")
                },
                "shared_validation_negatives": _array_sha256(
                    shared_evaluation_negatives["val"], "<i8"
                ),
                "shared_test_negatives": _array_sha256(
                    shared_evaluation_negatives["test"], "<i8"
                ),
            },
            "model_seed_by_model": model_seeds,
            "training_negative_selection_by_arm": {
                method: {
                    "rule": (
                        "fold_fitted_dwss_stratified_selection"
                        if method == "dwss"
                        else "uniform_without_replacement"
                    ),
                    "selected_coordinates_sha256": _array_sha256(
                        prepared_method_negatives[method]["train"], "<i8"
                    ),
                }
                for method in args.sampling_methods
            },
            "evaluation_set_identity_checks": {
                method: {
                    split: bool(
                        np.array_equal(
                            prepared_method_negatives[method][split],
                            shared_evaluation_negatives[split],
                        )
                    )
                    for split in ("val", "test")
                }
                for method in args.sampling_methods
            },
            "assertions": {
                "same_positive_coordinates": True,
                "same_inner_training_background_universe": True,
                "random_training_pool_is_uniform": True,
                "dwss_adaptive_proposals_are_uniform": True,
                "adaptive_search_is_part_of_dwss_selection_rule": True,
                "same_fold_feature_transformer": True,
                "same_validation_negatives": True,
                "same_test_negatives": True,
                "same_model_seed_for_each_model_across_arms": True,
                "dwss_fitted_once_per_outer_fold_on_inner_training_only": (
                    dwss is not None
                ),
            },
        }
        with open(
            os.path.join(fold_dir, "paired_sampling_control_audit.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(pairing_audit, handle, indent=2, ensure_ascii=False)

        for method in args.sampling_methods:
            method_dir = os.path.join(fold_dir, method.upper())
            os.makedirs(method_dir, exist_ok=True)
            negatives = prepared_method_negatives[method]
            sampling_diagnostics = prepared_sampling_diagnostics[method]
            write_sample_manifest(
                os.path.join(method_dir, "sample_manifest.csv"),
                grid["transform"],
                method,
                fold_index,
                region_names,
                split_data,
                negatives,
            )
            with open(os.path.join(method_dir, "sampling_diagnostics.json"), "w", encoding="utf-8") as handle:
                json.dump(sampling_diagnostics, handle, indent=2, ensure_ascii=False)
            if args.region_balance_training:
                train_weights, region_weighting_audit = region_class_balanced_weights(
                    split_data["train"]["positive_points"],
                    negatives["train"],
                    split_regions["train"],
                    balance_power=args.region_balance_power,
                )
            else:
                train_weights = np.ones(
                    len(split_data["train"]["positive_points"])
                    + len(negatives["train"]),
                    dtype=np.float32,
                )
                region_weighting_audit = {
                    "policy": "uniform_sample_weight",
                    "normalization": "all_weights_equal_one",
                }
            with open(os.path.join(method_dir, "training_region_weighting.json"), "w",
                      encoding="utf-8") as handle:
                json.dump(region_weighting_audit, handle, indent=2, ensure_ascii=False)

            console(
                f"Fold {fold_index}/{len(region_ids)} | test={test_region} | "
                f"val={validation_region} | train={training_regions} | method={method}"
            )
            for split in ("train", "val", "test"):
                console(
                    f"  {split}: positives={len(split_data[split]['positive_points'])}, "
                    f"negatives={len(negatives[split])}, regions={split_regions[split]}"
                )
            if args.prepare_only:
                overall_progress.update(1)
                continue

            classical_arrays = classical_points = None
            if requested_classical:
                classical_arrays, classical_points = make_classical_arrays(
                    split_data, negatives
                )

            for model_name in requested_models:
                spec = MODEL_SPECS[model_name]
                model_seed = model_seeds[model_name]
                model_dir = os.path.join(method_dir, f"Model_{model_name}")
                os.makedirs(model_dir, exist_ok=True)
                metrics_path = os.path.join(model_dir, "test_metrics.json")
                expected_model_artifact = os.path.join(
                    model_dir,
                    "best_model_weight.pth" if spec.family == "deep" else "best_model.joblib",
                )
                if resume and all(os.path.exists(path) for path in (
                    metrics_path,
                    expected_model_artifact,
                    os.path.join(model_dir, "test_predictions.npz"),
                    os.path.join(model_dir, "success_predictions.npz"),
                    os.path.join(model_dir, "training_history.csv"),
                )):
                    with open(metrics_path, encoding="utf-8") as handle:
                        completed_metrics = json.load(handle)
                    if (
                        completed_metrics.get("model") != model_name
                        or completed_metrics.get("sampling_method") != method
                        or int(completed_metrics.get("test_region", -1)) != test_region
                        or int(completed_metrics.get("test_evaluations", -1)) != 1
                    ):
                        raise RuntimeError(
                            f"Completed artifact metadata is inconsistent: {metrics_path}"
                        )
                    console(f"  resume: verified and skipped completed model {model_name}")
                    all_metrics.append(completed_metrics)
                    write_comparison_tables(all_metrics, output_dir)
                    overall_progress.update(1)
                    continue
                console(
                    f"  model={spec.display_name} ({model_name}); "
                    f"implementation={spec.implementation}; seed={model_seed}"
                )
                metadata = {
                    "model": spec.to_dict(),
                    "fold": fold_index,
                    "sampling_method": method,
                    "test_region": test_region,
                    "validation_region": validation_region,
                    "training_regions": training_regions,
                    "num_bands": args.num_bands,
                    "factor_paths": factor_paths,
                    "feature_transformer_source": os.path.relpath(
                        os.path.join(fold_dir, "fold_leakage_audit.json"), model_dir
                    ),
                    "seed": model_seed,
                    "test_evaluation_policy": "exactly_once_after_validation_selection",
                }
                with open(os.path.join(model_dir, "model_metadata.json"), "w",
                          encoding="utf-8") as handle:
                    json.dump(metadata, handle, indent=2, ensure_ascii=False)

                if spec.family == "deep":
                    set_global_seed(model_seed)
                    loaders = make_loaders(
                        args,
                        factor_paths,
                        feature_transformer,
                        split_regions,
                        split_data,
                        negatives,
                        model_seed,
                    )
                    logger = setup_fold_logger(
                        model_dir, f"{fold_index}_{method}_{model_name}"
                    )
                    model = build_deep_model(model_name, args.num_bands)
                    metadata["trainable_parameters"] = int(sum(
                        parameter.numel()
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ))
                    training_supervision_audit = loaders["train"].supervision_audit
                    metadata["training_supervision_audit"] = training_supervision_audit
                    metadata["training_region_weighting_audit"] = (
                        loaders["train"].region_weighting_audit
                    )
                    console(
                        f"  Deep-model parameters={metadata['trainable_parameters']:,} | "
                        f"train/val/test batches={len(loaders['train'])}/"
                        f"{len(loaders['val'])}/{len(loaders['test'])} | "
                        f"unique samples="
                        f"{training_supervision_audit['unique_real_sample_count']:,} | "
                        f"context instances="
                        f"{training_supervision_audit['supervision_instance_count']:,}"
                    )
                    with open(os.path.join(model_dir, "model_metadata.json"), "w",
                              encoding="utf-8") as handle:
                        json.dump(metadata, handle, indent=2, ensure_ascii=False)
                    try:
                        metrics = train_model(
                            model=model,
                            train_loader=loaders["train"],
                            val_loader=loaders["val"],
                            test_loader=loaders["test"],
                            num_epochs=args.num_epochs,
                            lr=args.lr,
                            device_ids=args.device_ids,
                            patience=args.patience,
                            output_dir=model_dir,
                            weight_decay=args.weight_decay,
                            logger=logger,
                            selection_metric=args.selection_metric,
                            threshold_metric=args.threshold_metric,
                            use_tta=args.eval_tta,
                            test_bootstrap_replicates=bootstrap_replicates,
                            test_bootstrap_confidence=bootstrap_confidence,
                            test_bootstrap_seed=model_seed + 91,
                            train_evaluation_loader=loaders["train_eval"],
                            minimum_epochs=args.minimum_epochs,
                            threshold_score_tolerance=args.threshold_score_tolerance,
                            lr_warmup_epochs=args.lr_warmup_epochs,
                            lr_plateau_patience=args.lr_plateau_patience,
                            lr_plateau_factor=args.lr_plateau_factor,
                            min_lr=args.minimum_lr,
                            selection_min_delta=args.selection_min_delta,
                            domain_risk_variance_weight=(
                                args.domain_risk_variance_weight
                            ),
                            domain_risk_warmup_epochs=(
                                args.domain_risk_warmup_epochs
                            ),
                            ema_decay=args.ema_decay,
                            ema_start_epoch=args.ema_start_epoch,
                        )
                        metrics.update({
                            "training_unique_real_samples": int(
                                training_supervision_audit["unique_real_sample_count"]
                            ),
                            "training_supervision_instances": int(
                                training_supervision_audit["supervision_instance_count"]
                            ),
                            "training_context_view_count": int(
                                training_supervision_audit["context_view_count"]
                            ),
                            "training_optimizer_items": int(
                                training_supervision_audit[
                                    "optimizer_item_count_after_supervision_chunking"
                                ]
                            ),
                            "training_observed_max_supervised_points_per_item": int(
                                training_supervision_audit[
                                    "observed_max_supervised_points_per_optimizer_item"
                                ]
                            ),
                            "training_batch_weight_mass_coefficient_of_variation": float(
                                loaders["train"].batch_mass_audit[
                                    "batch_weight_mass_coefficient_of_variation"
                                ]
                            ),
                            "training_batch_weight_mass_max_to_mean_ratio": float(
                                loaders["train"].batch_mass_audit[
                                    "batch_weight_mass_max_to_mean_ratio"
                                ]
                            ),
                        })
                    finally:
                        for loader in loaders.values():
                            close = getattr(loader.dataset, "close", None)
                            if close is not None:
                                close()
                        for handler in list(logger.handlers):
                            handler.flush()
                            handler.close()
                            logger.removeHandler(handler)
                        del model, loaders
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                else:
                    set_global_seed(model_seed)
                    metrics = train_classical_model(
                        model_name,
                        classical_arrays,
                        classical_points,
                        train_weights,
                        output_dir=model_dir,
                        seed=model_seed,
                        threshold_metric=args.threshold_metric,
                        crop_size=args.crop_size,
                        bootstrap_replicates=bootstrap_replicates,
                        bootstrap_confidence=bootstrap_confidence,
                        bootstrap_seed=model_seed + 91,
                        n_jobs=args.classical_n_jobs,
                        iterations=args.classical_iterations,
                        selection_metric=args.selection_metric,
                        threshold_score_tolerance=args.threshold_score_tolerance,
                    )

                metrics.update({
                    "fold": fold_index,
                    "model": model_name,
                    "model_display_name": spec.display_name,
                    "model_family": spec.family,
                    "implementation": spec.implementation,
                    "paper_role": spec.paper_role,
                    "sampling_method": method,
                    "test_region": test_region,
                    "validation_region": validation_region,
                    "train_regions": "|".join(map(str, training_regions)),
                    "train_positive_samples": len(split_data["train"]["positive_points"]),
                    "validation_positive_samples": len(split_data["val"]["positive_points"]),
                    "test_positive_samples": len(split_data["test"]["positive_points"]),
                    "prediction_file": os.path.relpath(
                        os.path.join(model_dir, "test_predictions.npz"), output_dir
                    ),
                })
                all_metrics.append(metrics)
                serializable_metrics = {
                    key: (
                        value.tolist()
                        if isinstance(value, np.ndarray)
                        else value.item() if isinstance(value, np.generic) else value
                    )
                    for key, value in metrics.items()
                }
                with open(metrics_path, "w",
                          encoding="utf-8") as handle:
                    json.dump(serializable_metrics, handle, indent=2, ensure_ascii=False)
                write_comparison_tables(all_metrics, output_dir)
                overall_progress.update(1)
                overall_progress.set_postfix_str(
                    f"F{fold_index} {method}/{model_name}", refresh=False
                )
                console(
                    f"Model completed | Fold={fold_index} | sampling={method} | "
                    f"model={spec.display_name} | "
                    f"{metric_line(metrics, ('auc', 'pr_auc', 'f1', 'kappa', 'precision', 'recall'))}"
                )

    overall_progress.close()
    report_progress = track(total=2, desc="Generating manuscript results", unit="stage", leave=True)
    write_comparison_tables(all_metrics, output_dir)
    report_progress.update(1)
    report_progress.set_postfix_str("SRC/PRC", refresh=False)
    write_rate_curve_reports(all_metrics, output_dir)
    report_progress.update(1)
    report_progress.close()
    console(f"Step 2/3 complete; contiguous regional holdout results directory: {output_dir}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Nested continuous macro-region hold-out with DWSS/random controls."
    )
    parser.add_argument("xml", help="Path to Landslide_susceptibility_mapping.xml")
    parser.add_argument(
        "--suite",
        choices=tuple(PAPER_MODEL_SUITES),
        default=None,
        help=(
            "Model suite: reviewer (LandslideNet DWSS/random), machine_learning, "
            "deep_learning, ablation, or paper (all manuscript models)."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=(
            "Model names or groups (machine_learning, deep_learning, ablation, "
            "proposed, all). Overrides --suite and XML <model>."
        ),
    )
    parser.add_argument(
        "--inventory",
        default=None,
        help=(
            "Landslide point Shapefile/GeoPackage or aligned inventory raster. "
            "Point geometries are mapped to unique cells of the frozen regional grid. "
            "For rasters only --positive-value is treated as landslide."
        ),
    )
    parser.add_argument(
        "--regions",
        default=None,
        help=(
            "Aligned integer GeoTIFF of frozen continuous macro-regions; 0/NoData is "
            "excluded. Defaults to the configured XML macro_region_output."
        ),
    )
    parser.add_argument("--factors-dir", default=None, help="Full-resolution factor GeoTIFF directory.")
    parser.add_argument(
        "--frequency-ratio-factors",
        default=None,
        help=(
            "Categorical rasters as Factor:NormalizedLevelCount, for example "
            "Geology:176,Soil:36,Earthquake:5. Defaults to XML."
        ),
    )
    parser.add_argument("--frequency-ratio-smoothing", type=float, default=None)
    parser.add_argument("--frequency-ratio-log2-clip", type=float, default=None)
    parser.add_argument("--region-ids", type=int, nargs="+", default=None)
    parser.add_argument(
        "--test-region-ids",
        type=int,
        nargs="+",
        default=None,
        help="Run only selected outer folds while retaining the complete region universe.",
    )
    parser.add_argument("--region-names", default=None, help="Optional CSV: region_id,region_name")
    parser.add_argument(
        "--validation-map",
        default=None,
        help=(
            "Optional preregistered CSV: test_region,validation_region. Overrides the XML/CLI "
            "validation strategy."
        ),
    )
    parser.add_argument(
        "--validation-strategy",
        choices=("cyclic", "support_aware"),
        default=None,
        help=(
            "Inner validation-region selection. support_aware chooses the smallest whole "
            "region meeting preferred validation support while preserving minimum training "
            "support; if impossible it uses the strongest feasible region."
        ),
    )
    parser.add_argument("--min-inner-training-positives", type=int, default=None)
    parser.add_argument("--preferred-validation-positives", type=int, default=None)
    parser.add_argument(
        "--sampling-methods",
        nargs="+",
        choices=("dwss", "random"),
        default=None,
    )
    parser.add_argument("--positive-value", type=int, default=None)
    parser.add_argument("--min-region-positives", type=int, default=None)
    parser.add_argument("--low-support-warning", type=int, default=None)
    parser.add_argument("--adequate-support", type=int, default=None)
    parser.add_argument("--bootstrap-replicates", type=int, default=None)
    parser.add_argument("--bootstrap-confidence", type=float, default=None)
    parser.add_argument(
        "--region-balance-training",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Temper dominant inner-training region x class groups while preserving "
            "the configured positive/negative class prior; strength is set by balance power."
        ),
    )
    parser.add_argument("--region-balance-power", type=float, default=None)
    parser.add_argument("--negative-ratio", type=float, default=None)
    parser.add_argument("--candidate-multiplier", type=float, default=None)
    parser.add_argument("--candidate-minimum", type=int, default=None)
    parser.add_argument("--candidate-maximum", type=int, default=None)
    parser.add_argument("--training-candidate-minimum", type=int, default=None)
    parser.add_argument("--training-candidate-maximum", type=int, default=None)
    parser.add_argument("--adaptive-candidate-maximum", type=int, default=None)
    parser.add_argument("--adaptive-batch-size", type=int, default=None)
    parser.add_argument("--screening-neighbors", type=int, default=None)
    parser.add_argument("--theta-min", type=float, default=None)
    parser.add_argument("--n-strata", type=int, default=None)
    parser.add_argument("--weight-power", type=float, default=None)
    parser.add_argument("--min-per-stratum", type=int, default=None)
    parser.add_argument(
        "--max-prototypes",
        type=int,
        default=None,
        help=(
            "Maximum joint-KDE prototypes; the manuscript-strict default 0 "
            "uses every inner-training positive. Any positive cap is audited."
        ),
    )
    parser.add_argument("--kde-chunk-size", type=int, default=None)
    parser.add_argument("--raster-chunk-size", type=int, default=None)
    parser.add_argument("--feature-tile-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--selection-metric", choices=("auc", "f1", "kappa", "iou"), default=None)
    parser.add_argument("--threshold-metric", choices=("f1", "kappa", "iou"), default=None)
    parser.add_argument(
        "--threshold-score-tolerance",
        type=float,
        default=None,
        help=(
            "Maximum validation metric sacrificed when choosing the threshold "
            "with the smallest precision-recall gap."
        ),
    )
    parser.add_argument("--selection-min-delta", type=float, default=None)
    parser.add_argument("--eval-tta", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--augmentation-mode",
        choices=("none", "aspect_safe_d4"),
        default=None,
    )
    parser.add_argument("--aspect-period", type=float, default=None)
    parser.add_argument("--crop-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--num-bands", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--minimum-epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lr-warmup-epochs", type=int, default=None)
    parser.add_argument("--lr-plateau-patience", type=int, default=None)
    parser.add_argument("--lr-plateau-factor", type=float, default=None)
    parser.add_argument("--minimum-lr", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--domain-risk-variance-weight", type=float, default=None)
    parser.add_argument("--domain-risk-warmup-epochs", type=int, default=None)
    parser.add_argument(
        "--training-context-views",
        type=int,
        choices=(1, 2, 4),
        default=None,
        help=(
            "Training-only shifted crop-grid views per real sample; no coordinates "
            "or labels are synthesized."
        ),
    )
    parser.add_argument(
        "--max-supervised-points-per-training-tile",
        type=int,
        default=None,
        help=(
            "Split exceptionally dense training contexts into bounded supervision "
            "items; 0 disables the bound."
        ),
    )
    parser.add_argument("--ema-decay", type=float, default=None)
    parser.add_argument("--ema-start-epoch", type=int, default=None)
    parser.add_argument("--device-ids", default=None)
    parser.add_argument("--classical-iterations", type=int, default=None)
    parser.add_argument("--classical-n-jobs", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show live progress bars (enabled by default; disable with --no-progress).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Verify and skip completed model/fold artifacts in an identical output run.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip the macro-region layout PNG.")
    parser.add_argument("--audit-only", action="store_true", help="Validate region coverage and stop.")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Fit fold-specific preprocessing and write sample/leakage manifests without model training.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
