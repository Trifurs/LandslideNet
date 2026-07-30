"""Validation and full-domain susceptibility mapping.

OOF maps predict each macro-region only with the fold that held that complete
region out, so they remain validation products. Full maps are explicitly
separate deployment-oriented cross-fold ensembles: dense deep models use MSMF,
whereas classical models use direct per-pixel inference without MSMF.

The command-line entry point accepts either:

1. a completed experiment directory created by ``2_model_train.py``; or
2. the project XML file. In the latter case, the newest complete experiment
   below ``train_output`` is selected automatically.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import xml.etree.ElementTree as ET
from contextlib import ExitStack
from pathlib import Path
from typing import Mapping, Sequence

import joblib
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from rasterio.windows import Window

from .data import FrozenFoldFeatureTransformer, validate_aligned_rasters
from .model_registry import (
    MODEL_SPECS,
    build_deep_model,
    expand_model_selection,
)
from .progress import configure_progress, console, track, window_count
from .training import predict_probabilities, resolve_device


FOLD_PATTERN = re.compile(r"Fold_(\d+)_TestRegion_(\d+)$")
NODATA = -9999.0
_XML_SUFFIXES = {".xml"}
_REQUIRED_EXPERIMENT_FILES = ("validation_protocol.json", "model_registry.json")


def _load_json(path: str | Path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON file does not exist: {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON file: {path}: {error}") from error


def _normalize_path(value: str | os.PathLike) -> str:
    text = os.path.expandvars(os.path.expanduser(str(value).strip()))
    if os.name != "nt":
        text = text.replace("\\", "/")
    return text


def _read_xml_params(xml_path: str | Path) -> dict[str, str]:
    xml_path = Path(xml_path).expanduser().resolve()
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as error:
        raise ValueError(f"Invalid XML configuration: {xml_path}: {error}") from error

    params: dict[str, str] = {}
    for param in root.findall("param"):
        name_node = param.find("name")
        value_node = param.find("value")
        if (
            name_node is not None
            and value_node is not None
            and name_node.text
            and value_node.text is not None
        ):
            params[name_node.text.strip()] = _normalize_path(value_node.text)
    return params


def _parse_int_tokens(value, default: Sequence[int] = (0,)) -> list[int]:
    if value is None:
        return list(default)
    if isinstance(value, (list, tuple)):
        tokens = value
    else:
        tokens = re.split(r"[\s,]+", str(value).strip().strip("[]"))
    result = [int(str(token).strip()) for token in tokens if str(token).strip()]
    return result or list(default)


def _parse_bool(value, default: bool = False) -> bool:
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


def _valid(data, nodata):
    mask = np.isfinite(data)
    if nodata is not None and np.isfinite(nodata):
        mask &= ~np.isclose(data, nodata)
    return mask


def _windows(height, width, size):
    for row in range(0, height, size):
        for col in range(0, width, size):
            yield Window(col, row, min(size, width - col), min(size, height - row))


def _initialize(destination, value, tile_size, description="初始化输出栅格"):
    windows = _windows(destination.height, destination.width, tile_size)
    for window in track(
        windows,
        total=window_count(destination.height, destination.width, tile_size),
        desc=description,
        unit="tile",
    ):
        destination.write(
            np.full(
                (int(window.height), int(window.width)),
                value,
                dtype=destination.dtypes[0],
            ),
            1,
            window=window,
        )


def discover_folds(experiment_dir: Path) -> dict[int, Path]:
    if not experiment_dir.is_dir():
        raise NotADirectoryError(f"Experiment path is not a directory: {experiment_dir}")
    result = {}
    for path in experiment_dir.iterdir():
        match = FOLD_PATTERN.fullmatch(path.name) if path.is_dir() else None
        if match:
            test_region = int(match.group(2))
            if test_region in result:
                raise RuntimeError(f"Duplicate fold for test region {test_region}.")
            result[test_region] = path
    return result


def _experiment_artifact_audit(experiment_dir: Path) -> dict:
    """Inspect whether a training output is structurally usable for prediction."""
    experiment_dir = Path(experiment_dir)
    missing: list[str] = []

    if not experiment_dir.is_dir():
        return {
            "path": str(experiment_dir),
            "structure_valid": False,
            "complete": False,
            "missing": ["not_a_directory"],
            "mtime": 0.0,
        }

    for filename in _REQUIRED_EXPERIMENT_FILES:
        if not (experiment_dir / filename).is_file():
            missing.append(filename)
    if missing:
        return {
            "path": str(experiment_dir),
            "structure_valid": False,
            "complete": False,
            "missing": missing,
            "mtime": experiment_dir.stat().st_mtime,
        }

    try:
        protocol = _load_json(experiment_dir / "validation_protocol.json")
        registry = _load_json(experiment_dir / "model_registry.json")
    except (OSError, ValueError) as error:
        return {
            "path": str(experiment_dir),
            "structure_valid": False,
            "complete": False,
            "missing": [f"invalid_metadata: {error}"],
            "mtime": experiment_dir.stat().st_mtime,
        }

    model_rows = registry.get("requested_models", [])
    model_names = [row.get("name") for row in model_rows if row.get("name")]
    sampling_methods = list(protocol.get("sampling_methods", []))
    executed_regions = list(
        map(
            int,
            protocol.get(
                "outer_test_regions_executed",
                protocol.get("region_ids", []),
            ),
        )
    )
    folds = discover_folds(experiment_dir)

    if not model_names:
        missing.append("model_registry.requested_models")
    if not sampling_methods:
        missing.append("validation_protocol.sampling_methods")
    if not executed_regions:
        missing.append("validation_protocol.outer_test_regions_executed/region_ids")

    for test_region in executed_regions:
        fold_dir = folds.get(test_region)
        if fold_dir is None:
            missing.append(f"Fold_*_TestRegion_{test_region}")
            continue
        fold_audit = fold_dir / "fold_leakage_audit.json"
        if not fold_audit.is_file():
            missing.append(str(fold_audit.relative_to(experiment_dir)))
        for method in sampling_methods:
            for model_name in model_names:
                if model_name not in MODEL_SPECS:
                    missing.append(f"unknown_model_in_registry:{model_name}")
                    continue
                model_dir = fold_dir / method.upper() / f"Model_{model_name}"
                model_artifact = model_dir / (
                    "best_model_weight.pth"
                    if MODEL_SPECS[model_name].family == "deep"
                    else "best_model.joblib"
                )
                for required in (model_artifact, model_dir / "test_metrics.json"):
                    if not required.is_file():
                        missing.append(str(required.relative_to(experiment_dir)))

    metadata_mtime = max(
        (experiment_dir / filename).stat().st_mtime
        for filename in _REQUIRED_EXPERIMENT_FILES
    )
    return {
        "path": str(experiment_dir),
        "structure_valid": True,
        "complete": not missing,
        "missing": missing,
        "mtime": metadata_mtime,
        "executed_regions": executed_regions,
        "models": model_names,
        "sampling_methods": sampling_methods,
    }


def _candidate_experiment_dirs(train_output: Path) -> list[Path]:
    train_output = train_output.expanduser().resolve()
    if not train_output.exists():
        raise FileNotFoundError(
            "The XML train_output path does not exist: "
            f"{train_output}. Run 2_model_train.py first or pass an experiment directory."
        )

    candidates: set[Path] = set()
    if all((train_output / name).is_file() for name in _REQUIRED_EXPERIMENT_FILES):
        candidates.add(train_output)

    if train_output.is_dir():
        for protocol_path in train_output.rglob("validation_protocol.json"):
            parent = protocol_path.parent
            if (parent / "model_registry.json").is_file():
                candidates.add(parent.resolve())
    return sorted(candidates)


def _format_candidate_audits(audits: Sequence[Mapping], limit: int = 5) -> str:
    lines = []
    for audit in sorted(audits, key=lambda row: row.get("mtime", 0), reverse=True)[:limit]:
        missing = audit.get("missing", [])
        preview = ", ".join(missing[:4])
        if len(missing) > 4:
            preview += f", ... (+{len(missing) - 4})"
        lines.append(
            f"- {audit['path']} | complete={audit.get('complete', False)}"
            + (f" | missing: {preview}" if preview else "")
        )
    return "\n".join(lines)


def resolve_experiment_dir(
    source: str | Path,
    *,
    explicit_experiment_dir: str | Path | None = None,
    allow_partial: bool = False,
) -> tuple[Path, dict[str, str]]:
    """Resolve a completed experiment from a directory or project XML."""
    source_path = Path(_normalize_path(source)).expanduser().resolve()
    xml_params: dict[str, str] = {}
    if source_path.is_file() and source_path.suffix.lower() in _XML_SUFFIXES:
        xml_params = _read_xml_params(source_path)

    if explicit_experiment_dir is not None:
        experiment_dir = Path(_normalize_path(explicit_experiment_dir)).expanduser().resolve()
    elif source_path.is_dir():
        experiment_dir = source_path
    elif source_path.is_file() and source_path.suffix.lower() in _XML_SUFFIXES:
        configured_experiment = xml_params.get("prediction_experiment_dir")
        if configured_experiment:
            experiment_dir = Path(configured_experiment).expanduser().resolve()
        else:
            train_output = xml_params.get("train_output")
            if not train_output:
                raise ValueError(
                    f"XML {source_path} does not contain <train_output>. "
                    "Pass the completed experiment directory directly or add train_output."
                )
            candidate_dirs = _candidate_experiment_dirs(Path(train_output))
            if not candidate_dirs:
                raise FileNotFoundError(
                    "No experiment containing validation_protocol.json and "
                    f"model_registry.json was found below: {Path(train_output).expanduser()}"
                )
            audits = [_experiment_artifact_audit(path) for path in candidate_dirs]
            complete = [audit for audit in audits if audit["complete"]]
            if complete:
                selected = max(complete, key=lambda row: row["mtime"])
            elif allow_partial:
                structurally_valid = [audit for audit in audits if audit["structure_valid"]]
                if not structurally_valid:
                    raise RuntimeError(
                        "No structurally valid experiment was found. Candidates:\n"
                        + _format_candidate_audits(audits)
                    )
                selected = max(structurally_valid, key=lambda row: row["mtime"])
            else:
                raise RuntimeError(
                    "Experiments were found below train_output, but none is complete enough "
                    "for prediction. Resume/finish training, pass --allow-partial for a "
                    "diagnostic map, or provide --experiment-dir explicitly.\n"
                    + _format_candidate_audits(audits)
                )
            experiment_dir = Path(selected["path"])
            console(f"XML 自动选择训练实验：{experiment_dir}")
    elif source_path.is_file():
        raise ValueError(
            f"Unsupported source file: {source_path}. Expected an XML configuration "
            "or a completed experiment directory."
        )
    else:
        raise FileNotFoundError(f"Prediction source does not exist: {source_path}")

    if not experiment_dir.is_dir():
        raise NotADirectoryError(
            f"Resolved experiment path is not a directory: {experiment_dir}. "
            "The prediction script needs the completed output directory from "
            "2_model_train.py, not an arbitrary file."
        )
    missing_top = [
        filename
        for filename in _REQUIRED_EXPERIMENT_FILES
        if not (experiment_dir / filename).is_file()
    ]
    if missing_top:
        raise FileNotFoundError(
            f"Not a valid experiment directory: {experiment_dir}. Missing: {missing_top}. "
            "Use the timestamped Continuous_Regional_Holdout_* directory created by "
            "2_model_train.py."
        )
    return experiment_dir, xml_params


def _load_deep_model(name, num_bands, weights_path, device):
    if not Path(weights_path).is_file():
        raise FileNotFoundError(f"Missing model weights: {weights_path}")
    model = build_deep_model(name, num_bands)
    try:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(weights_path, map_location="cpu")
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def _read_tile(sources, window, transformer):
    arrays = []
    valid = None
    for source in sources:
        data = source.read(
            1,
            window=window,
            boundless=True,
            fill_value=source.nodata if source.nodata is not None else np.nan,
            out_dtype="float32",
        )
        current_valid = _valid(data, source.nodata)
        safe = data.copy()
        safe[~current_valid] = 0.0
        arrays.append(safe)
        valid = current_valid if valid is None else valid & current_valid
    raw = np.stack(arrays, axis=0)
    transformed = transformer.transform(raw.reshape(len(arrays), -1).T).T.reshape(raw.shape)
    return transformed.astype(np.float32, copy=False), valid


def _predict_fold(
    model_name,
    model_dir,
    test_region,
    region_source,
    factor_sources,
    transformer,
    destination,
    crop_size,
    device,
    use_tta,
    binary_destination=None,
    threshold=None,
):
    spec = MODEL_SPECS[model_name]
    if spec.family == "deep":
        predictor = _load_deep_model(
            model_name,
            len(factor_sources),
            model_dir / "best_model_weight.pth",
            device,
        )
    else:
        model_path = model_dir / "best_model.joblib"
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing classical model: {model_path}")
        predictor = joblib.load(model_path)

    predicted_cells = 0
    with torch.inference_mode():
        windows = _windows(region_source.height, region_source.width, crop_size)
        tile_progress = track(
            windows,
            total=window_count(region_source.height, region_source.width, crop_size),
            desc=f"区域 {test_region} OOF 推理",
            unit="tile",
        )
        for window in tile_progress:
            region = region_source.read(1, window=window)
            region_mask = _valid(region, region_source.nodata) & np.isclose(
                region, test_region
            )
            if not np.any(region_mask):
                continue
            factors, factor_valid = _read_tile(factor_sources, window, transformer)
            usable = region_mask & factor_valid
            if not np.any(usable):
                continue

            if spec.family == "deep":
                factors[:, ~usable] = 0.0
                tensor = torch.from_numpy(factors[None]).to(device)
                original_height, original_width = tensor.shape[-2:]
                pad_height = (16 - original_height % 16) % 16
                pad_width = (16 - original_width % 16) % 16
                if pad_height or pad_width:
                    tensor = F.pad(tensor, (0, pad_width, 0, pad_height))
                probability, _logits = predict_probabilities(
                    predictor, tensor, use_tta=use_tta
                )
                probability = (
                    probability[0, :original_height, :original_width]
                    .float()
                    .cpu()
                    .numpy()
                )
            else:
                probability = np.full(region.shape, NODATA, dtype=np.float32)
                features = factors[:, usable].T
                probability[usable] = predictor.predict_proba(features)[:, 1]

            current = destination.read(1, window=window)
            current[usable] = probability[usable]
            destination.write(current.astype(np.float32), 1, window=window)
            if binary_destination is not None:
                if threshold is None:
                    raise ValueError("A threshold is required when writing binary maps.")
                binary = binary_destination.read(1, window=window)
                binary[usable] = (probability[usable] >= threshold).astype(np.uint8)
                binary_destination.write(binary, 1, window=window)
            predicted_cells += int(np.count_nonzero(usable))
            tile_progress.set_postfix(predicted=f"{predicted_cells:,}", refresh=False)

    del predictor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return predicted_cells


def _resolve_models(args, registry):
    available = [row["name"] for row in registry["requested_models"]]
    if args.models is None or args.models == ["all"]:
        return available
    result = expand_model_selection(args.models)
    missing = sorted(set(result) - set(available))
    if missing:
        raise ValueError(f"Models were not trained in this experiment: {missing}")
    return result


def _validate_selected_artifacts(
    experiment_dir: Path,
    folds: Mapping[int, Path],
    models: Sequence[str],
    sampling_methods: Sequence[str],
) -> None:
    missing = []
    for test_region, fold_dir in sorted(folds.items()):
        if not (fold_dir / "fold_leakage_audit.json").is_file():
            missing.append(f"{fold_dir.name}/fold_leakage_audit.json")
        for method in sampling_methods:
            for model_name in models:
                model_dir = fold_dir / method.upper() / f"Model_{model_name}"
                artifact = model_dir / (
                    "best_model_weight.pth"
                    if MODEL_SPECS[model_name].family == "deep"
                    else "best_model.joblib"
                )
                for required in (artifact, model_dir / "test_metrics.json"):
                    if not required.is_file():
                        missing.append(str(required.relative_to(experiment_dir)))
    if missing:
        preview = "\n".join(f"- {item}" for item in missing[:20])
        remainder = "" if len(missing) <= 20 else f"\n... and {len(missing) - 20} more"
        raise FileNotFoundError(
            "Prediction artifacts are incomplete for the selected folds/models/sampling "
            f"arms:\n{preview}{remainder}"
        )


def _prepare_output_path(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {path}. Use --overwrite to replace existing maps."
        )
    path.parent.mkdir(parents=True, exist_ok=True)


def _full_map_profile(reference, dtype="float32", nodata=NODATA):
    profile = reference.profile.copy()
    profile.update(
        driver="GTiff",
        dtype=dtype,
        count=1,
        nodata=nodata,
        compress="deflate",
        predictor=3 if dtype == "float32" else 2,
        BIGTIFF="IF_SAFER",
    )
    return profile


def _write_json_atomic(path: Path, payload: dict) -> None:
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(partial, path)


def _write_classical_fold_map(
    *,
    model_name: str,
    model_dir: Path,
    transformer: FrozenFoldFeatureTransformer,
    regions_path: str,
    factor_paths: Sequence[str],
    destination_path: Path,
    tile_size: int,
    test_region: int,
) -> int:
    """Predict the full factor-valid domain once with one classical fold model."""
    estimator_path = model_dir / "best_model.joblib"
    if not estimator_path.is_file():
        raise FileNotFoundError(f"Missing classical model: {estimator_path}")
    estimator = joblib.load(estimator_path)
    partial = destination_path.with_name(f".{destination_path.name}.partial")
    partial.unlink(missing_ok=True)
    predicted = 0
    try:
        with ExitStack() as stack:
            region_source = stack.enter_context(rasterio.open(regions_path))
            factor_sources = [
                stack.enter_context(rasterio.open(path)) for path in factor_paths
            ]
            destination = stack.enter_context(
                rasterio.open(
                    partial,
                    "w",
                    **_full_map_profile(region_source),
                )
            )
            windows = _windows(region_source.height, region_source.width, tile_size)
            for window in track(
                windows,
                total=window_count(
                    region_source.height, region_source.width, tile_size
                ),
                desc=f"{model_name} Fold-R{test_region} 全域直接推理",
                unit="tile",
            ):
                region = region_source.read(1, window=window)
                domain = (
                    _valid(region, region_source.nodata)
                    & np.isfinite(region)
                    & (region > 0)
                )
                factors, factor_valid = _read_tile(
                    factor_sources, window, transformer
                )
                usable = domain & factor_valid
                probability = np.full(region.shape, NODATA, dtype=np.float32)
                if np.any(usable):
                    values = estimator.predict_proba(factors[:, usable].T)
                    if values.ndim != 2 or values.shape[1] != 2:
                        raise RuntimeError(
                            f"{model_name} did not return two probability columns."
                        )
                    probability[usable] = np.asarray(
                        values[:, 1], dtype=np.float32
                    )
                    predicted += int(np.count_nonzero(usable))
                destination.write(probability, 1, window=window)
        os.replace(partial, destination_path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    finally:
        del estimator
    return predicted


def _fuse_classical_fold_maps(
    *,
    fold_paths: Sequence[Path],
    probability_path: Path,
    binary_path: Path | None,
    threshold: float | None,
    tile_size: int,
) -> int:
    probability_partial = probability_path.with_name(
        f".{probability_path.name}.partial"
    )
    probability_partial.unlink(missing_ok=True)
    binary_partial = (
        binary_path.with_name(f".{binary_path.name}.partial")
        if binary_path is not None
        else None
    )
    if binary_partial is not None:
        binary_partial.unlink(missing_ok=True)

    try:
        with ExitStack() as stack:
            sources = [
                stack.enter_context(rasterio.open(path)) for path in fold_paths
            ]
            reference = sources[0]
            probability_destination = stack.enter_context(
                rasterio.open(
                    probability_partial,
                    "w",
                    **_full_map_profile(reference),
                )
            )
            binary_destination = (
                stack.enter_context(
                    rasterio.open(
                        binary_partial,
                        "w",
                        **_full_map_profile(reference, "uint8", 255),
                    )
                )
                if binary_partial is not None
                else None
            )
            if binary_destination is not None and threshold is None:
                raise ValueError("A threshold is required for a binary map.")

            valid_cells = 0
            windows = _windows(reference.height, reference.width, tile_size)
            for window in track(
                windows,
                total=window_count(reference.height, reference.width, tile_size),
                desc="机器学习跨折中位数融合（非 MSMF）",
                unit="tile",
                leave=True,
            ):
                arrays = []
                for source in sources:
                    values = source.read(1, window=window, out_dtype="float32")
                    arrays.append(
                        np.where(_valid(values, source.nodata), values, np.nan)
                    )
                stack_array = np.stack(arrays, axis=0)
                all_invalid = np.all(~np.isfinite(stack_array), axis=0)
                with np.errstate(all="ignore"):
                    probability = np.nanmedian(stack_array, axis=0)
                usable = ~all_invalid & np.isfinite(probability)
                output = np.full(probability.shape, NODATA, dtype=np.float32)
                output[usable] = np.clip(probability[usable], 0.0, 1.0)
                probability_destination.write(output, 1, window=window)
                valid_cells += int(np.count_nonzero(usable))
                if binary_destination is not None:
                    binary = np.full(output.shape, 255, dtype=np.uint8)
                    binary[usable] = (
                        output[usable] >= float(threshold)
                    ).astype(np.uint8)
                    binary_destination.write(binary, 1, window=window)
        os.replace(probability_partial, probability_path)
        if binary_partial is not None:
            os.replace(binary_partial, binary_path)
    except Exception:
        probability_partial.unlink(missing_ok=True)
        if binary_partial is not None:
            binary_partial.unlink(missing_ok=True)
        raise
    return valid_cells


def generate_classical_full_map(
    *,
    experiment_dir: Path,
    protocol: dict,
    folds: Mapping[int, Path],
    model_name: str,
    sampling_method: str,
    output_dir: Path,
    tile_size: int,
    write_binary: bool,
    threshold: float | None,
    overwrite: bool,
    resume: bool,
    keep_intermediate: bool,
) -> str:
    """Generate a direct per-pixel classical map; MSMF is never invoked."""
    if MODEL_SPECS[model_name].family != "classical":
        raise ValueError(f"{model_name} is not a classical model.")
    factors = [str(Path(path).expanduser()) for path in protocol["factor_paths"]]
    factor_names = [Path(path).stem for path in factors]
    regions_path = str(Path(protocol["macro_regions"]).expanduser())
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"full_{model_name}_{sampling_method}"
    probability_path = output_dir / f"{stem}_probability.tif"
    binary_path = output_dir / f"{stem}_class.tif" if write_binary else None
    metadata_path = output_dir / f"{stem}_metadata.json"
    completed_outputs = [probability_path, metadata_path]
    if binary_path is not None:
        completed_outputs.append(binary_path)
    if resume and all(path.is_file() for path in completed_outputs):
        console(f"恢复运行：已完成机器学习全域图，跳过 {probability_path}")
        return str(probability_path)
    for path in (probability_path, binary_path, metadata_path):
        if path is not None and path.exists() and not overwrite:
            raise FileExistsError(
                f"Output already exists: {path}. Use --overwrite to replace it."
            )

    expected_regions = sorted(map(int, protocol["region_ids"]))
    if set(folds) != set(expected_regions):
        raise RuntimeError(
            "A full-domain fold ensemble requires every regional fold. "
            f"Expected {expected_regions}, found {sorted(folds)}."
        )

    work_dir = output_dir / f".{stem}_work"
    contract = {
        "source_experiment": str(experiment_dir),
        "model": model_name,
        "sampling_method": sampling_method,
        "fold_regions": expected_regions,
        "tile_size": int(tile_size),
        "inference": "direct_per_pixel_without_msmf",
        "fold_fusion": "pixelwise_median",
    }
    contract_path = work_dir / "run_contract.json"
    if work_dir.exists():
        if overwrite:
            shutil.rmtree(work_dir)
        elif resume:
            if not contract_path.is_file() or _load_json(contract_path) != contract:
                raise RuntimeError(
                    "Classical-map resume settings differ from the existing contract."
                )
        else:
            raise FileExistsError(
                f"Incomplete work directory exists: {work_dir}. "
                "Use --resume or --overwrite."
            )
    work_dir.mkdir(parents=True, exist_ok=True)
    if not contract_path.exists():
        _write_json_atomic(contract_path, contract)

    fold_paths = []
    thresholds = {}
    predicted_by_fold = {}
    for test_region in expected_regions:
        fold_dir = folds[test_region]
        fold_audit = _load_json(fold_dir / "fold_leakage_audit.json")
        transformer = FrozenFoldFeatureTransformer.from_dict(
            fold_audit["feature_transformer"]
        )
        if list(transformer.factor_names) != factor_names:
            raise RuntimeError(
                f"Fold R{test_region} factor order differs from the protocol."
            )
        model_dir = (
            fold_dir / sampling_method.upper() / f"Model_{model_name}"
        )
        metrics = _load_json(model_dir / "test_metrics.json")
        thresholds[test_region] = float(
            metrics["threshold_selected_on_validation"]
        )
        fold_path = work_dir / f"{stem}_fold_R{test_region}.tif"
        fold_paths.append(fold_path)
        if resume and fold_path.is_file():
            console(f"恢复运行：跳过已完成机器学习 Fold-R{test_region} 全图")
            continue
        predicted_by_fold[test_region] = _write_classical_fold_map(
            model_name=model_name,
            model_dir=model_dir,
            transformer=transformer,
            regions_path=regions_path,
            factor_paths=factors,
            destination_path=fold_path,
            tile_size=tile_size,
            test_region=test_region,
        )

    selected_threshold = (
        float(threshold)
        if threshold is not None
        else float(np.median(list(thresholds.values())))
    )
    valid_cells = _fuse_classical_fold_maps(
        fold_paths=fold_paths,
        probability_path=probability_path,
        binary_path=binary_path,
        threshold=selected_threshold if write_binary else None,
        tile_size=tile_size,
    )
    _write_json_atomic(
        metadata_path,
        {
            "map_type": "deployment_cv_ensemble_direct_classical",
            "model": MODEL_SPECS[model_name].to_dict(),
            "sampling_method": sampling_method,
            "source_experiment": str(experiment_dir),
            "fold_models_used": expected_regions,
            "fold_validation_thresholds": thresholds,
            "deployment_binary_threshold": (
                selected_threshold if write_binary else None
            ),
            "threshold_policy": (
                "user_supplied"
                if threshold is not None
                else "median_of_fold_validation_selected_thresholds"
            ),
            "probability_raster": str(probability_path),
            "binary_raster": str(binary_path) if binary_path else None,
            "valid_factor_domain_cells": int(valid_cells),
            "predicted_cells_by_fold": predicted_by_fold,
            "inference_algorithm": "direct_per_pixel",
            "msmf_used": False,
            "fold_fusion": "pixelwise_median",
            "scientific_interpretation": (
                "Deployment-oriented cross-fold ensemble map; it is not an OOF "
                "validation product and must not be used for held-out accuracy."
            ),
        },
    )
    if not keep_intermediate:
        shutil.rmtree(work_dir, ignore_errors=True)
    console(f"机器学习全域图完成（未使用 MSMF）：{probability_path}")
    return str(probability_path)


def run_oof(args):
    if hasattr(args, "progress"):
        configure_progress(bool(args.progress))

    source = getattr(args, "source", None) or getattr(
        args, "experiment_dir", None
    )
    experiment_dir, xml_params = resolve_experiment_dir(
        source,
        explicit_experiment_dir=args.experiment_dir,
        allow_partial=bool(args.allow_partial),
    )
    protocol = _load_json(experiment_dir / "validation_protocol.json")
    registry = _load_json(experiment_dir / "model_registry.json")

    if args.device_ids is None:
        args.device_ids = _parse_int_tokens(xml_params.get("device_ids"), default=(0,))
    if args.output_dir is None and xml_params.get("prediction_output"):
        args.output_dir = xml_params["prediction_output"]
    if args.write_binary is None:
        args.write_binary = _parse_bool(xml_params.get("prediction_write_binary"), False)

    models = _resolve_models(args, registry)
    sampling_methods = args.sampling_methods or protocol["sampling_methods"]
    unknown_methods = sorted(set(sampling_methods) - set(protocol["sampling_methods"]))
    if unknown_methods:
        raise ValueError(f"Sampling arms were not trained: {unknown_methods}")

    folds = discover_folds(experiment_dir)
    if not folds:
        raise FileNotFoundError(
            f"No Fold_*_TestRegion_* directories were found in {experiment_dir}."
        )
    expected_regions = set(map(int, protocol["region_ids"]))
    found_regions = set(folds)
    if not args.allow_partial and found_regions != expected_regions:
        raise RuntimeError(
            f"A complete OOF map needs test folds {sorted(expected_regions)}, but found "
            f"{sorted(found_regions)}. Use --allow-partial only for a clearly labelled "
            "diagnostic map."
        )
    _validate_selected_artifacts(experiment_dir, folds, models, sampling_methods)

    factors = [str(Path(path).expanduser()) for path in protocol["factor_paths"]]
    regions_path = str(Path(protocol["macro_regions"]).expanduser())
    validate_aligned_rasters(regions_path, factors)
    crop_size = int(protocol["hyperparameters_fixed_before_outer_test"]["crop_size"])
    output_dir = Path(args.output_dir or experiment_dir / "oof_maps").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device, resolved_device_ids = resolve_device(args.device_ids)
    console(
        "步骤 3/3：生成区域外留 OOF 图 | "
        f"experiment={experiment_dir} | models={len(models)} | "
        f"sampling={sampling_methods} | folds={len(folds)} | "
        f"device={device} | GPU IDs={resolved_device_ids or 'none'} | output={output_dir}"
    )

    outputs = []
    overall_progress = track(
        total=len(sampling_methods) * len(models),
        desc="OOF 制图总进度",
        unit="map",
        leave=True,
    )
    with ExitStack() as stack:
        region_source = stack.enter_context(rasterio.open(regions_path))
        factor_sources = [stack.enter_context(rasterio.open(path)) for path in factors]
        profile = region_source.profile.copy()
        profile.update(
            driver="GTiff",
            dtype="float32",
            count=1,
            nodata=NODATA,
            compress="deflate",
            predictor=3,
            BIGTIFF="IF_SAFER",
        )
        binary_profile = profile.copy()
        binary_profile.update(dtype="uint8", nodata=255, predictor=2)

        for method in sampling_methods:
            for model_name in models:
                console(
                    f"开始制图：{MODEL_SPECS[model_name].display_name} | "
                    f"sampling={method.upper()}"
                )
                map_path = output_dir / f"oof_{model_name}_{method}_probability.tif"
                class_path = output_dir / f"oof_{model_name}_{method}_class.tif"
                metadata_path = map_path.with_suffix(".json")
                _prepare_output_path(map_path, args.overwrite)
                _prepare_output_path(metadata_path, args.overwrite)
                if args.write_binary:
                    _prepare_output_path(class_path, args.overwrite)

                temporary_map = map_path.with_name(f".{map_path.name}.partial")
                temporary_class = class_path.with_name(f".{class_path.name}.partial")
                temporary_map.unlink(missing_ok=True)
                temporary_class.unlink(missing_ok=True)

                try:
                    with rasterio.open(temporary_map, "w+", **profile) as destination:
                        _initialize(destination, NODATA, crop_size, "初始化概率图")
                        if args.write_binary:
                            binary_context = rasterio.open(
                                temporary_class, "w+", **binary_profile
                            )
                            _initialize(binary_context, 255, crop_size, "初始化分类图")
                        else:
                            binary_context = None
                        try:
                            region_counts = {}
                            thresholds = {}
                            for test_region, fold_dir in sorted(folds.items()):
                                model_dir = (
                                    fold_dir / method.upper() / f"Model_{model_name}"
                                )
                                fold_audit = _load_json(
                                    fold_dir / "fold_leakage_audit.json"
                                )
                                transformer = FrozenFoldFeatureTransformer.from_dict(
                                    fold_audit["feature_transformer"]
                                )
                                expected_factor_names = [
                                    Path(path).stem for path in factors
                                ]
                                if list(transformer.factor_names) != expected_factor_names:
                                    raise RuntimeError(
                                        f"Fold {test_region} transformer factor order differs "
                                        "from validation_protocol.json."
                                    )
                                metrics = _load_json(model_dir / "test_metrics.json")
                                threshold = float(
                                    metrics["threshold_selected_on_validation"]
                                )
                                thresholds[test_region] = threshold
                                region_counts[test_region] = _predict_fold(
                                    model_name,
                                    model_dir,
                                    test_region,
                                    region_source,
                                    factor_sources,
                                    transformer,
                                    destination,
                                    crop_size,
                                    device,
                                    bool(protocol.get("evaluation_tta", False)),
                                    binary_destination=binary_context,
                                    threshold=threshold,
                                )
                                console(
                                    f"区域 {test_region} 完成 | "
                                    f"threshold={threshold:.3f} | "
                                    f"predicted={region_counts[test_region]:,} cells"
                                )
                        finally:
                            if binary_context is not None:
                                binary_context.close()

                    os.replace(temporary_map, map_path)
                    if args.write_binary:
                        os.replace(temporary_class, class_path)
                except Exception:
                    temporary_map.unlink(missing_ok=True)
                    temporary_class.unlink(missing_ok=True)
                    raise

                metadata_path.write_text(
                    json.dumps(
                        {
                            "map_type": (
                                "cross_fitted_leave_one_macro_region_out_probability"
                            ),
                            "model": MODEL_SPECS[model_name].to_dict(),
                            "sampling_method": method,
                            "source_experiment": str(experiment_dir),
                            "test_regions_included": sorted(folds),
                            "complete_region_universe": sorted(expected_regions),
                            "complete_oof_coverage_requested": not args.allow_partial,
                            "predicted_factor_valid_cells_by_region": region_counts,
                            "validation_selected_thresholds_by_region": thresholds,
                            "outside_test_region_context_policy": (
                                "masked_to_zero_per_fold"
                            ),
                            "probability_raster": str(map_path),
                            "binary_raster": (
                                str(class_path) if args.write_binary else None
                            ),
                            "note": (
                                "This is an out-of-region validation map. It must not "
                                "be described as a model trained on all available labels."
                            ),
                        },
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                outputs.append(str(map_path))
                overall_progress.update(1)
                overall_progress.set_postfix_str(
                    f"{model_name}/{method}", refresh=False
                )
                console(f"已写出：{map_path}")
    overall_progress.close()
    console(f"步骤 3/3 完成，共生成 {len(outputs)} 幅概率图。")
    return outputs


def _configured_model_tokens(value) -> list[str] | None:
    if value is None:
        return None
    tokens = [
        token.strip().strip("'\"")
        for token in str(value).strip("[]").replace(",", " ").split()
        if token.strip().strip("'\"")
    ]
    return tokens or None


def run_full(args):
    """Generate full-domain maps, routing deep models to MSMF only."""
    configure_progress(bool(getattr(args, "progress", True)))
    experiment_dir, xml_params = resolve_experiment_dir(
        args.source,
        explicit_experiment_dir=getattr(args, "experiment_dir", None),
        allow_partial=False,
    )
    protocol = _load_json(experiment_dir / "validation_protocol.json")
    registry = _load_json(experiment_dir / "model_registry.json")
    if getattr(args, "models", None) is None:
        args.models = _configured_model_tokens(
            xml_params.get("prediction_models")
            or xml_params.get("models")
            or xml_params.get("model")
        )
    models = _resolve_models(args, registry)
    sampling_methods = (
        getattr(args, "sampling_methods", None)
        or protocol["sampling_methods"]
    )
    unknown_methods = sorted(
        set(sampling_methods) - set(protocol["sampling_methods"])
    )
    if unknown_methods:
        raise ValueError(f"Sampling arms were not trained: {unknown_methods}")
    if getattr(args, "allow_partial", False):
        raise ValueError("--allow-partial is valid only with --map-type oof.")

    folds = discover_folds(experiment_dir)
    expected_regions = set(map(int, protocol["region_ids"]))
    if set(folds) != expected_regions:
        raise RuntimeError(
            f"Full-domain maps require folds {sorted(expected_regions)}, "
            f"but found {sorted(folds)}."
        )
    _validate_selected_artifacts(
        experiment_dir, folds, models, sampling_methods
    )

    device_ids = getattr(args, "device_ids", None)
    if device_ids is None:
        device_ids = _parse_int_tokens(
            xml_params.get("device_ids"), default=(0,)
        )
    write_binary = getattr(args, "write_binary", None)
    if write_binary is None:
        write_binary = _parse_bool(
            xml_params.get("prediction_write_binary"), False
        )
    output_dir = Path(
        getattr(args, "output_dir", None)
        or xml_params.get("prediction_output")
        or experiment_dir / "full_maps"
    ).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    crop_size = getattr(args, "crop_size", None)
    if crop_size is None and xml_params.get("msmf_crop_size"):
        crop_size = int(xml_params["msmf_crop_size"])
    strides = getattr(args, "strides", None)
    if strides is None and xml_params.get("msmf_strides"):
        strides = _parse_int_tokens(xml_params["msmf_strides"], default=())
    batch_size = int(
        getattr(args, "batch_size", None)
        or xml_params.get("prediction_batch_size", 2)
    )
    minimum_weight = float(
        getattr(args, "minimum_weight", None)
        or xml_params.get("msmf_minimum_weight", 0.05)
    )
    tta = getattr(args, "tta", None)
    if tta is None:
        tta = _parse_bool(xml_params.get("prediction_tta"), False)
    amp = getattr(args, "amp", None)
    if amp is None:
        amp = _parse_bool(xml_params.get("prediction_amp"), False)
    write_disagreement = getattr(args, "write_disagreement", None)
    if write_disagreement is None:
        write_disagreement = _parse_bool(
            xml_params.get("msmf_write_disagreement"), True
        )
    fusion_tile_size = int(
        getattr(args, "fusion_tile_size", None)
        or xml_params.get("prediction_tile_size", 1024)
    )
    if crop_size is not None and int(crop_size) < 16:
        raise ValueError("MSMF crop size must be at least 16.")
    if batch_size < 1:
        raise ValueError("MSMF batch size must be positive.")
    if not 0 < minimum_weight <= 1:
        raise ValueError("MSMF minimum weight must be in (0, 1].")
    if fusion_tile_size < 1:
        raise ValueError("Prediction tile size must be positive.")

    console(
        "步骤 3/3：生成全域部署易发性图 | "
        f"models={len(models)} | sampling={sampling_methods} | "
        "deep=MSMF | classical=direct-per-pixel | "
        f"output={output_dir}"
    )
    outputs = []
    progress = track(
        total=len(models) * len(sampling_methods),
        desc="全域制图总进度",
        unit="map",
        leave=True,
    )
    for method in sampling_methods:
        for model_name in models:
            stem = f"full_{model_name}_{method}"
            probability_path = output_dir / f"{stem}_probability.tif"
            metadata_path = output_dir / f"{stem}_metadata.json"
            spec = MODEL_SPECS[model_name]
            completed_outputs = [probability_path, metadata_path]
            if write_binary:
                completed_outputs.append(output_dir / f"{stem}_class.tif")
            if spec.family == "deep" and write_disagreement:
                completed_outputs.append(
                    output_dir / f"{stem}_stride_mad.tif"
                )
            if (
                getattr(args, "resume", False)
                and all(path.is_file() for path in completed_outputs)
            ):
                console(f"恢复运行：已完成 {model_name}/{method}，跳过。")
                outputs.append(str(probability_path))
                progress.update(1)
                continue

            if spec.family == "deep":
                # Lazy import avoids a module cycle: MSMF reuses prediction
                # artifact-discovery and checkpoint-loading helpers.
                from .msmf import run_deep_msmf

                msmf_args = argparse.Namespace(
                    source=str(experiment_dir),
                    experiment_dir=str(experiment_dir),
                    model=model_name,
                    sampling_method=method,
                    fold_regions=None,
                    crop_size=crop_size,
                    strides=strides,
                    batch_size=batch_size,
                    device_ids=device_ids,
                    minimum_weight=minimum_weight,
                    tta=bool(tta),
                    use_protocol_tta=bool(
                        getattr(args, "use_protocol_tta", False)
                    ),
                    amp=bool(amp),
                    output_dir=str(output_dir),
                    output_name=stem,
                    write_binary=bool(write_binary),
                    threshold=getattr(args, "threshold", None),
                    write_disagreement=bool(write_disagreement),
                    fusion_tile_size=fusion_tile_size,
                    resume=bool(getattr(args, "resume", False)),
                    keep_intermediate=bool(
                        getattr(args, "keep_intermediate", False)
                    ),
                    overwrite=bool(getattr(args, "overwrite", False)),
                    progress=bool(getattr(args, "progress", True)),
                )
                outputs.append(run_deep_msmf(msmf_args))
            else:
                outputs.append(
                    generate_classical_full_map(
                        experiment_dir=experiment_dir,
                        protocol=protocol,
                        folds=folds,
                        model_name=model_name,
                        sampling_method=method,
                        output_dir=output_dir,
                        tile_size=fusion_tile_size,
                        write_binary=bool(write_binary),
                        threshold=getattr(args, "threshold", None),
                        overwrite=bool(getattr(args, "overwrite", False)),
                        resume=bool(getattr(args, "resume", False)),
                        keep_intermediate=bool(
                            getattr(args, "keep_intermediate", False)
                        ),
                    )
                )
            progress.update(1)
            progress.set_postfix_str(
                f"{model_name}/{method}", refresh=False
            )
    progress.close()
    console(f"步骤 3/3 完成，共生成 {len(outputs)} 幅全域概率图。")
    return outputs


def run(args):
    map_type = getattr(args, "map_type", None)
    if map_type is None:
        source = getattr(args, "source", None) or getattr(
            args, "experiment_dir", None
        )
        source_path = Path(_normalize_path(source)).expanduser()
        if source_path.suffix.lower() == ".xml" and source_path.is_file():
            map_type = _read_xml_params(source_path).get(
                "prediction_map_type", "full"
            )
        else:
            map_type = "full"
    map_type = str(map_type).strip().lower()
    if map_type == "oof":
        return run_oof(args)
    if map_type == "full":
        return run_full(args)
    raise ValueError("prediction map type must be 'full' or 'oof'.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Generate full-domain deployment maps or cross-fitted OOF validation maps. "
            "Deep full maps always use MSMF; classical full maps never use MSMF."
        )
    )
    parser.add_argument(
        "source",
        help=(
            "Completed output directory from 2_model_train.py, or the project XML. "
            "When XML is supplied, the newest complete experiment below train_output "
            "is selected automatically."
        ),
    )
    parser.add_argument(
        "--experiment-dir",
        default=None,
        help=(
            "Explicit completed experiment directory. Useful with an XML source when "
            "multiple runs exist; overrides automatic selection."
        ),
    )
    parser.add_argument(
        "--models", nargs="+", default=None, help="Default: all trained models."
    )
    parser.add_argument(
        "--map-type",
        choices=("full", "oof"),
        default=None,
        help=(
            "full (default, or XML prediction_map_type) builds deployment maps; "
            "oof builds held-out regional validation maps."
        ),
    )
    parser.add_argument(
        "--sampling-methods", nargs="+", choices=("dwss", "random")
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--device-ids",
        type=int,
        nargs="+",
        default=None,
        help="CUDA device IDs. With XML input, defaults to XML device_ids; otherwise [0].",
    )
    parser.add_argument(
        "--write-binary",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Also write thresholded class rasters.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="OOF only: allow a diagnostic map from incomplete outer folds.",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=None,
        help="Deep MSMF window size; defaults to the frozen training crop size.",
    )
    parser.add_argument(
        "--strides",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Deep MSMF strides; default for crop 512 is "
            "384 432 457 313 337 (or XML msmf_strides)."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Deep MSMF inference windows per batch (default/XML: 2).",
    )
    parser.add_argument(
        "--minimum-weight",
        type=float,
        default=None,
        help="MSMF within-stride edge weight (default: 0.05).",
    )
    parser.add_argument(
        "--tta",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use flip TTA for deep full-map inference.",
    )
    parser.add_argument(
        "--use-protocol-tta",
        action="store_true",
        help="Use evaluation_tta frozen in validation_protocol.json.",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use CUDA float16 autocast for deep MSMF inference.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Binary threshold; default is the median fold validation threshold.",
    )
    parser.add_argument(
        "--write-disagreement",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Deep only: write cross-stride median-absolute-deviation map.",
    )
    parser.add_argument(
        "--fusion-tile-size",
        type=int,
        default=None,
        help="Raster tile size for fusion and classical direct inference.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an identical interrupted full-map run.",
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep per-stride/per-fold intermediate full maps.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing probability/class/metadata outputs.",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="显示实时进度条（默认开启，可用 --no-progress 关闭）。",
    )
    args = parser.parse_args(argv)
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be positive.")
    if args.crop_size is not None and args.crop_size < 16:
        parser.error("--crop-size must be at least 16.")
    if args.minimum_weight is not None and not 0 < args.minimum_weight <= 1:
        parser.error("--minimum-weight must be in (0, 1].")
    if args.threshold is not None and not 0 <= args.threshold <= 1:
        parser.error("--threshold must be in [0, 1].")
    if args.fusion_tile_size is not None and args.fusion_tile_size < 1:
        parser.error("--fusion-tile-size must be positive.")
    return args
