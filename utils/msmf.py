"""MSMF implementation used internally by ``3_model_predict.py``.

All available regional fold models predict the complete study domain and their
probabilities are combined by a robust pixel-wise median. The ensemble is then
evaluated under an odd set of at least five non-multiple sliding-window
strides. Each stride-specific surface uses centre-weighted overlap blending,
and the final surface is the pixel-wise median of the stride maps.

The final operation is the multi-stride median fusion (MSMF) described in the
manuscript.  Weighted blending inside each stride is an implementation
improvement: it avoids arbitrary last-window overwriting before the cross-stride
median is calculated.

This map is a deployment/CV-ensemble susceptibility surface and is not used to
report held-out test accuracy.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
from contextlib import ExitStack, nullcontext
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from rasterio.windows import Window

from .data import FrozenFoldFeatureTransformer, validate_aligned_rasters
from .model_registry import MODEL_SPECS, canonical_model_name
from .prediction import (
    NODATA,
    _load_deep_model,
    _load_json,
    _parse_int_tokens,
    _read_xml_params,
    _valid,
    discover_folds,
    resolve_experiment_dir,
)
from .progress import configure_progress, console, track, window_count
from .training import predict_probabilities


# The upstream implementation used overlaps 128, 80, 55, 199 and 175 for a
# 512-pixel window. MSMF is defined by the corresponding strides below.
DEFAULT_STRIDES_512 = (384, 432, 457, 313, 337)
BINARY_NODATA = 255


def _parse_bool(value, default=False):
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


def _parse_int_list(value) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        tokens = value
    else:
        tokens = re.split(r"[\s,]+", str(value).strip().strip("[]"))
    return [int(str(token).strip()) for token in tokens if str(token).strip()]


def _default_strides(crop_size: int) -> list[int]:
    """Scale the original five 512-pixel MSMF strides when crop size changes."""
    if crop_size == 512:
        return list(DEFAULT_STRIDES_512)
    scaled = []
    for original in DEFAULT_STRIDES_512:
        value = int(round(original * crop_size / 512.0))
        value = min(max(value, 1), crop_size - 1)
        if value not in scaled:
            scaled.append(value)
    if len(scaled) < 5:
        candidates = np.linspace(crop_size * 0.70, crop_size * 0.96, 9)
        for candidate in candidates[::-1]:
            value = min(max(int(round(candidate)), 1), crop_size - 1)
            if value not in scaled:
                scaled.append(value)
            if len(scaled) == 5:
                break
    return scaled[:5]


def _validate_strides(strides: Sequence[int], crop_size: int) -> list[int]:
    strides = [int(value) for value in strides]
    if len(strides) < 5 or len(strides) % 2 == 0:
        raise ValueError(
            "MSMF requires an odd number of at least five strides, matching the manuscript."
        )
    if len(set(strides)) != len(strides):
        raise ValueError(f"MSMF strides contain duplicates: {strides}")
    invalid = [value for value in strides if value <= 0 or value >= crop_size]
    if invalid:
        raise ValueError(
            f"Every MSMF stride must satisfy 0 < stride < crop_size={crop_size}; "
            f"invalid={invalid}"
        )
    # The paper requires no obvious multiple relation.  Warn conservatively by
    # rejecting exact integer multiples, although this is unlikely for strides
    # close to the crop size.
    for left_index, left in enumerate(strides):
        for right in strides[left_index + 1 :]:
            large, small = max(left, right), min(left, right)
            if large % small == 0:
                raise ValueError(
                    f"MSMF strides {small} and {large} have an exact multiple relation."
                )
    return strides


def _resolve_cuda_devices(device_ids: Sequence[int]) -> list[torch.device]:
    if not torch.cuda.is_available():
        return [torch.device("cpu")]
    available = torch.cuda.device_count()
    valid = [int(value) for value in device_ids if 0 <= int(value) < available]
    if not valid:
        valid = [0]
    return [torch.device(f"cuda:{value}") for value in valid]


def _window_starts(length: int, crop_size: int, stride: int) -> list[int]:
    if length <= crop_size:
        return [0]
    last = length - crop_size
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return starts


def _sliding_windows(height: int, width: int, crop_size: int, stride: int):
    rows = _window_starts(height, crop_size, stride)
    cols = _window_starts(width, crop_size, stride)
    for row in rows:
        for col in cols:
            yield Window(col, row, crop_size, crop_size)


def _sliding_window_count(height: int, width: int, crop_size: int, stride: int) -> int:
    return (
        len(_window_starts(height, crop_size, stride))
        * len(_window_starts(width, crop_size, stride))
    )


def _batched(iterable: Iterable, batch_size: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _centre_weight_kernel(size: int, minimum_weight: float = 0.05) -> np.ndarray:
    if not 0 < minimum_weight <= 1:
        raise ValueError("minimum_weight must be in (0, 1].")
    if size == 1:
        return np.ones((1, 1), dtype=np.float32)
    axis = np.hanning(size).astype(np.float32)
    kernel = np.outer(axis, axis)
    maximum = float(kernel.max())
    if maximum > 0:
        kernel /= maximum
    kernel = minimum_weight + (1.0 - minimum_weight) * kernel
    return kernel.astype(np.float32, copy=False)


def _read_raw_batch(
    factor_sources,
    region_source,
    windows: Sequence[Window],
    crop_size: int,
):
    raw_batch = []
    valid_batch = []
    for window in windows:
        arrays = []
        valid = None
        for source in factor_sources:
            fill = source.nodata if source.nodata is not None else np.nan
            data = source.read(
                1,
                window=window,
                boundless=True,
                fill_value=fill,
                out_dtype="float32",
            )
            current_valid = _valid(data, source.nodata)
            safe = data.copy()
            safe[~current_valid] = 0.0
            arrays.append(safe)
            valid = current_valid if valid is None else valid & current_valid

        region_fill = (
            region_source.nodata if region_source.nodata is not None else 0
        )
        region = region_source.read(
            1,
            window=window,
            boundless=True,
            fill_value=region_fill,
        )
        region_valid = _valid(region, region_source.nodata) & (region > 0)
        valid &= region_valid

        raw = np.stack(arrays, axis=0).astype(np.float32, copy=False)
        raw[:, ~valid] = 0.0
        raw_batch.append(raw)
        valid_batch.append(valid)

    return (
        np.stack(raw_batch, axis=0),
        np.stack(valid_batch, axis=0),
    )


def _transform_batch(
    raw_batch: np.ndarray,
    valid_batch: np.ndarray,
    transformer: FrozenFoldFeatureTransformer,
) -> np.ndarray:
    batch, bands, height, width = raw_batch.shape
    transformed = np.empty_like(raw_batch, dtype=np.float32)
    for index in range(batch):
        values = raw_batch[index].reshape(bands, -1).T
        current = transformer.transform(values).T.reshape(bands, height, width)
        current = np.asarray(current, dtype=np.float32)
        current[:, ~valid_batch[index]] = 0.0
        transformed[index] = current
    return transformed


def _predict_with_model(
    model,
    transformed: np.ndarray,
    device: torch.device,
    use_tta: bool,
    use_amp: bool,
) -> np.ndarray:
    tensor = torch.from_numpy(transformed).to(device, non_blocking=True)
    original_height, original_width = tensor.shape[-2:]
    pad_height = (16 - original_height % 16) % 16
    pad_width = (16 - original_width % 16) % 16
    if pad_height or pad_width:
        tensor = F.pad(tensor, (0, pad_width, 0, pad_height))

    amp_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if use_amp and device.type == "cuda"
        else nullcontext()
    )
    with torch.inference_mode(), amp_context:
        probability, _logits = predict_probabilities(model, tensor, use_tta=use_tta)
    probability = probability[
        :, :original_height, :original_width
    ].float().cpu().numpy()
    return probability.astype(np.float32, copy=False)


def _prepare_fold_predictors(
    experiment_dir: Path,
    folds: dict[int, Path],
    selected_regions: Sequence[int],
    model_name: str,
    sampling_method: str,
    factor_names: Sequence[str],
    devices: Sequence[torch.device],
):
    predictors = []
    thresholds = {}
    for index, test_region in enumerate(selected_regions):
        fold_dir = folds[test_region]
        fold_audit_path = fold_dir / "fold_leakage_audit.json"
        fold_audit = _load_json(fold_audit_path)
        transformer = FrozenFoldFeatureTransformer.from_dict(
            fold_audit["feature_transformer"]
        )
        if list(transformer.factor_names) != list(factor_names):
            raise RuntimeError(
                f"Fold {test_region} factor order differs from validation_protocol.json: "
                f"{list(transformer.factor_names)} != {list(factor_names)}"
            )

        model_dir = (
            fold_dir
            / sampling_method.upper()
            / f"Model_{model_name}"
        )
        weights_path = model_dir / "best_model_weight.pth"
        metrics_path = model_dir / "test_metrics.json"
        if not weights_path.is_file():
            raise FileNotFoundError(f"Missing model weights: {weights_path}")
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Missing model metrics: {metrics_path}")

        device = devices[index % len(devices)]
        model = _load_deep_model(
            model_name,
            len(factor_names),
            weights_path,
            device,
        )
        metrics = _load_json(metrics_path)
        threshold = float(metrics["threshold_selected_on_validation"])
        thresholds[int(test_region)] = threshold
        predictors.append(
            {
                "test_region": int(test_region),
                "fold_dir": fold_dir,
                "model_dir": model_dir,
                "transformer": transformer,
                "model": model,
                "device": device,
                "threshold": threshold,
            }
        )
        console(
            f"Loaded Fold test-region={test_region} | device={device} | "
            f"validation threshold={threshold:.3f}"
        )
    return predictors, thresholds


def _profile_from_reference(reference_source, dtype="float32", nodata=NODATA):
    profile = reference_source.profile.copy()
    profile.update(
        driver="GTiff",
        dtype=dtype,
        count=1,
        nodata=nodata,
        compress="deflate",
        predictor=3 if dtype == "float32" else 2,
        zlevel=4,
        tiled=True,
        blockxsize=512,
        blockysize=512,
        BIGTIFF="IF_SAFER",
    )
    return profile


def _write_stride_surface(
    sum_map: np.memmap,
    weight_map: np.memmap,
    destination_path: Path,
    profile: dict,
    tile_size: int,
):
    partial = destination_path.with_name(f".{destination_path.name}.partial")
    partial.unlink(missing_ok=True)
    try:
        with rasterio.open(partial, "w", **profile) as destination:
            windows = (
                Window(
                    col,
                    row,
                    min(tile_size, destination.width - col),
                    min(tile_size, destination.height - row),
                )
                for row in range(0, destination.height, tile_size)
                for col in range(0, destination.width, tile_size)
            )
            for window in track(
                windows,
                total=window_count(destination.height, destination.width, tile_size),
                desc=f"Writing {destination_path.stem}",
                unit="tile",
            ):
                r0 = int(window.row_off)
                c0 = int(window.col_off)
                r1 = r0 + int(window.height)
                c1 = c0 + int(window.width)
                sums = np.asarray(sum_map[r0:r1, c0:c1], dtype=np.float32)
                weights = np.asarray(weight_map[r0:r1, c0:c1], dtype=np.float32)
                output = np.full(sums.shape, NODATA, dtype=np.float32)
                usable = np.isfinite(sums) & np.isfinite(weights) & (weights > 0)
                output[usable] = sums[usable] / weights[usable]
                output[usable] = np.clip(output[usable], 0.0, 1.0)
                destination.write(output, 1, window=window)
        os.replace(partial, destination_path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def _run_one_stride(
    stride: int,
    crop_size: int,
    batch_size: int,
    factor_sources,
    region_source,
    predictors,
    use_tta: bool,
    use_amp: bool,
    minimum_weight: float,
    work_dir: Path,
    stride_path: Path,
    profile: dict,
    tile_size: int,
):
    height, width = region_source.height, region_source.width
    sum_path = work_dir / f".stride_{stride}_sum.float32.dat"
    weight_path = work_dir / f".stride_{stride}_weight.float32.dat"
    sum_path.unlink(missing_ok=True)
    weight_path.unlink(missing_ok=True)

    sum_map = np.memmap(
        sum_path, mode="w+", dtype="float32", shape=(height, width)
    )
    weight_map = np.memmap(
        weight_path, mode="w+", dtype="float32", shape=(height, width)
    )
    sum_map[:] = 0.0
    weight_map[:] = 0.0
    sum_map.flush()
    weight_map.flush()

    kernel = _centre_weight_kernel(crop_size, minimum_weight)
    windows = _sliding_windows(height, width, crop_size, stride)
    total_windows = _sliding_window_count(height, width, crop_size, stride)
    total_batches = int(math.ceil(total_windows / batch_size))
    batch_progress = track(
        _batched(windows, batch_size),
        total=total_batches,
        desc=f"MSMF stride={stride}",
        unit="batch",
        leave=True,
    )

    try:
        for window_batch in batch_progress:
            raw_batch, valid_batch = _read_raw_batch(
                factor_sources,
                region_source,
                window_batch,
                crop_size,
            )

            fold_predictions = []
            for predictor in predictors:
                transformed = _transform_batch(
                    raw_batch,
                    valid_batch,
                    predictor["transformer"],
                )
                probabilities = _predict_with_model(
                    predictor["model"],
                    transformed,
                    predictor["device"],
                    use_tta=use_tta,
                    use_amp=use_amp,
                )
                fold_predictions.append(probabilities)

            ensemble_probability = np.median(
                np.stack(fold_predictions, axis=0),
                axis=0,
            ).astype(np.float32)

            for index, window in enumerate(window_batch):
                row = int(window.row_off)
                col = int(window.col_off)
                height_here = min(crop_size, height - row)
                width_here = min(crop_size, width - col)
                if height_here <= 0 or width_here <= 0:
                    continue
                valid = valid_batch[index, :height_here, :width_here]
                if not np.any(valid):
                    continue
                current_weight = (
                    kernel[:height_here, :width_here] * valid.astype(np.float32)
                )
                probability = ensemble_probability[
                    index, :height_here, :width_here
                ]
                sum_map[
                    row : row + height_here,
                    col : col + width_here,
                ] += probability * current_weight
                weight_map[
                    row : row + height_here,
                    col : col + width_here,
                ] += current_weight

        sum_map.flush()
        weight_map.flush()
        _write_stride_surface(
            sum_map,
            weight_map,
            stride_path,
            profile,
            tile_size,
        )
    finally:
        del sum_map
        del weight_map
        sum_path.unlink(missing_ok=True)
        weight_path.unlink(missing_ok=True)


def _fuse_stride_maps(
    stride_paths: Sequence[Path],
    output_path: Path,
    binary_path: Path | None,
    disagreement_path: Path | None,
    threshold: float | None,
    tile_size: int,
    overwrite: bool,
):
    for path in (output_path, binary_path, disagreement_path):
        if path is not None and path.exists() and not overwrite:
            raise FileExistsError(
                f"Output already exists: {path}. Use --overwrite to replace it."
            )

    probability_partial = output_path.with_name(f".{output_path.name}.partial")
    binary_partial = (
        binary_path.with_name(f".{binary_path.name}.partial")
        if binary_path is not None else None
    )
    disagreement_partial = (
        disagreement_path.with_name(f".{disagreement_path.name}.partial")
        if disagreement_path is not None else None
    )
    probability_partial.unlink(missing_ok=True)
    if binary_partial:
        binary_partial.unlink(missing_ok=True)
    if disagreement_partial:
        disagreement_partial.unlink(missing_ok=True)

    with ExitStack() as stack:
        sources = [stack.enter_context(rasterio.open(path)) for path in stride_paths]
        reference = sources[0]
        probability_profile = _profile_from_reference(
            reference, dtype="float32", nodata=NODATA
        )
        probability_destination = stack.enter_context(
            rasterio.open(probability_partial, "w", **probability_profile)
        )

        if binary_partial is not None:
            if threshold is None:
                raise ValueError("A threshold is required when writing a binary map.")
            binary_profile = _profile_from_reference(
                reference, dtype="uint8", nodata=BINARY_NODATA
            )
            binary_destination = stack.enter_context(
                rasterio.open(binary_partial, "w", **binary_profile)
            )
        else:
            binary_destination = None

        if disagreement_partial is not None:
            disagreement_destination = stack.enter_context(
                rasterio.open(
                    disagreement_partial,
                    "w",
                    **probability_profile,
                )
            )
        else:
            disagreement_destination = None

        windows = (
            Window(
                col,
                row,
                min(tile_size, reference.width - col),
                min(tile_size, reference.height - row),
            )
            for row in range(0, reference.height, tile_size)
            for col in range(0, reference.width, tile_size)
        )
        valid_cells = 0
        for window in track(
            windows,
            total=window_count(reference.height, reference.width, tile_size),
            desc="Pixel-wise MSMF median fusion",
            unit="tile",
            leave=True,
        ):
            arrays = []
            for source in sources:
                data = source.read(1, window=window, out_dtype="float32")
                valid = _valid(data, source.nodata)
                arrays.append(np.where(valid, data, np.nan))
            stack_array = np.stack(arrays, axis=0)
            all_invalid = np.all(~np.isfinite(stack_array), axis=0)

            with np.errstate(all="ignore"):
                median = np.nanmedian(stack_array, axis=0)
            median[all_invalid] = NODATA
            median = median.astype(np.float32)
            usable = ~all_invalid & np.isfinite(median)
            median[usable] = np.clip(median[usable], 0.0, 1.0)
            probability_destination.write(median, 1, window=window)
            valid_cells += int(np.count_nonzero(usable))

            if binary_destination is not None:
                binary = np.full(median.shape, BINARY_NODATA, dtype=np.uint8)
                binary[usable] = (median[usable] >= float(threshold)).astype(np.uint8)
                binary_destination.write(binary, 1, window=window)

            if disagreement_destination is not None:
                # Robust cross-stride disagreement: median absolute deviation.
                with np.errstate(all="ignore"):
                    deviation = np.nanmedian(
                        np.abs(stack_array - np.expand_dims(median, axis=0)),
                        axis=0,
                    )
                deviation[all_invalid] = NODATA
                disagreement_destination.write(
                    deviation.astype(np.float32),
                    1,
                    window=window,
                )

    os.replace(probability_partial, output_path)
    if binary_partial is not None:
        os.replace(binary_partial, binary_path)
    if disagreement_partial is not None:
        os.replace(disagreement_partial, disagreement_path)
    return valid_cells


def _select_model(registry: dict, requested: str | None) -> str:
    available = [row["name"] for row in registry.get("requested_models", [])]
    if not available:
        raise ValueError("model_registry.json contains no requested models.")
    if requested is not None:
        selected = canonical_model_name(requested)
        if selected not in available:
            raise ValueError(
                f"Model {selected!r} was not trained in this experiment; "
                f"available={available}"
            )
    elif "landslidenet" in available:
        selected = "landslidenet"
    else:
        deep = [
            name for name in available
            if MODEL_SPECS[name].family == "deep"
        ]
        if not deep:
            raise ValueError(
                "MSMF requires a dense deep model, but this experiment contains "
                f"only: {available}"
            )
        selected = deep[0]
    if MODEL_SPECS[selected].family != "deep":
        raise ValueError(
            f"MSMF full-domain mapping requires a dense deep model; {selected} is "
            f"{MODEL_SPECS[selected].family}."
        )
    return selected


def _atomic_write_json(path: Path, data: dict):
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(partial, path)


def run_deep_msmf(args):
    configure_progress(bool(args.progress))
    experiment_dir, xml_params = resolve_experiment_dir(
        args.source,
        explicit_experiment_dir=args.experiment_dir,
    )
    protocol = _load_json(experiment_dir / "validation_protocol.json")
    registry = _load_json(experiment_dir / "model_registry.json")

    model_name = _select_model(registry, args.model)
    sampling_method = (
        args.sampling_method
        or xml_params.get("msmf_sampling_method")
        or "dwss"
    ).strip().lower()
    trained_sampling = list(protocol.get("sampling_methods", []))
    if sampling_method not in trained_sampling:
        raise ValueError(
            f"Sampling method {sampling_method!r} was not trained; "
            f"available={trained_sampling}"
        )

    crop_size = int(
        args.crop_size
        or protocol["hyperparameters_fixed_before_outer_test"]["crop_size"]
    )
    configured_strides = (
        args.strides
        or _parse_int_list(xml_params.get("msmf_strides"))
        or _default_strides(crop_size)
    )
    strides = _validate_strides(configured_strides, crop_size)

    device_ids = (
        args.device_ids
        if args.device_ids is not None
        else _parse_int_tokens(xml_params.get("device_ids"), default=(0,))
    )
    devices = _resolve_cuda_devices(device_ids)

    factors = [str(Path(path).expanduser()) for path in protocol["factor_paths"]]
    regions_path = str(Path(protocol["macro_regions"]).expanduser())
    validate_aligned_rasters(regions_path, factors)
    factor_names = [Path(path).stem for path in factors]

    folds = discover_folds(experiment_dir)
    expected_regions = sorted(map(int, protocol["region_ids"]))
    if set(folds) != set(expected_regions):
        raise RuntimeError(
            "A full deployment ensemble requires all regional fold artifacts. "
            f"Expected {expected_regions}, found {sorted(folds)}."
        )
    selected_regions = (
        [int(value) for value in args.fold_regions]
        if args.fold_regions
        else expected_regions
    )
    unknown_regions = sorted(set(selected_regions) - set(folds))
    if unknown_regions:
        raise ValueError(f"Unknown fold test regions: {unknown_regions}")
    if len(set(selected_regions)) != len(selected_regions):
        raise ValueError("--fold-regions must not contain duplicates.")

    output_dir = Path(
        args.output_dir
        or xml_params.get("prediction_output")
        or xml_params.get("msmf_output_dir")
        or experiment_dir / "full_maps"
    ).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = args.output_name or f"msmf_full_{model_name}_{sampling_method}"
    probability_path = output_dir / f"{stem}_probability.tif"
    binary_path = output_dir / f"{stem}_class.tif" if args.write_binary else None
    disagreement_path = (
        output_dir / f"{stem}_stride_mad.tif"
        if args.write_disagreement else None
    )
    metadata_path = output_dir / f"{stem}_metadata.json"
    for path in (probability_path, binary_path, disagreement_path, metadata_path):
        if path is not None and path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {path}. Use --overwrite to replace it."
            )

    work_dir = output_dir / f".{stem}_work"
    protocol_tta = bool(protocol.get("evaluation_tta", False))
    use_tta = protocol_tta if args.use_protocol_tta else bool(args.tta)
    run_contract = {
        "source_experiment": str(experiment_dir),
        "model": model_name,
        "sampling_method": sampling_method,
        "fold_regions": selected_regions,
        "factor_paths": factors,
        "crop_size": crop_size,
        "strides": strides,
        "batch_size": int(args.batch_size),
        "minimum_weight": float(args.minimum_weight),
        "use_tta": use_tta,
        "use_amp": bool(args.amp),
        "fold_fusion": "pixelwise_median_before_window_blending",
        "within_stride_fusion": "centre_weighted_overlap_average",
        "cross_stride_fusion": "pixelwise_median",
    }
    contract_path = work_dir / "run_contract.json"
    if work_dir.exists():
        if args.overwrite:
            shutil.rmtree(work_dir)
        elif args.resume:
            if not contract_path.is_file():
                raise RuntimeError(
                    f"Cannot safely resume: missing {contract_path}"
                )
            previous = _load_json(contract_path)
            if previous != run_contract:
                raise RuntimeError(
                    "MSMF resume settings differ from the existing run contract. "
                    "Use identical arguments or --overwrite."
                )
        else:
            raise FileExistsError(
                f"An incomplete MSMF work directory already exists: {work_dir}. "
                "Use --resume to continue or --overwrite to restart."
            )
    work_dir.mkdir(parents=True, exist_ok=True)
    if not contract_path.exists():
        _atomic_write_json(contract_path, run_contract)

    predictors, thresholds = _prepare_fold_predictors(
        experiment_dir,
        folds,
        selected_regions,
        model_name,
        sampling_method,
        factor_names,
        devices,
    )
    selected_threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(np.median(list(thresholds.values())))
    )

    console(
        "MSMF full-domain deployment mapping started | "
        f"model={model_name} | sampling={sampling_method} | "
        f"fold ensemble={selected_regions} | crop={crop_size} | "
        f"strides={strides} | devices={[str(value) for value in devices]} | "
        f"TTA={use_tta} | AMP={bool(args.amp)} | output={probability_path}"
    )

    stride_paths = [
        work_dir / f"{stem}_stride_{stride}.tif"
        for stride in strides
    ]

    with ExitStack() as stack:
        region_source = stack.enter_context(rasterio.open(regions_path))
        factor_sources = [
            stack.enter_context(rasterio.open(path))
            for path in factors
        ]
        profile = _profile_from_reference(
            region_source, dtype="float32", nodata=NODATA
        )

        for index, (stride, stride_path) in enumerate(
            zip(strides, stride_paths), start=1
        ):
            if args.resume and stride_path.is_file():
                console(
                    f"Resuming: complete result already exists for stride={stride}; skipping "
                    f"({index}/{len(strides)})"
                )
                continue
            console(
                f"Starting MSMF stride={stride} "
                f"({index}/{len(strides)})"
            )
            _run_one_stride(
                stride=stride,
                crop_size=crop_size,
                batch_size=int(args.batch_size),
                factor_sources=factor_sources,
                region_source=region_source,
                predictors=predictors,
                use_tta=use_tta,
                use_amp=bool(args.amp),
                minimum_weight=float(args.minimum_weight),
                work_dir=work_dir,
                stride_path=stride_path,
                profile=profile,
                tile_size=int(args.fusion_tile_size),
            )
            console(f"stride={stride} completed: {stride_path}")

    valid_cells = _fuse_stride_maps(
        stride_paths=stride_paths,
        output_path=probability_path,
        binary_path=binary_path,
        disagreement_path=disagreement_path,
        threshold=selected_threshold if args.write_binary else None,
        tile_size=int(args.fusion_tile_size),
        overwrite=bool(args.overwrite),
    )

    metadata = {
        "map_type": "deployment_cv_ensemble_multi_stride_median_fusion",
        "inference_algorithm": "multi_stride_median_fusion",
        "msmf_used": True,
        "probability_raster": str(probability_path),
        "binary_raster": str(binary_path) if binary_path else None,
        "stride_disagreement_mad_raster": (
            str(disagreement_path) if disagreement_path else None
        ),
        "source_experiment": str(experiment_dir),
        "model": MODEL_SPECS[model_name].to_dict(),
        "sampling_method": sampling_method,
        "fold_models_used": selected_regions,
        "fold_validation_thresholds": thresholds,
        "deployment_binary_threshold": (
            selected_threshold if args.write_binary else None
        ),
        "threshold_policy": (
            "user_supplied"
            if args.threshold is not None
            else "median_of_fold_validation_selected_thresholds"
        ),
        "factor_paths": factors,
        "macro_region_domain_mask": regions_path,
        "crop_size": crop_size,
        "strides": strides,
        "stride_count": len(strides),
        "batch_size": int(args.batch_size),
        "devices": [str(value) for value in devices],
        "tta": use_tta,
        "amp": bool(args.amp),
        "fusion": {
            "fold_models": "pixelwise_median_for_each_window",
            "overlapping_windows_within_each_stride": (
                "centre_weighted_average; minimum edge weight="
                f"{float(args.minimum_weight)}"
            ),
            "strides": "pixelwise_nanmedian",
            "disagreement": (
                "median_absolute_deviation_across_stride_maps"
                if args.write_disagreement else None
            ),
        },
        "valid_factor_domain_cells": int(valid_cells),
        "scientific_interpretation": (
            "Deployment-oriented seamless susceptibility map. It must not be used "
            "to compute or report held-out test accuracy. The fold ensemble avoids "
            "selecting a single regional fold model, "
            "while MSMF suppresses sliding-window boundary artifacts."
        ),
    }
    _atomic_write_json(metadata_path, metadata)

    if not args.keep_intermediate:
        shutil.rmtree(work_dir, ignore_errors=True)

    console(f"MSMF full-domain probability map completed: {probability_path}")
    if binary_path:
        console(
            f"Binary map completed: {binary_path} | threshold={selected_threshold:.3f}"
        )
    if disagreement_path:
        console(f"Cross-stride MAD diagnostic map completed: {disagreement_path}")
    console(f"Metadata: {metadata_path}")
    return str(probability_path)


__all__ = [
    "DEFAULT_STRIDES_512",
    "_default_strides",
    "_validate_strides",
    "run_deep_msmf",
]
