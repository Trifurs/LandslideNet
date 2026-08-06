"""Leakage-safe data and sampling utilities for continuous regional hold-out tests.

The key design rule in this module is that every learned preprocessing object is
fitted from the inner-training macro-regions only.  The held-out validation and
test regions are transformed with frozen training-region parameters.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import jenkspy
import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde
from torch.utils.data import DataLoader, Dataset

from .progress import timed_task, track, window_count


VECTOR_INVENTORY_SUFFIXES = {
    ".shp",
    ".gpkg",
    ".geojson",
    ".json",
    ".fgb",
}


def iter_windows(height: int, width: int, size: int) -> Iterable[Window]:
    for row_off in range(0, height, size):
        for col_off in range(0, width, size):
            yield Window(
                col_off,
                row_off,
                min(size, width - col_off),
                min(size, height - row_off),
            )


def list_factor_paths(factors_dir: str) -> list[str]:
    paths = sorted(
        str(path)
        for path in Path(factors_dir).iterdir()
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    )
    if not paths:
        raise FileNotFoundError(f"No factor GeoTIFFs found in {factors_dir}")
    return paths


def _same_transform(left, right, atol: float = 1e-8) -> bool:
    return bool(np.allclose(tuple(left), tuple(right), rtol=0.0, atol=atol))


def validate_aligned_rasters(reference_path: str, other_paths: Sequence[str]) -> dict:
    with rasterio.open(reference_path) as reference:
        profile = {
            "height": reference.height,
            "width": reference.width,
            "transform": reference.transform,
            "crs": reference.crs,
        }

    other_paths = list(other_paths)
    for path in track(
        other_paths,
        total=len(other_paths),
        desc="检查栅格对齐",
        unit="raster",
    ):
        with rasterio.open(path) as source:
            problems = []
            if (source.height, source.width) != (profile["height"], profile["width"]):
                problems.append(
                    f"shape {source.height}x{source.width} != "
                    f"{profile['height']}x{profile['width']}"
                )
            if source.crs != profile["crs"]:
                problems.append(f"CRS {source.crs} != {profile['crs']}")
            if not _same_transform(source.transform, profile["transform"]):
                problems.append("affine transform differs")
            if problems:
                raise ValueError(f"Raster grid mismatch for {path}: {'; '.join(problems)}")
    return profile


def is_vector_inventory(path: str | os.PathLike) -> bool:
    """Return whether an inventory path is a supported point-vector dataset."""
    return Path(path).suffix.lower() in VECTOR_INVENTORY_SUFFIXES


def _vector_inventory_cells(
    inventory_path: str,
    grid_path: str,
) -> tuple[np.ndarray, dict]:
    """Project point inventory geometries onto unique cells of ``grid_path``."""
    try:
        import geopandas as gpd
    except ImportError as error:
        raise ModuleNotFoundError(
            "Point-vector inventories require geopandas. Install/update environment.yml."
        ) from error

    inventory = gpd.read_file(inventory_path)
    feature_count = int(len(inventory))
    if feature_count == 0:
        raise RuntimeError(f"Point inventory is empty: {inventory_path}")
    if inventory.crs is None:
        raise ValueError(
            f"Point inventory has no CRS and cannot be aligned safely: {inventory_path}"
        )
    inventory = inventory.loc[
        inventory.geometry.notna() & ~inventory.geometry.is_empty
    ].copy()
    inventory = inventory.explode(index_parts=False, ignore_index=True)
    non_points = sorted(
        set(inventory.geometry.geom_type) - {"Point"}
    )
    if non_points:
        raise ValueError(
            "The landslide inventory must contain point geometries only; "
            f"found geometry types {non_points}."
        )

    with rasterio.open(grid_path) as grid:
        grid_crs = grid.crs
        height, width = grid.height, grid.width
        transform = grid.transform
    if grid_crs is None:
        raise ValueError(f"Reference raster has no CRS: {grid_path}")
    source_crs = str(inventory.crs)
    if inventory.crs != grid_crs:
        inventory = inventory.to_crs(grid_crs)

    xs = inventory.geometry.x.to_numpy(dtype=np.float64)
    ys = inventory.geometry.y.to_numpy(dtype=np.float64)
    rows, cols = rasterio.transform.rowcol(transform, xs, ys)
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    in_grid = (
        (rows >= 0)
        & (rows < height)
        & (cols >= 0)
        & (cols < width)
    )
    cells_before_deduplication = int(np.count_nonzero(in_grid))
    cells = np.column_stack((rows[in_grid], cols[in_grid])).astype(
        np.int64,
        copy=False,
    )
    if len(cells):
        cells = np.unique(cells, axis=0)
    if len(cells) == 0:
        raise RuntimeError(
            "No point-inventory geometry falls inside the macro-region raster grid."
        )
    return cells, {
        "inventory_type": "point_vector",
        "inventory_path": str(inventory_path),
        "source_crs": source_crs,
        "grid_crs": str(grid_crs),
        "input_features": feature_count,
        "nonempty_point_geometries": int(len(inventory)),
        "geometries_outside_grid": int(np.count_nonzero(~in_grid)),
        "point_cells_before_deduplication": cells_before_deduplication,
        "duplicate_point_cells_removed": int(
            cells_before_deduplication - len(cells)
        ),
        "unique_point_cells_in_grid": int(len(cells)),
    }


def _count_region_cells(
    regions_path: str,
    chunk_size: int,
) -> dict[int, int]:
    """Count valid positive region identifiers without loading the raster at once."""
    counts: dict[int, int] = {}
    with rasterio.open(regions_path) as regions:
        windows = iter_windows(regions.height, regions.width, chunk_size)
        for window in track(
            windows,
            total=window_count(regions.height, regions.width, chunk_size),
            desc="统计宏区域有效像元",
            unit="tile",
        ):
            data = regions.read(1, window=window)
            valid = _valid_values(data, regions.nodata)
            region_ids = _integer_regions(data, valid)
            valid &= region_ids > 0
            values, frequencies = np.unique(region_ids[valid], return_counts=True)
            for value, frequency in zip(values, frequencies):
                key = int(value)
                counts[key] = counts.get(key, 0) + int(frequency)
    return counts


def _valid_values(data: np.ndarray, nodata) -> np.ndarray:
    valid = np.isfinite(data)
    if nodata is not None and np.isfinite(nodata):
        valid &= ~np.isclose(data, nodata, equal_nan=True)
    return valid


def _integer_regions(data: np.ndarray, valid: np.ndarray) -> np.ndarray:
    rounded = np.zeros(data.shape, dtype=np.int64)
    rounded[valid] = np.rint(data[valid]).astype(np.int64, copy=False)
    if np.any(valid & ~np.isclose(data, rounded, rtol=0.0, atol=1e-6)):
        raise ValueError("Macro-region raster must contain integer region identifiers.")
    return rounded


def audit_region_connectivity(
    regions_path: str,
    region_ids: Sequence[int] | None = None,
) -> dict:
    """Require each positive region identifier to be one 4-neighbour component."""
    with timed_task("读取宏区域并检查连通性"):
        with rasterio.open(regions_path) as source:
            data = source.read(1, masked=True)
            values = np.asarray(data.data)
            valid = ~np.ma.getmaskarray(data) & np.isfinite(values)
            regions = _integer_regions(values, valid)
            valid &= regions > 0

    available = sorted(map(int, np.unique(regions[valid])))
    selected = sorted(set(map(int, region_ids))) if region_ids is not None else available
    missing = sorted(set(selected) - set(available))
    if missing:
        raise ValueError(f"Macro-region raster is missing selected identifiers: {missing}")

    structure = ndimage.generate_binary_structure(2, 1)
    audit = {}
    disconnected = {}
    for region_id in track(
        selected,
        total=len(selected),
        desc="检查区域连通性",
        unit="region",
    ):
        components, component_count = ndimage.label(
            regions == region_id,
            structure=structure,
        )
        sizes = np.bincount(components.ravel())[1:]
        ordered_sizes = [int(value) for value in sorted(sizes, reverse=True)]
        audit[region_id] = {
            "cell_count": int(np.count_nonzero(regions == region_id)),
            "component_count": int(component_count),
            "component_sizes": ordered_sizes,
        }
        if component_count != 1:
            disconnected[region_id] = ordered_sizes
    if disconnected:
        raise ValueError(
            "Every macro-region identifier must be one continuous 4-neighbour "
            f"component; disconnected identifiers and component sizes: {disconnected}"
        )
    return {
        "connectivity": "4-neighbour",
        "region_ids": selected,
        "regions": audit,
        "all_regions_are_single_components": True,
    }


def collect_positive_points(
    inventory_path: str,
    regions_path: str,
    positive_value: int = 1,
    chunk_size: int = 1024,
    return_audit: bool = False,
):
    """Return unique ``[row, col, region]`` positives and regional counts.

    A point Shapefile/GeoPackage is rasterised by cell centre membership on the
    frozen macro-region grid. Raster inventories remain supported for backward
    compatibility.
    """
    if is_vector_inventory(inventory_path):
        cells, audit = _vector_inventory_cells(inventory_path, regions_path)
        region_values, region_valid = read_point_features(
            [regions_path],
            cells,
            tile_size=chunk_size,
        )
        values = region_values[:, 0]
        rounded = np.zeros(len(values), dtype=np.int64)
        rounded[region_valid] = np.rint(values[region_valid]).astype(np.int64)
        integer_valid = region_valid & np.isclose(
            values,
            rounded,
            rtol=0.0,
            atol=1e-6,
        )
        selected = integer_valid & (rounded > 0)
        positives = np.column_stack((cells[selected], rounded[selected])).astype(
            np.int64,
            copy=False,
        )
        if len(positives) == 0:
            raise RuntimeError(
                "No point-inventory cells fall in valid positive macro-regions."
            )
        region_values_unique, frequencies = np.unique(
            positives[:, 2],
            return_counts=True,
        )
        positive_counts = {
            int(value): int(frequency)
            for value, frequency in zip(region_values_unique, frequencies)
        }
        region_cell_counts = _count_region_cells(regions_path, chunk_size)
        background_counts = {
            region_id: int(count - positive_counts.get(region_id, 0))
            for region_id, count in region_cell_counts.items()
        }
        audit.update({
            "point_cells_outside_valid_macro_regions": int(
                len(cells) - len(positives)
            ),
            "positive_cells_used": int(len(positives)),
            "positive_cells_by_region": positive_counts,
            "positive_value_ignored_for_point_vector": True,
        })
        if audit["point_cells_outside_valid_macro_regions"]:
            warnings.warn(
                f"{audit['point_cells_outside_valid_macro_regions']} point-inventory "
                "cells fall outside valid positive macro-regions and are excluded.",
                RuntimeWarning,
                stacklevel=2,
            )
        result = (positives, positive_counts, background_counts)
        return (*result, audit) if return_audit else result

    positives = []
    positive_counts: dict[int, int] = {}
    background_counts: dict[int, int] = {}

    with rasterio.open(inventory_path) as inventory, rasterio.open(regions_path) as regions:
        windows = iter_windows(inventory.height, inventory.width, chunk_size)
        for window in track(
            windows,
            total=window_count(inventory.height, inventory.width, chunk_size),
            desc="扫描滑坡清单",
            unit="tile",
        ):
            inventory_data = inventory.read(1, window=window)
            region_data = regions.read(1, window=window)
            inventory_valid = _valid_values(inventory_data, inventory.nodata)
            region_valid = _valid_values(region_data, regions.nodata)
            region_ids = _integer_regions(region_data, region_valid)
            region_valid &= region_ids > 0

            study_mask = inventory_valid & region_valid
            positive_mask = study_mask & np.isclose(inventory_data, positive_value)
            background_mask = study_mask & ~np.isclose(inventory_data, positive_value)

            for mask, counts in (
                (positive_mask, positive_counts),
                (background_mask, background_counts),
            ):
                values, frequencies = np.unique(region_ids[mask], return_counts=True)
                for value, frequency in zip(values, frequencies):
                    region_id = int(value)
                    counts[region_id] = counts.get(region_id, 0) + int(frequency)

            local_rows, local_cols = np.nonzero(positive_mask)
            if local_rows.size:
                positives.append(
                    np.column_stack(
                        (
                            local_rows + int(window.row_off),
                            local_cols + int(window.col_off),
                            region_ids[local_rows, local_cols],
                        )
                    ).astype(np.int64, copy=False)
                )

    if not positives:
        raise RuntimeError("No positive landslide pixels were found in valid macro-regions.")
    positive_array = np.concatenate(positives)
    audit = {
        "inventory_type": "aligned_raster",
        "inventory_path": str(inventory_path),
        "positive_value": int(positive_value),
        "positive_cells_used": int(len(positive_array)),
        "positive_cells_by_region": positive_counts,
    }
    result = (positive_array, positive_counts, background_counts)
    return (*result, audit) if return_audit else result


def _candidate_mask(inventory_data, inventory_nodata, region_data, region_nodata,
                    allowed_regions, positive_value):
    inventory_valid = _valid_values(inventory_data, inventory_nodata)
    region_valid = _valid_values(region_data, region_nodata)
    region_ids = _integer_regions(region_data, region_valid)
    allowed = region_valid & np.isin(region_ids, allowed_regions)
    mask = allowed & inventory_valid & ~np.isclose(inventory_data, positive_value)
    return mask, region_ids


def _allocate_hypergeometric(counts: Sequence[int], sample_size: int,
                             rng: np.random.Generator) -> list[int]:
    remaining_population = int(sum(counts))
    remaining_sample = min(int(sample_size), remaining_population)
    allocations = []
    for index, count in enumerate(counts):
        count = int(count)
        if index == len(counts) - 1:
            draw = remaining_sample
        elif remaining_sample <= 0 or count <= 0:
            draw = 0
        else:
            draw = int(
                rng.hypergeometric(
                    ngood=count,
                    nbad=remaining_population - count,
                    nsample=remaining_sample,
                )
            )
        allocations.append(draw)
        remaining_sample -= draw
        remaining_population -= count
    return allocations


def sample_background_points(
    inventory_path: str,
    regions_path: str,
    allowed_regions: Sequence[int],
    sample_size: int,
    seed: int,
    positive_value: int = 1,
    chunk_size: int = 1024,
) -> tuple[np.ndarray, int]:
    """Draw an exact uniform sample without replacement from allowed background."""
    allowed_regions = np.asarray(sorted(set(map(int, allowed_regions))), dtype=np.int64)
    if allowed_regions.size == 0:
        raise ValueError("At least one allowed macro-region is required.")

    if is_vector_inventory(inventory_path):
        positive_cells, _audit = _vector_inventory_cells(
            inventory_path,
            regions_path,
        )
        positive_tiles: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for row, col in positive_cells:
            key = (int(row // chunk_size), int(col // chunk_size))
            positive_tiles.setdefault(key, []).append((int(row), int(col)))

        def vector_mask(window, region_data, region_nodata):
            region_valid = _valid_values(region_data, region_nodata)
            region_ids = _integer_regions(region_data, region_valid)
            mask = region_valid & np.isin(region_ids, allowed_regions)
            key = (
                int(window.row_off) // chunk_size,
                int(window.col_off) // chunk_size,
            )
            for row, col in positive_tiles.get(key, ()):
                local_row = row - int(window.row_off)
                local_col = col - int(window.col_off)
                if (
                    0 <= local_row < int(window.height)
                    and 0 <= local_col < int(window.width)
                ):
                    mask[local_row, local_col] = False
            return mask, region_ids

        window_list = []
        counts = []
        with rasterio.open(regions_path) as regions:
            windows = iter_windows(regions.height, regions.width, chunk_size)
            for window in track(
                windows,
                total=window_count(regions.height, regions.width, chunk_size),
                desc=f"统计背景候选区 {allowed_regions.tolist()}",
                unit="tile",
            ):
                region_data = regions.read(1, window=window)
                mask, _ = vector_mask(window, region_data, regions.nodata)
                window_list.append(window)
                counts.append(int(mask.sum()))

            total_candidates = int(sum(counts))
            if total_candidates == 0:
                raise RuntimeError(
                    "No vector-inventory background candidates found in regions "
                    f"{allowed_regions.tolist()}."
                )
            rng = np.random.default_rng(seed)
            allocations = _allocate_hypergeometric(counts, sample_size, rng)
            selected = []
            selected_windows = [
                (window, take)
                for window, take in zip(window_list, allocations)
                if take > 0
            ]
            for window, take in track(
                selected_windows,
                total=len(selected_windows),
                desc="抽取背景候选像元",
                unit="tile",
            ):
                region_data = regions.read(1, window=window)
                mask, region_ids = vector_mask(
                    window,
                    region_data,
                    regions.nodata,
                )
                flat_candidates = np.flatnonzero(mask)
                chosen = rng.choice(flat_candidates, size=take, replace=False)
                local_rows, local_cols = np.unravel_index(chosen, mask.shape)
                selected.append(
                    np.column_stack(
                        (
                            local_rows + int(window.row_off),
                            local_cols + int(window.col_off),
                            region_ids[local_rows, local_cols],
                        )
                    ).astype(np.int64, copy=False)
                )
        points = (
            np.concatenate(selected)
            if selected
            else np.empty((0, 3), dtype=np.int64)
        )
        return points, total_candidates

    window_list = []
    counts = []
    with rasterio.open(inventory_path) as inventory, rasterio.open(regions_path) as regions:
        windows = iter_windows(inventory.height, inventory.width, chunk_size)
        for window in track(
            windows,
            total=window_count(inventory.height, inventory.width, chunk_size),
            desc=f"统计背景候选区 {allowed_regions.tolist()}",
            unit="tile",
        ):
            inventory_data = inventory.read(1, window=window)
            region_data = regions.read(1, window=window)
            mask, _ = _candidate_mask(
                inventory_data,
                inventory.nodata,
                region_data,
                regions.nodata,
                allowed_regions,
                positive_value,
            )
            window_list.append(window)
            counts.append(int(mask.sum()))

        total_candidates = int(sum(counts))
        if total_candidates == 0:
            raise RuntimeError(
                f"No background candidates found in regions {allowed_regions.tolist()}."
            )

        rng = np.random.default_rng(seed)
        allocations = _allocate_hypergeometric(counts, sample_size, rng)
        selected = []
        selected_windows = [
            (window, take)
            for window, take in zip(window_list, allocations)
            if take > 0
        ]
        for window, take in track(
            selected_windows,
            total=len(selected_windows),
            desc="抽取背景候选像元",
            unit="tile",
        ):
            inventory_data = inventory.read(1, window=window)
            region_data = regions.read(1, window=window)
            mask, region_ids = _candidate_mask(
                inventory_data,
                inventory.nodata,
                region_data,
                regions.nodata,
                allowed_regions,
                positive_value,
            )
            flat_candidates = np.flatnonzero(mask)
            chosen = rng.choice(flat_candidates, size=take, replace=False)
            local_rows, local_cols = np.unravel_index(chosen, mask.shape)
            selected.append(
                np.column_stack(
                    (
                        local_rows + int(window.row_off),
                        local_cols + int(window.col_off),
                        region_ids[local_rows, local_cols],
                    )
                ).astype(np.int64, copy=False)
            )

    points = np.concatenate(selected) if selected else np.empty((0, 3), dtype=np.int64)
    return points, total_candidates


def read_point_features(
    factor_paths: Sequence[str],
    points: np.ndarray,
    tile_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Read factor values at [row, col, ...] points with tiled raster access."""
    points = np.asarray(points, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("points must have shape [n, >=2] with row and column first.")
    if len(points) == 0:
        return np.empty((0, len(factor_paths)), dtype=np.float32), np.zeros(0, dtype=bool)

    rows, cols = points[:, 0], points[:, 1]
    tile_rows, tile_cols = rows // tile_size, cols // tile_size
    order = np.lexsort((tile_cols, tile_rows))
    ordered_tile_rows = tile_rows[order]
    ordered_tile_cols = tile_cols[order]
    boundaries = np.flatnonzero(
        (np.diff(ordered_tile_rows) != 0) | (np.diff(ordered_tile_cols) != 0)
    ) + 1
    groups = np.split(order, boundaries)

    features = np.empty((len(points), len(factor_paths)), dtype=np.float32)
    valid = np.ones(len(points), dtype=bool)
    task_total = len(factor_paths) * len(groups)
    feature_progress = track(
        total=task_total,
        desc=f"读取点位因子 ({len(points):,} points)",
        unit="tile",
    )
    try:
        for factor_index, path in enumerate(factor_paths):
            with rasterio.open(path) as source:
                for indices in groups:
                    row_off = int(tile_rows[indices[0]] * tile_size)
                    col_off = int(tile_cols[indices[0]] * tile_size)
                    window = Window(
                        col_off,
                        row_off,
                        min(tile_size, source.width - col_off),
                        min(tile_size, source.height - row_off),
                    )
                    data = source.read(1, window=window)
                    local_rows = rows[indices] - row_off
                    local_cols = cols[indices] - col_off
                    values = data[local_rows, local_cols]
                    features[indices, factor_index] = values
                    valid[indices] &= _valid_values(values, source.nodata)
                    feature_progress.update(1)
    finally:
        feature_progress.close()
    valid &= np.all(np.isfinite(features), axis=1)
    return features, valid


def compute_region_factor_ranges(
    factor_paths: Sequence[str],
    regions_path: str,
    region_ids: Sequence[int],
    chunk_size: int = 1024,
) -> dict[int, dict[str, np.ndarray]]:
    """Compute exact per-region factor ranges for leakage-safe later combining.

    The sufficient statistics of different regions remain isolated.  An outer
    fold constructs its normalizer by combining only its inner-training rows.
    """
    selected_regions = sorted(set(map(int, region_ids)))
    if not selected_regions:
        raise ValueError("At least one region is required for factor statistics.")
    n_factors = len(factor_paths)
    ranges = {
        region_id: {
            "minima": np.full(n_factors, np.inf, dtype=np.float64),
            "maxima": np.full(n_factors, -np.inf, dtype=np.float64),
            "counts": np.zeros(n_factors, dtype=np.int64),
        }
        for region_id in selected_regions
    }

    with rasterio.open(regions_path) as regions:
        statistics_progress = track(
            total=len(factor_paths)
            * window_count(regions.height, regions.width, chunk_size),
            desc="计算训练区因子范围",
            unit="tile",
        )
        for factor_index, path in enumerate(factor_paths):
            with rasterio.open(path) as factor:
                for window in iter_windows(regions.height, regions.width, chunk_size):
                    region_data = regions.read(1, window=window)
                    region_valid = _valid_values(region_data, regions.nodata)
                    integer_regions = _integer_regions(region_data, region_valid)
                    factor_data = factor.read(1, window=window)
                    valid = (
                        region_valid
                        & _valid_values(factor_data, factor.nodata)
                        & np.isin(integer_regions, selected_regions)
                    )
                    for region_id in np.unique(integer_regions[valid]):
                        values = factor_data[valid & (integer_regions == region_id)]
                        stats = ranges[int(region_id)]
                        stats["minima"][factor_index] = min(
                            stats["minima"][factor_index], float(np.min(values))
                        )
                        stats["maxima"][factor_index] = max(
                            stats["maxima"][factor_index], float(np.max(values))
                        )
                        stats["counts"][factor_index] += int(len(values))
                    statistics_progress.update(1)
        statistics_progress.close()

    factor_names = [Path(path).stem for path in factor_paths]
    for region_id, stats in ranges.items():
        missing = [
            factor_names[index]
            for index, count in enumerate(stats["counts"])
            if count == 0
        ]
        if missing:
            raise RuntimeError(
                f"Macro-region {region_id} has no valid pixels for factors: {missing}"
            )
    return ranges


@dataclass(frozen=True)
class CategoricalFactorSpec:
    """Definition of a categorical raster already scaled to [0, 1]."""

    factor_name: str
    factor_index: int
    normalized_levels: int

    def codes(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if self.normalized_levels == 0:
            return np.rint(values).astype(np.int64)
        if self.normalized_levels < 2:
            raise ValueError(
                f"{self.factor_name}: normalized_levels must be 0 or at least 2."
            )
        return np.rint(
            np.clip(values, 0.0, 1.0) * (self.normalized_levels - 1)
        ).astype(np.int64)

    def reconstructed_values(self, codes: np.ndarray) -> np.ndarray:
        if self.normalized_levels == 0:
            return np.asarray(codes, dtype=np.float64)
        return np.asarray(codes, dtype=np.float64) / (self.normalized_levels - 1)


def parse_frequency_ratio_specs(value, factor_paths: Sequence[str]) -> tuple[CategoricalFactorSpec, ...]:
    """Parse `Geology:176,Soil:36`; level 0 means unscaled integer codes."""
    if value is None or str(value).strip() in {"", "[]", "none", "None"}:
        return ()
    factor_lookup = {
        Path(path).stem.casefold(): (index, Path(path).stem)
        for index, path in enumerate(factor_paths)
    }
    specs = []
    seen = set()
    for token in str(value).strip("[]").split(","):
        token = token.strip().strip("'\"")
        if not token:
            continue
        name, separator, level_text = token.partition(":")
        if not separator:
            raise ValueError(
                "frequency_ratio_factors entries must use FactorName:NumberOfLevels, "
                "for example Geology:176."
            )
        key = name.strip().casefold()
        if key not in factor_lookup:
            raise ValueError(
                f"Frequency-ratio factor {name!r} was not found. Available factors: "
                f"{[Path(path).stem for path in factor_paths]}"
            )
        if key in seen:
            raise ValueError(f"Duplicate frequency-ratio factor: {name}")
        seen.add(key)
        factor_index, canonical_name = factor_lookup[key]
        levels = int(level_text.strip())
        if levels == 1 or levels < 0:
            raise ValueError(
                f"{canonical_name}: levels must be 0 (raw integer codes) or at least 2."
            )
        specs.append(CategoricalFactorSpec(canonical_name, factor_index, levels))
    return tuple(sorted(specs, key=lambda spec: spec.factor_index))


def compute_training_category_counts(
    factor_paths: Sequence[str],
    regions_path: str,
    training_regions: Sequence[int],
    specs: Sequence[CategoricalFactorSpec],
    chunk_size: int = 1024,
) -> tuple[dict[str, dict[int, int]], dict[str, dict]]:
    """Count category area using only complete inner-training regions."""
    training_regions = np.asarray(sorted(set(map(int, training_regions))), dtype=np.int64)
    if training_regions.size == 0:
        raise ValueError("At least one inner-training region is required.")
    counts = {spec.factor_name: {} for spec in specs}
    residuals = {
        spec.factor_name: {"sum": 0.0, "maximum": 0.0, "count": 0}
        for spec in specs
    }
    if not specs:
        return counts, {}

    with rasterio.open(regions_path) as regions:
        sources = {
            spec.factor_name: rasterio.open(factor_paths[spec.factor_index])
            for spec in specs
        }
        try:
            windows = iter_windows(regions.height, regions.width, chunk_size)
            for window in track(
                windows,
                total=window_count(regions.height, regions.width, chunk_size),
                desc="统计训练区枚举因子",
                unit="tile",
            ):
                region_data = regions.read(1, window=window)
                region_valid = _valid_values(region_data, regions.nodata)
                region_ids = _integer_regions(region_data, region_valid)
                selected_region = region_valid & np.isin(region_ids, training_regions)
                if not np.any(selected_region):
                    continue
                for spec in specs:
                    source = sources[spec.factor_name]
                    data = source.read(1, window=window)
                    valid = selected_region & _valid_values(data, source.nodata)
                    values = np.asarray(data[valid], dtype=np.float64)
                    if values.size == 0:
                        continue
                    codes = spec.codes(values)
                    unique, frequencies = np.unique(codes, return_counts=True)
                    factor_counts = counts[spec.factor_name]
                    for code, frequency in zip(unique, frequencies):
                        key = int(code)
                        factor_counts[key] = factor_counts.get(key, 0) + int(frequency)
                    reconstructed = spec.reconstructed_values(codes)
                    difference = np.abs(values - reconstructed)
                    diagnostic = residuals[spec.factor_name]
                    diagnostic["sum"] += float(difference.sum())
                    diagnostic["maximum"] = max(
                        diagnostic["maximum"], float(difference.max(initial=0.0))
                    )
                    diagnostic["count"] += int(values.size)
        finally:
            for source in sources.values():
                source.close()

    diagnostics = {}
    for spec in specs:
        factor_counts = counts[spec.factor_name]
        if not factor_counts:
            raise RuntimeError(
                f"No valid inner-training category pixels found for {spec.factor_name}."
            )
        residual = residuals[spec.factor_name]
        diagnostics[spec.factor_name] = {
            "factor_index": spec.factor_index,
            "normalized_levels": spec.normalized_levels,
            "observed_training_categories": len(factor_counts),
            "training_area_pixels": int(sum(factor_counts.values())),
            "mean_quantization_residual": residual["sum"] / max(residual["count"], 1),
            "maximum_quantization_residual": residual["maximum"],
        }
    return counts, diagnostics


@dataclass(frozen=True)
class FrozenMinMaxNormalizer:
    factor_names: tuple[str, ...]
    minima: np.ndarray
    maxima: np.ndarray

    @classmethod
    def fit(cls, factor_paths: Sequence[str], feature_arrays: Sequence[np.ndarray]):
        usable = [np.asarray(array) for array in feature_arrays if len(array)]
        if not usable:
            raise ValueError("No inner-training features were provided for normalization.")
        features = np.concatenate(usable, axis=0).astype(np.float64, copy=False)
        if not np.all(np.isfinite(features)):
            raise ValueError("Training-only normalization features contain invalid values.")
        minima = np.min(features, axis=0)
        maxima = np.max(features, axis=0)
        return cls(
            tuple(Path(path).stem for path in factor_paths),
            minima.astype(np.float64),
            maxima.astype(np.float64),
        )

    @classmethod
    def from_region_ranges(
        cls,
        factor_paths: Sequence[str],
        region_ranges: dict[int, dict[str, np.ndarray]],
        training_regions: Sequence[int],
    ):
        training_regions = list(map(int, training_regions))
        if not training_regions:
            raise ValueError("No inner-training regions were provided for normalization.")
        missing = set(training_regions) - set(region_ranges)
        if missing:
            raise ValueError(f"Factor ranges are missing regions: {sorted(missing)}")
        minima = np.min(
            np.stack([region_ranges[region_id]["minima"] for region_id in training_regions]),
            axis=0,
        )
        maxima = np.max(
            np.stack([region_ranges[region_id]["maxima"] for region_id in training_regions]),
            axis=0,
        )
        return cls(
            tuple(Path(path).stem for path in factor_paths),
            minima.astype(np.float64),
            maxima.astype(np.float64),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Mapping[str, float]]):
        names = tuple(payload)
        if not names:
            raise ValueError("Serialized normalizer is empty.")
        return cls(
            names,
            np.asarray([payload[name]["train_min"] for name in names], dtype=np.float64),
            np.asarray([payload[name]["train_max"] for name in names], dtype=np.float64),
        )

    @property
    def scales(self):
        return np.where(self.maxima - self.minima > 1e-12, self.maxima - self.minima, 1.0)

    def transform(self, features: np.ndarray) -> np.ndarray:
        normalized = (np.asarray(features, dtype=np.float64) - self.minima) / self.scales
        return np.clip(normalized, 0.0, 1.0).astype(np.float32)

    def to_dict(self):
        return {
            name: {"train_min": float(minimum), "train_max": float(maximum)}
            for name, minimum, maximum in zip(self.factor_names, self.minima, self.maxima)
        }


@dataclass(frozen=True)
class FrozenFrequencyRatioEncoder:
    """Credibility-smoothed log-frequency-ratio fitted on inner-training data."""

    spec: CategoricalFactorSpec
    category_codes: np.ndarray
    area_counts: np.ndarray
    positive_counts: np.ndarray
    expected_positive_counts: np.ndarray
    frequency_ratios: np.ndarray
    encoded_values: np.ndarray
    smoothing: float
    log2_clip: float
    unknown_encoded_value: float = 0.5

    @classmethod
    def fit(
        cls,
        spec: CategoricalFactorSpec,
        area_counts: Mapping[int, int],
        positive_values: np.ndarray,
        smoothing: float = 0.5,
        log2_clip: float = 4.0,
    ) -> "FrozenFrequencyRatioEncoder":
        if smoothing <= 0:
            raise ValueError("Frequency-ratio smoothing must be positive.")
        if log2_clip <= 0:
            raise ValueError("Frequency-ratio log2 clip must be positive.")
        if not area_counts:
            raise ValueError(f"No training-area category counts for {spec.factor_name}.")
        positive_values = np.asarray(positive_values, dtype=np.float64)
        if positive_values.size == 0 or not np.all(np.isfinite(positive_values)):
            raise ValueError(
                f"No finite inner-training positive values for {spec.factor_name}."
            )
        positive_codes = spec.codes(positive_values)
        category_codes = np.asarray(sorted(map(int, area_counts)), dtype=np.int64)
        unknown_positive = sorted(set(map(int, np.unique(positive_codes))) - set(category_codes))
        if unknown_positive:
            raise RuntimeError(
                f"Positive categories are absent from the training-area counts for "
                f"{spec.factor_name}: {unknown_positive}"
            )
        category_area = np.asarray(
            [int(area_counts[int(code)]) for code in category_codes], dtype=np.int64
        )
        positive_lookup = dict(zip(*np.unique(positive_codes, return_counts=True)))
        category_positive = np.asarray(
            [int(positive_lookup.get(int(code), 0)) for code in category_codes],
            dtype=np.int64,
        )
        # Expected positives under no category association are proportional to
        # category area. Adding the same event-scale credibility constant to
        # observed and expected counts shrinks sparse categories toward FR=1.
        # In contrast, independent category pseudocounts in the two probability
        # distributions can spuriously give a one-pixel/zero-positive category
        # a very large FR because the area denominator is orders of magnitude larger.
        expected_positive = (
            float(category_positive.sum())
            * category_area.astype(np.float64)
            / float(category_area.sum())
        )
        frequency_ratios = (
            category_positive.astype(np.float64) + smoothing
        ) / (expected_positive + smoothing)
        log_ratios = np.clip(np.log2(frequency_ratios), -log2_clip, log2_clip)
        encoded = (log_ratios + log2_clip) / (2.0 * log2_clip)
        return cls(
            spec=spec,
            category_codes=category_codes,
            area_counts=category_area,
            positive_counts=category_positive,
            expected_positive_counts=expected_positive.astype(np.float64),
            frequency_ratios=frequency_ratios.astype(np.float64),
            encoded_values=encoded.astype(np.float64),
            smoothing=float(smoothing),
            log2_clip=float(log2_clip),
        )

    @classmethod
    def from_dict(cls, factor_name: str, payload: Mapping):
        rows = payload.get("categories", [])
        if not rows:
            raise ValueError(f"Serialized FR encoder for {factor_name} has no categories.")
        spec = CategoricalFactorSpec(
            factor_name=factor_name,
            factor_index=int(payload["factor_index"]),
            normalized_levels=int(payload["normalized_levels"]),
        )
        return cls(
            spec=spec,
            category_codes=np.asarray([row["category_code"] for row in rows], dtype=np.int64),
            area_counts=np.asarray([row["training_area_pixels"] for row in rows], dtype=np.int64),
            positive_counts=np.asarray(
                [row["training_positive_pixels"] for row in rows], dtype=np.int64
            ),
            expected_positive_counts=np.asarray([
                row["expected_positive_pixels_under_area_null"] for row in rows
            ], dtype=np.float64),
            frequency_ratios=np.asarray(
                [row["smoothed_frequency_ratio"] for row in rows], dtype=np.float64
            ),
            encoded_values=np.asarray(
                [row["encoded_value"] for row in rows], dtype=np.float64
            ),
            smoothing=float(payload["smoothing"]),
            log2_clip=float(payload["log2_clip"]),
            unknown_encoded_value=float(payload.get("unknown_encoded_value", 0.5)),
        )

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values)
        flat_codes = self.spec.codes(values.reshape(-1))
        positions = np.searchsorted(self.category_codes, flat_codes)
        matched = positions < len(self.category_codes)
        matched_indices = np.flatnonzero(matched)
        if matched_indices.size:
            matched[matched_indices] &= (
                self.category_codes[positions[matched_indices]] == flat_codes[matched_indices]
            )
        result = np.full(flat_codes.shape, self.unknown_encoded_value, dtype=np.float32)
        if np.any(matched):
            result[matched] = self.encoded_values[positions[matched]].astype(np.float32)
        return result.reshape(values.shape)

    def to_dict(self) -> dict:
        rows = []
        for code, area, positives, expected, ratio, encoded in zip(
            self.category_codes,
            self.area_counts,
            self.positive_counts,
            self.expected_positive_counts,
            self.frequency_ratios,
            self.encoded_values,
        ):
            rows.append({
                "category_code": int(code),
                "training_area_pixels": int(area),
                "training_positive_pixels": int(positives),
                "expected_positive_pixels_under_area_null": float(expected),
                "smoothed_frequency_ratio": float(ratio),
                "encoded_value": float(encoded),
            })
        return {
            "factor_index": self.spec.factor_index,
            "normalized_levels": self.spec.normalized_levels,
            "smoothing": self.smoothing,
            "smoothing_formula": "(observed_positives + alpha) / (area_expected_positives + alpha)",
            "log2_clip": self.log2_clip,
            "unknown_category_policy": "neutral_log_frequency_ratio",
            "unknown_encoded_value": self.unknown_encoded_value,
            "mapping_fitted_on": "inner_training_regions_only",
            "zero_positive_training_categories": int(np.count_nonzero(self.positive_counts == 0)),
            "categories_clipped_at_lower_bound": int(np.count_nonzero(
                np.log2(self.frequency_ratios) < -self.log2_clip
            )),
            "categories_clipped_at_upper_bound": int(np.count_nonzero(
                np.log2(self.frequency_ratios) > self.log2_clip
            )),
            "categories": rows,
        }


@dataclass(frozen=True)
class FrozenFoldFeatureTransformer:
    """Continuous min-max plus categorical FR, all frozen per outer fold."""

    normalizer: FrozenMinMaxNormalizer
    frequency_ratio_encoders: tuple[FrozenFrequencyRatioEncoder, ...] = ()

    @property
    def factor_names(self):
        return self.normalizer.factor_names

    def transform(self, features: np.ndarray) -> np.ndarray:
        raw = np.asarray(features)
        if raw.ndim != 2 or raw.shape[1] != len(self.factor_names):
            raise ValueError(
                f"features must have shape [n, {len(self.factor_names)}], got {raw.shape}."
            )
        transformed = self.normalizer.transform(raw)
        for encoder in self.frequency_ratio_encoders:
            index = encoder.spec.factor_index
            transformed[:, index] = encoder.transform(raw[:, index])
        return transformed

    def to_dict(self):
        categorical_names = {
            encoder.spec.factor_name for encoder in self.frequency_ratio_encoders
        }
        return {
            "factor_order": list(self.factor_names),
            "normalizer": self.normalizer.to_dict(),
            "continuous_minmax": {
                name: values
                for name, values in self.normalizer.to_dict().items()
                if name not in categorical_names
            },
            "categorical_frequency_ratio": {
                encoder.spec.factor_name: encoder.to_dict()
                for encoder in self.frequency_ratio_encoders
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping):
        if "normalizer" not in payload:
            raise ValueError(
                "Fold artifact predates reversible transformer serialization; rerun "
                "2_model_train.py --prepare-only or training with the current code."
            )
        normalizer_payload = payload["normalizer"]
        factor_order = payload.get("factor_order", list(normalizer_payload))
        ordered_payload = {
            name: normalizer_payload[name] for name in factor_order
        }
        normalizer = FrozenMinMaxNormalizer.from_dict(ordered_payload)
        encoders = tuple(
            FrozenFrequencyRatioEncoder.from_dict(name, encoder_payload)
            for name, encoder_payload in payload.get(
                "categorical_frequency_ratio", {}
            ).items()
        )
        return cls(normalizer, encoders)


def build_fold_feature_transformer(
    normalizer: FrozenMinMaxNormalizer,
    specs: Sequence[CategoricalFactorSpec],
    category_area_counts: Mapping[str, Mapping[int, int]],
    training_positive_raw: np.ndarray,
    smoothing: float = 0.5,
    log2_clip: float = 4.0,
) -> FrozenFoldFeatureTransformer:
    encoders = []
    for spec in specs:
        encoders.append(
            FrozenFrequencyRatioEncoder.fit(
                spec,
                category_area_counts[spec.factor_name],
                training_positive_raw[:, spec.factor_index],
                smoothing=smoothing,
                log2_clip=log2_clip,
            )
        )
    return FrozenFoldFeatureTransformer(normalizer, tuple(encoders))


def evaluate_kde(kde, features: np.ndarray, chunk_size: int, progress_desc=None) -> np.ndarray:
    densities = np.empty(len(features), dtype=np.float64)
    starts = range(0, len(features), chunk_size)
    for start in track(
        starts,
        total=int(np.ceil(len(features) / max(chunk_size, 1))),
        desc=progress_desc or "",
        unit="chunk",
        disable=(progress_desc is None),
    ):
        stop = min(start + chunk_size, len(features))
        densities[start:stop] = kde.evaluate(features[start:stop].T)
    return densities


def allocate_stratified_counts(total: int, weights: Sequence[float],
                               minimum_per_stratum: int = 0) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    if total < 0 or weights.ndim != 1 or weights.size == 0:
        raise ValueError("Invalid stratified allocation request.")
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Stratum weights must be non-negative with a positive sum.")
    weights = weights / weights.sum()
    counts = np.zeros(weights.size, dtype=np.int64)
    if total >= weights.size * minimum_per_stratum:
        counts += int(minimum_per_stratum)
    remaining = int(total - counts.sum())
    raw = remaining * weights
    counts += np.floor(raw).astype(np.int64)
    for index in np.argsort(-(raw - np.floor(raw)))[: total - int(counts.sum())]:
        counts[index] += 1
    return counts


class FrozenDWSS:
    """Fold-fitted prototype divergence and manuscript DWSS stratification.

    The project's original DWSS implementation defines prototype similarity as
    a joint Gaussian KDE in the complete min-max-scaled factor space. This is
    intentionally distinct from the per-factor frequency-ratio transform used
    as model input.
    """

    def __init__(
        self,
        kde,
        feature_transformer,
        density_scale,
        theta_min,
        breaks,
        stratum_means,
        weights,
        prototype_count,
        prototype_total,
        normalized_prototypes,
        kde_chunk_size,
        training_candidate_divergence,
    ):
        self.kde = kde
        self.feature_transformer = feature_transformer
        self.density_scale = float(density_scale)
        self.theta_min = float(theta_min)
        self.breaks = np.asarray(breaks, dtype=np.float64)
        self.stratum_means = np.asarray(stratum_means, dtype=np.float64)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.prototype_count = int(prototype_count)
        self.prototype_total = int(prototype_total)
        self.normalized_prototypes = np.asarray(
            normalized_prototypes,
            dtype=np.float64,
        )
        self.kde_chunk_size = int(kde_chunk_size)
        self.training_candidate_divergence = np.asarray(
            training_candidate_divergence,
            dtype=np.float64,
        )
        self.selection_candidate_count = int(len(self.training_candidate_divergence))
        self.selection_candidate_divergence = self.training_candidate_divergence.copy()
        self.stratum_statistics_candidate_count = int(
            len(self.training_candidate_divergence)
        )
        self._screening_tree = None
        self._screening_whitener = None
        self._kernel_log_normalization = None

    @classmethod
    def fit(
        cls,
        positive_features: np.ndarray,
        candidate_features: np.ndarray,
        feature_transformer: FrozenFoldFeatureTransformer,
        theta_min: float,
        n_strata: int,
        weight_power: float,
        seed: int,
        max_prototypes: int = 0,
        kde_chunk_size: int = 2048,
    ):
        if not 0.0 <= theta_min <= 1.0:
            raise ValueError("theta_min must be within [0, 1].")
        if n_strata < 1:
            raise ValueError("n_strata must be positive.")
        if not np.isclose(weight_power, 1.0):
            raise ValueError(
                "The manuscript DWSS equation uses unpowered stratum means; "
                "weight_power must equal 1."
            )
        positive_features = np.asarray(positive_features, dtype=np.float64)
        candidate_features = np.asarray(candidate_features, dtype=np.float64)
        if (
            positive_features.ndim != 2
            or candidate_features.ndim != 2
            or positive_features.shape[1] != candidate_features.shape[1]
        ):
            raise ValueError("DWSS positive/candidate feature matrices are misaligned.")
        if len(positive_features) < 2:
            raise ValueError("DWSS requires at least two inner-training positives.")
        if not np.all(np.isfinite(positive_features)) or not np.all(
            np.isfinite(candidate_features)
        ):
            raise ValueError("DWSS inputs must contain finite factor values only.")

        rng = np.random.default_rng(seed)
        prototype_total = len(positive_features)
        if max_prototypes and prototype_total > max_prototypes:
            prototype_indices = np.sort(
                rng.choice(prototype_total, max_prototypes, replace=False)
            )
        else:
            prototype_indices = np.arange(prototype_total, dtype=np.int64)

        normalized_positive = np.asarray(
            feature_transformer.normalizer.transform(positive_features),
            dtype=np.float64,
        )
        normalized_candidates = np.asarray(
            feature_transformer.normalizer.transform(candidate_features),
            dtype=np.float64,
        )
        prototypes = normalized_positive[prototype_indices]
        try:
            kde = gaussian_kde(prototypes.T, bw_method="scott")
        except np.linalg.LinAlgError as error:
            raise RuntimeError(
                "The inner-training landslide prototype has a singular joint "
                "factor covariance. Check constant/duplicate factor rasters."
            ) from error
        prototype_density = evaluate_kde(
            kde,
            prototypes,
            max(1, int(kde_chunk_size)),
            "DWSS KDE 正样本原型",
        )
        candidate_density = evaluate_kde(
            kde,
            normalized_candidates,
            max(1, int(kde_chunk_size)),
            "DWSS KDE 训练候选",
        )
        density_scale = max(
            float(np.max(prototype_density)),
            float(np.max(candidate_density)),
        )
        if not np.isfinite(density_scale) or density_scale <= 0:
            raise RuntimeError(
                "DWSS joint KDE produced an invalid training-fitted density maximum."
            )
        similarity = np.clip(candidate_density / density_scale, 0.0, 1.0)
        divergence = np.clip(1.0 - similarity, 0.0, 1.0)
        eligible = divergence[divergence >= theta_min]
        if len(eligible) < n_strata:
            raise RuntimeError(
                f"Only {len(eligible)} training candidates pass theta_min={theta_min}; "
                f"cannot form {n_strata} strata."
            )
        if np.unique(eligible).size < n_strata:
            raise RuntimeError(
                "Training-region DWSS divergences have too few unique values."
            )

        breaks = np.asarray(
            jenkspy.jenks_breaks(eligible, n_classes=n_strata),
            dtype=np.float64,
        )
        # ``side='left'`` reproduces the manuscript/original implementation:
        # low <= b1, b1 < medium <= b2, and high > b2.
        strata = np.searchsorted(breaks[1:-1], eligible, side="left")
        means = np.asarray(
            [eligible[strata == index].mean() for index in range(n_strata)],
            dtype=np.float64,
        )
        # Manuscript Eq. (1): m_k = M_total * mean(zeta_k) / sum(mean(zeta_j)).
        weights = means / means.sum()
        return cls(
            kde,
            feature_transformer,
            density_scale,
            theta_min,
            breaks,
            means,
            weights,
            len(prototype_indices),
            prototype_total,
            prototypes,
            kde_chunk_size,
            divergence,
        )

    def assign_strata(self, divergence: np.ndarray) -> np.ndarray:
        """Assign eligible divergence values to the frozen natural-break strata."""
        divergence = np.asarray(divergence, dtype=np.float64)
        return np.searchsorted(
            self.breaks[1:-1],
            divergence,
            side="left",
        )

    def allocation_status(self, divergence: np.ndarray, total: int) -> dict:
        """Return strict Eq. (1) targets and currently available candidates."""
        divergence = np.asarray(divergence, dtype=np.float64)
        eligible = divergence >= self.theta_min
        strata = self.assign_strata(divergence[eligible])
        desired = allocate_stratified_counts(total, self.weights, 0)
        available = np.asarray(
            [
                np.count_nonzero(strata == index)
                for index in range(len(self.weights))
            ],
            dtype=np.int64,
        )
        return {
            "desired": desired,
            "available": available,
            "deficit": np.maximum(desired - available, 0),
            "eligible_count": int(np.count_nonzero(eligible)),
        }

    def refresh_stratum_statistics(self, divergence: np.ndarray) -> None:
        """Refine frozen-break stratum means from uniform conditional draws.

        Natural-break boundaries remain fixed after the initial fold-only fit.
        Additional candidates are used only to improve each conditional
        stratum mean and to satisfy the exact manuscript allocation.
        """
        divergence = np.asarray(divergence, dtype=np.float64)
        eligible = divergence[divergence >= self.theta_min]
        strata = self.assign_strata(eligible)
        means = np.asarray(
            [
                eligible[strata == index].mean()
                if np.any(strata == index)
                else np.nan
                for index in range(len(self.weights))
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(means)) or np.any(means <= 0):
            raise RuntimeError(
                "Cannot update DWSS weights because at least one frozen stratum "
                "contains no eligible candidate."
            )
        self.stratum_means = means
        self.weights = means / means.sum()
        self.selection_candidate_divergence = divergence.copy()
        self.selection_candidate_count = int(len(divergence))
        self.stratum_statistics_candidate_count = int(len(divergence))

    def _prepare_screening_index(self) -> None:
        """Build the exact KDE-metric neighbour index used for safe pruning."""
        if self._screening_tree is not None:
            return
        inverse_covariance = np.asarray(self.kde.inv_cov, dtype=np.float64)
        whitener = np.linalg.cholesky(inverse_covariance)
        whitened_prototypes = self.normalized_prototypes @ whitener
        sign, log_determinant = np.linalg.slogdet(
            np.asarray(self.kde.covariance, dtype=np.float64)
        )
        if sign <= 0 or not np.isfinite(log_determinant):
            raise RuntimeError("DWSS KDE covariance is not positive definite.")
        dimension = self.normalized_prototypes.shape[1]
        self._screening_whitener = whitener
        self._screening_tree = cKDTree(whitened_prototypes)
        self._kernel_log_normalization = float(
            -0.5 * (dimension * np.log(2.0 * np.pi) + log_determinant)
        )

    def screen_for_strata(
        self,
        features: np.ndarray,
        highest_needed_stratum: int,
        neighbors: int = 64,
        chunk_size: int = 20000,
    ) -> np.ndarray:
        """Safely reject candidates that cannot enter a deficient lower stratum.

        For the k nearest prototypes, all unseen Gaussian kernels are bounded
        by the k-th kernel. The resulting density upper bound can only reject a
        point when its exact KDE density is too small to reach the requested
        stratum. Candidates that pass this screen are still evaluated with the
        original exact SciPy Gaussian KDE.
        """
        features = np.asarray(features, dtype=np.float64)
        n_strata = len(self.weights)
        if not 0 <= highest_needed_stratum < n_strata:
            raise ValueError("highest_needed_stratum is outside the DWSS strata.")
        if highest_needed_stratum == n_strata - 1:
            return np.ones(len(features), dtype=bool)
        if neighbors < 1 or chunk_size < 1:
            raise ValueError("DWSS screening neighbors/chunk size must be positive.")

        upper_break = float(self.breaks[highest_needed_stratum + 1])
        required_density = self.density_scale * max(0.0, 1.0 - upper_break)
        if required_density <= 0:
            return np.ones(len(features), dtype=bool)

        self._prepare_screening_index()
        normalized = np.asarray(
            self.feature_transformer.normalizer.transform(features),
            dtype=np.float64,
        )
        whitened = normalized @ self._screening_whitener
        prototype_count = len(self.normalized_prototypes)
        neighbor_count = min(int(neighbors), prototype_count)
        log_required = np.log(required_density)
        keep = np.zeros(len(features), dtype=bool)

        for start in range(0, len(features), int(chunk_size)):
            stop = min(start + int(chunk_size), len(features))
            distances, _ = self._screening_tree.query(
                whitened[start:stop],
                k=neighbor_count,
                workers=-1,
            )
            if neighbor_count == 1:
                distances = distances[:, None]
            squared = np.square(np.asarray(distances, dtype=np.float64))
            kernels = np.exp(-0.5 * squared)
            upper_sum = kernels.sum(axis=1)
            if neighbor_count < prototype_count:
                upper_sum += (
                    prototype_count - neighbor_count
                ) * kernels[:, -1]
            positive = upper_sum > 0
            log_upper = np.full(len(upper_sum), -np.inf, dtype=np.float64)
            log_upper[positive] = (
                self._kernel_log_normalization
                + np.log(upper_sum[positive])
                - np.log(prototype_count)
            )
            # A small tolerance prevents a floating-point boundary rejection.
            keep[start:stop] = log_upper >= log_required - 1e-12
        return keep

    def divergence(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != len(
            self.feature_transformer.factor_names
        ):
            raise ValueError("DWSS feature matrix has an unexpected shape.")
        normalized = self.feature_transformer.normalizer.transform(features)
        density = evaluate_kde(
            self.kde,
            np.asarray(normalized, dtype=np.float64),
            self.kde_chunk_size,
        )
        return np.clip(
            1.0 - density / self.density_scale,
            0.0,
            1.0,
        )

    def select(
        self,
        points: np.ndarray,
        features: np.ndarray,
        total: int,
        seed: int,
        minimum_per_stratum: int = 0,
        precomputed_divergence=None,
    ):
        if minimum_per_stratum:
            raise ValueError(
                "The strict manuscript allocation has no per-stratum minimum; "
                "minimum_per_stratum must be 0."
            )
        divergence = (
            self.divergence(features)
            if precomputed_divergence is None
            else np.asarray(precomputed_divergence, dtype=np.float64)
        )
        if len(divergence) != len(points):
            raise ValueError("Precomputed DWSS divergence does not match the candidate pool.")
        eligible_indices = np.flatnonzero(divergence >= self.theta_min)
        if len(eligible_indices) < total:
            raise RuntimeError(
                f"DWSS frozen training rule yields {len(eligible_indices)} eligible "
                f"candidates but {total} are required. Increase --candidate-multiplier "
                "without changing any test-derived DWSS parameter."
            )
        eligible_divergence = divergence[eligible_indices]
        strata = self.assign_strata(eligible_divergence)
        desired = allocate_stratified_counts(total, self.weights, 0)
        available = np.asarray(
            [
                np.count_nonzero(strata == index)
                for index in range(len(self.weights))
            ],
            dtype=np.int64,
        )
        shortage = np.maximum(desired - available, 0)
        if np.any(shortage):
            raise RuntimeError(
                "DWSS cannot satisfy the manuscript Eq. (1) allocation without "
                "cross-stratum substitution. "
                f"targets={desired.tolist()}, available={available.tolist()}, "
                f"deficits={shortage.tolist()}. Increase the adaptive candidate "
                "budget; the implementation will not silently redistribute quotas."
            )
        rng = np.random.default_rng(seed)

        selected = []
        for stratum_index, wanted in enumerate(desired):
            members = eligible_indices[strata == stratum_index]
            rng.shuffle(members)
            selected.extend(members[:int(wanted)].tolist())

        selected = np.asarray(selected, dtype=np.int64)
        rng.shuffle(selected)
        selected_divergence = divergence[selected]
        selected_strata = self.assign_strata(selected_divergence)
        diagnostics = {
            "candidate_pool": int(len(points)),
            "eligible_candidates": int(len(eligible_indices)),
            "eligible_fraction": float(len(eligible_indices) / max(len(points), 1)),
            "selected": int(len(selected)),
            "selected_divergence_mean": float(np.mean(selected_divergence)),
            "selected_divergence_min": float(np.min(selected_divergence)),
            "selected_divergence_max": float(np.max(selected_divergence)),
            "stratum_target_counts": desired.tolist(),
            "stratum_available_counts": available.tolist(),
            "stratum_selected_counts": [
                int(np.count_nonzero(selected_strata == index))
                for index in range(len(self.weights))
            ],
            "strict_manuscript_allocation_satisfied": True,
            "cross_stratum_quota_redistribution": False,
        }
        return points[selected], diagnostics

    def to_dict(self):
        quantile_levels = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
        quantiles = np.quantile(self.training_candidate_divergence, quantile_levels)
        selection_eligible = self.selection_candidate_divergence[
            self.selection_candidate_divergence >= self.theta_min
        ]
        selection_strata = self.assign_strata(selection_eligible)
        stratum_counts = np.asarray(
            [
                np.count_nonzero(selection_strata == index)
                for index in range(len(self.weights))
            ],
            dtype=np.int64,
        )
        stratum_standard_errors = []
        for index, count in enumerate(stratum_counts):
            values = selection_eligible[selection_strata == index]
            standard_error = (
                float(np.std(values, ddof=1) / np.sqrt(count))
                if count > 1 else None
            )
            stratum_standard_errors.append(standard_error)
        return {
            "theta_min": self.theta_min,
            "n_strata": int(len(self.weights)),
            "jenks_breaks_fitted_on_inner_train": self.breaks.tolist(),
            "stratum_mean_divergence_fitted_on_inner_train": self.stratum_means.tolist(),
            "stratum_mean_divergence_standard_error": stratum_standard_errors,
            "stratum_statistics_sample_counts": stratum_counts.tolist(),
            "stratum_weights_fitted_on_inner_train": self.weights.tolist(),
            "stratum_weight_formula": "mean_zeta_k / sum_j(mean_zeta_j)",
            "minimum_per_stratum": 0,
            "prototype_similarity_method": (
                "joint multivariate Gaussian KDE in the complete fold-fitted "
                "min-max-scaled factor space"
            ),
            "kde_bandwidth_method": "scott",
            "similarity_formula": "joint_kde_density / training_fitted_max_density",
            "divergence_formula": "1 - normalized_joint_kde_density",
            "density_scale_fitted_on_inner_train": self.density_scale,
            "model_input_frequency_ratio_transform_used_for_dwss": False,
            "factor_order": list(self.feature_transformer.factor_names),
            "prototype_count_used": self.prototype_count,
            "prototype_count_available": self.prototype_total,
            "kde_chunk_size": self.kde_chunk_size,
            "training_candidate_count": int(len(self.training_candidate_divergence)),
            "selection_candidate_count": self.selection_candidate_count,
            "stratum_statistics_candidate_count": (
                self.stratum_statistics_candidate_count
            ),
            "training_candidate_eligible_fraction": float(np.mean(
                self.training_candidate_divergence >= self.theta_min
            )),
            "training_candidate_divergence_quantiles": {
                str(level): float(value)
                for level, value in zip(quantile_levels, quantiles)
            },
        }


def choose_random_points(points: np.ndarray, total: int, seed: int) -> np.ndarray:
    if len(points) < total:
        raise RuntimeError(f"Random sampling needs {total} candidates but only {len(points)} exist.")
    rng = np.random.default_rng(seed)
    return points[rng.choice(len(points), size=total, replace=False)]


def seed_numpy_worker(_worker_id):
    """Pickle-safe DataLoader worker seeding for Linux and Windows."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)


def region_class_balanced_weights(
    positive_points: np.ndarray,
    negative_points: np.ndarray,
    allowed_regions: Sequence[int],
    balance_power: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """Temper regional dominance without changing the designed class prior.

    Region balancing is performed *within each class*.  The weights of all
    landslide samples sum to the number of landslide samples and the weights of
    all non-landslide samples sum to the number of non-landslide samples.  This
    matters for the manuscript's 1:1 design: a single global normalization
    silently gave the more spatially dispersed class a larger total loss weight
    and shifted every model's decision scores toward that class.
    """
    if not 0 <= balance_power <= 1:
        raise ValueError("region balance power must be in [0, 1].")
    allowed_regions = sorted(set(map(int, allowed_regions)))
    groups = (
        (1, np.asarray(positive_points)),
        (0, np.asarray(negative_points)),
    )
    normalized_weights = []
    rows = []
    for label, points in groups:
        weights = np.zeros(len(points), dtype=np.float64)
        class_rows = []
        for region_id in allowed_regions:
            selected = points[:, 2] == region_id
            count = int(np.count_nonzero(selected))
            if count:
                weights[selected] = count ** (-balance_power)
            row = {
                "region_id": region_id,
                "class_label": label,
                "sample_count": count,
                "raw_per_sample_weight": count ** (-balance_power) if count else 0.0,
            }
            rows.append(row)
            class_rows.append(row)
        if np.any(weights == 0):
            raise RuntimeError(
                "Region-balanced weighting found samples outside allowed regions."
            )
        # Preserve the original total contribution of this class.  With the
        # required 1:1 samples this makes the effective positive/negative loss
        # prior exactly 1:1, irrespective of their different regional layouts.
        class_scale = len(points) / weights.sum()
        weights *= class_scale
        normalized_weights.append(weights)
        for row in class_rows:
            row["class_normalization_scale"] = float(class_scale)
            row["normalized_per_sample_weight"] = (
                row["raw_per_sample_weight"] * class_scale
            )
            row["normalized_group_total_weight"] = (
                row["normalized_per_sample_weight"] * row["sample_count"]
            )
            row["normalized_class_total_weight"] = float(len(points))
    combined = np.concatenate(normalized_weights)
    class_totals = {
        str(label): float(weights.sum())
        for (label, _points), weights in zip(groups, normalized_weights)
    }
    return combined.astype(np.float32), {
        "policy": "class_prior_preserving_inverse_region_frequency_power",
        "balance_power": float(balance_power),
        "power_interpretation": {
            "0": "uniform_samples",
            "0.5": "square_root_tempering",
            "1": "equal_group_total_weight",
        },
        "normalization": "within_class_mean_weight_equals_one",
        "class_total_weights": class_totals,
        "effective_negative_to_positive_weight_ratio": float(
            class_totals["0"] / class_totals["1"]
        ),
        "class_prior_preserved": bool(
            np.isclose(class_totals["0"] / class_totals["1"], len(negative_points) / len(positive_points))
        ),
        "groups": rows,
    }


class SparseRasterDataset(Dataset):
    """Read raster context while supervising only audited sample coordinates.

    Training can expose every real sample through several shifted crop grids.  No
    coordinate or label is synthesized: a context view only changes where the
    same point falls inside the crop.  Dense crop groups can also be split into
    bounded supervision chunks so a single exceptionally dense tile cannot
    dominate one optimizer step.
    """

    def __init__(
        self,
        factor_paths: Sequence[str],
        regions_path: str,
        points: np.ndarray,
        labels: np.ndarray,
        allowed_regions: Sequence[int],
        normalizer: FrozenMinMaxNormalizer | FrozenFoldFeatureTransformer,
        sample_weights: np.ndarray | None = None,
        crop_size: int = 512,
        train: bool = False,
        augmentation_mode: str = "aspect_safe_d4",
        aspect_period: float = 1.0,
        training_context_views: int = 1,
        max_supervised_points_per_training_tile: int = 0,
    ):
        self.factor_paths = tuple(factor_paths)
        self.regions_path = str(regions_path)
        self.points = np.asarray(points, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.sample_weights = (
            np.ones(len(self.labels), dtype=np.float32)
            if sample_weights is None
            else np.asarray(sample_weights, dtype=np.float32)
        )
        self.allowed_regions = np.asarray(sorted(set(map(int, allowed_regions))), dtype=np.int64)
        self.normalizer = normalizer
        self.crop_size = int(crop_size)
        self.train = bool(train)
        self.augmentation_mode = str(augmentation_mode).strip().lower()
        self.aspect_period = float(aspect_period)
        requested_context_views = int(training_context_views)
        requested_supervision_limit = int(max_supervised_points_per_training_tile)
        self._sources = None
        self._region_source = None
        self._source_pid = None

        if self.points.ndim != 2 or self.points.shape[1] < 2:
            raise ValueError("points must be [n, >=2].")
        if (
            len(self.points) != len(self.labels)
            or len(self.points) != len(self.sample_weights)
            or len(self.points) == 0
        ):
            raise ValueError("SparseRasterDataset needs non-empty aligned points and labels.")
        if np.any(~np.isin(self.labels, [0, 1])):
            raise ValueError("Labels must be binary 0/1.")
        if self.augmentation_mode not in {"none", "aspect_safe_d4"}:
            raise ValueError(
                "augmentation_mode must be 'none' or 'aspect_safe_d4'."
            )
        if not np.isfinite(self.aspect_period) or self.aspect_period <= 0:
            raise ValueError("aspect_period must be finite and positive.")
        if requested_context_views not in {1, 2, 4}:
            raise ValueError("training_context_views must be one of 1, 2, or 4.")
        if requested_supervision_limit < 0:
            raise ValueError(
                "max_supervised_points_per_training_tile cannot be negative."
            )
        self.training_context_views = requested_context_views if self.train else 1
        self.max_supervised_points_per_training_tile = (
            requested_supervision_limit if self.train else 0
        )

        factor_names = tuple(map(str, self.normalizer.factor_names))
        aspect_matches = [
            index for index, name in enumerate(factor_names)
            if name.strip().lower() == "aspect"
        ]
        if len(aspect_matches) > 1:
            raise ValueError("Factor order contains more than one Aspect channel.")
        self.aspect_index = aspect_matches[0] if aspect_matches else None

        coordinate_labels = {}
        for point, label, weight in zip(self.points, self.labels, self.sample_weights):
            key = (int(point[0]), int(point[1]))
            previous = coordinate_labels.setdefault(key, (int(label), float(weight)))
            if previous[0] != int(label):
                raise ValueError(f"Conflicting labels for raster cell {key}.")
        unique = sorted(
            (row, col, label, weight)
            for (row, col), (label, weight) in coordinate_labels.items()
        )
        self.points = np.asarray([[row, col] for row, col, _, _ in unique], dtype=np.int64)
        self.labels = np.asarray([label for _, _, label, _ in unique], dtype=np.int64)
        self.sample_weights = np.asarray(
            [weight for _, _, _, weight in unique], dtype=np.float32
        )

        self.unique_supervised_points = int(len(self.points))
        self.unique_class_weight_totals = {
            label: float(self.sample_weights[self.labels == label].sum())
            for label in (0, 1)
        }
        half_crop = self.crop_size // 2
        context_offsets = {
            1: ((0, 0),),
            2: ((0, 0), (half_crop, half_crop)),
            4: ((0, 0), (0, half_crop), (half_crop, 0), (half_crop, half_crop)),
        }
        self.context_offsets = context_offsets[self.training_context_views]

        tile_map = {}
        for view_index, (row_shift, col_shift) in enumerate(self.context_offsets):
            for point_index, (row, col) in enumerate(self.points):
                row_off = int(
                    ((int(row) - row_shift) // self.crop_size) * self.crop_size
                    + row_shift
                )
                col_off = int(
                    ((int(col) - col_shift) // self.crop_size) * self.crop_size
                    + col_shift
                )
                tile_map.setdefault((view_index, row_off, col_off), []).append(
                    point_index
                )

        self.tiles = []
        unsplit_context_tiles = len(tile_map)
        for (_view_index, row_off, col_off), indices in sorted(tile_map.items()):
            indices = np.asarray(indices, dtype=np.int64)
            for supervision_indices in self._split_supervision_indices(
                indices,
                self.labels,
                self.max_supervised_points_per_training_tile,
            ):
                self.tiles.append(((row_off, col_off), supervision_indices))

        self.class_weight_totals = {
            label: total * self.training_context_views
            for label, total in self.unique_class_weight_totals.items()
        }
        points_per_item = [len(indices) for _origin, indices in self.tiles]
        self.supervision_audit = {
            "policy": (
                "label_preserving_shifted_context_views_with_bounded_supervision_chunks"
                if self.train
                else "single_fixed_context_without_training_augmentation"
            ),
            "training_only": bool(self.train),
            "unique_real_sample_count": self.unique_supervised_points,
            "unique_real_class_counts": {
                str(label): int(np.count_nonzero(self.labels == label))
                for label in (0, 1)
            },
            "unique_class_weight_totals": {
                str(label): total
                for label, total in self.unique_class_weight_totals.items()
            },
            "context_view_count": self.training_context_views,
            "context_offsets_pixels": [list(offset) for offset in self.context_offsets],
            "supervision_instance_count": int(
                self.unique_supervised_points * self.training_context_views
            ),
            "class_weight_totals_across_context_views": {
                str(label): total for label, total in self.class_weight_totals.items()
            },
            "unique_crop_context_count_before_supervision_chunking": int(
                unsplit_context_tiles
            ),
            "optimizer_item_count_after_supervision_chunking": int(len(self.tiles)),
            "max_supervised_points_per_training_tile": int(
                self.max_supervised_points_per_training_tile
            ),
            "observed_max_supervised_points_per_optimizer_item": int(
                max(points_per_item, default=0)
            ),
            "invented_coordinates_or_labels": False,
        }

    @staticmethod
    def _split_supervision_indices(indices, labels, limit):
        """Split a dense context into class-interleaved, bounded point groups."""
        indices = np.asarray(indices, dtype=np.int64)
        if limit <= 0 or len(indices) <= limit:
            return (indices,)

        ranked_indices = []
        fractional_ranks = []
        class_ties = []
        for label in (0, 1):
            selected = indices[labels[indices] == label]
            if not len(selected):
                continue
            ranked_indices.append(selected)
            fractional_ranks.append(
                (np.arange(len(selected), dtype=np.float64) + 0.5) / len(selected)
            )
            class_ties.append(np.full(len(selected), label, dtype=np.int8))
        ordered_indices = np.concatenate(ranked_indices)
        order = np.lexsort((np.concatenate(class_ties), np.concatenate(fractional_ranks)))
        ordered_indices = ordered_indices[order]
        return tuple(
            ordered_indices[start:start + limit]
            for start in range(0, len(ordered_indices), limit)
        )

    def __len__(self):
        return len(self.tiles)

    def _ensure_sources(self):
        pid = os.getpid()
        if self._sources is None or self._source_pid != pid:
            self.close()
            self._sources = [rasterio.open(path) for path in self.factor_paths]
            self._region_source = rasterio.open(self.regions_path)
            self._source_pid = pid

    def close(self):
        for source in getattr(self, "_sources", None) or []:
            source.close()
        region_source = getattr(self, "_region_source", None)
        if region_source is not None:
            region_source.close()
        self._sources = None
        self._region_source = None
        self._source_pid = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_sources"] = None
        state["_region_source"] = None
        state["_source_pid"] = None
        return state

    def __del__(self):
        self.close()

    def _adjust_aspect(self, raw_stack, valid_mask, operation, rotations=0):
        """Keep the circular Aspect channel consistent with image geometry."""
        if self.aspect_index is None:
            return
        values = raw_stack[self.aspect_index]
        selected = values[valid_mask]
        period = self.aspect_period
        if operation == "horizontal":
            selected = -selected
        elif operation == "vertical":
            selected = 0.5 * period - selected
        elif operation == "rotation":
            selected = selected - 0.25 * period * int(rotations)
        else:
            raise ValueError(f"Unknown Aspect augmentation operation: {operation}")
        values[valid_mask] = np.mod(selected, period)

    def _augment_raw_tile(
        self,
        raw_stack,
        label_raster,
        combined_mask,
        weight_raster,
        region_ids,
    ):
        """Apply a D4 transform while respecting Aspect's circular semantics."""
        if not self.train or self.augmentation_mode == "none":
            return raw_stack, label_raster, combined_mask, weight_raster, region_ids

        if torch.rand(1).item() < 0.5:
            raw_stack = np.flip(raw_stack, axis=2)
            label_raster = np.flip(label_raster, axis=1)
            combined_mask = np.flip(combined_mask, axis=1)
            weight_raster = np.flip(weight_raster, axis=1)
            region_ids = np.flip(region_ids, axis=1)
            self._adjust_aspect(raw_stack, combined_mask, "horizontal")
        if torch.rand(1).item() < 0.5:
            raw_stack = np.flip(raw_stack, axis=1)
            label_raster = np.flip(label_raster, axis=0)
            combined_mask = np.flip(combined_mask, axis=0)
            weight_raster = np.flip(weight_raster, axis=0)
            region_ids = np.flip(region_ids, axis=0)
            self._adjust_aspect(raw_stack, combined_mask, "vertical")
        rotations = int(torch.randint(0, 4, (1,)).item())
        if rotations:
            raw_stack = np.rot90(raw_stack, rotations, axes=(1, 2))
            label_raster = np.rot90(label_raster, rotations, axes=(0, 1))
            combined_mask = np.rot90(combined_mask, rotations, axes=(0, 1))
            weight_raster = np.rot90(weight_raster, rotations, axes=(0, 1))
            region_ids = np.rot90(region_ids, rotations, axes=(0, 1))
            self._adjust_aspect(
                raw_stack,
                combined_mask,
                "rotation",
                rotations=rotations,
            )
        return tuple(map(np.ascontiguousarray, (
            raw_stack,
            label_raster,
            combined_mask,
            weight_raster,
            region_ids,
        )))

    def __getitem__(self, index):
        self._ensure_sources()
        (row_off, col_off), point_indices = self.tiles[index]
        window = Window(col_off, row_off, self.crop_size, self.crop_size)

        region_data = self._region_source.read(
            1,
            window=window,
            boundless=True,
            fill_value=self._region_source.nodata or 0,
        )
        region_valid = _valid_values(region_data, self._region_source.nodata)
        region_ids = _integer_regions(region_data, region_valid)
        allowed_mask = region_valid & np.isin(region_ids, self.allowed_regions)

        raw_factor_arrays = []
        combined_mask = allowed_mask.copy()
        for source in self._sources:
            data = source.read(
                1,
                window=window,
                boundless=True,
                fill_value=source.nodata if source.nodata is not None else np.nan,
            ).astype(np.float32, copy=False)
            valid = _valid_values(data, source.nodata)
            safe_data = data.copy()
            safe_data[~valid] = 0.0
            raw_factor_arrays.append(safe_data)
            combined_mask &= valid

        label_raster = np.full((self.crop_size, self.crop_size), -1, dtype=np.int64)
        weight_raster = np.zeros((self.crop_size, self.crop_size), dtype=np.float32)
        selected_points = self.points[point_indices]
        local_rows = selected_points[:, 0] - row_off
        local_cols = selected_points[:, 1] - col_off
        label_raster[local_rows, local_cols] = self.labels[point_indices]
        weight_raster[local_rows, local_cols] = self.sample_weights[point_indices]

        raw_stack = np.stack(raw_factor_arrays, axis=0)
        (
            raw_stack,
            label_raster,
            combined_mask,
            weight_raster,
            region_ids,
        ) = self._augment_raw_tile(
            raw_stack,
            label_raster,
            combined_mask,
            weight_raster,
            region_ids,
        )
        flat_raw = raw_stack.reshape(len(raw_factor_arrays), -1).T
        transformed = self.normalizer.transform(flat_raw).T.reshape(raw_stack.shape)
        transformed[:, ~combined_mask] = 0.0

        factors = torch.from_numpy(transformed.astype(np.float32, copy=False))
        labels = torch.from_numpy(label_raster)
        mask = torch.from_numpy(combined_mask)
        weights = torch.from_numpy(weight_raster)
        groups = torch.from_numpy(region_ids.astype(np.int64, copy=False))
        return factors, labels, mask, weights, groups


class SupervisionMassBatchSampler:
    """Form deterministic training batches with comparable supervision mass.

    The sampler uses longest-processing-time assignment: dense supervision
    items are distributed first to the currently lightest non-full batch.  It
    changes neither item membership nor sampling probability, but prevents
    global mass normalization and gradient clipping from alternating between
    nearly empty and exceptionally heavy optimizer steps.
    """

    def __init__(self, dataset, batch_size, seed):
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive.")
        self.item_weight_mass = np.asarray([
            float(dataset.sample_weights[indices].sum())
            for _origin, indices in dataset.tiles
        ], dtype=np.float64)
        if not len(self.item_weight_mass) or np.any(self.item_weight_mass <= 0):
            raise ValueError("Training items must have positive supervision mass.")
        self.batch_count = int(np.ceil(len(self.item_weight_mass) / self.batch_size))

    def __len__(self):
        return self.batch_count

    def _build_batches(self, epoch):
        rng = np.random.default_rng(self.seed + int(epoch))
        random_ties = rng.random(len(self.item_weight_mass))
        item_order = np.lexsort((random_ties, -self.item_weight_mass))
        batches = [[] for _ in range(self.batch_count)]
        batch_mass = np.zeros(self.batch_count, dtype=np.float64)
        batch_ties = rng.random(self.batch_count)
        for item_index in item_order:
            eligible = np.flatnonzero(
                np.fromiter(
                    (len(batch) < self.batch_size for batch in batches),
                    dtype=bool,
                    count=self.batch_count,
                )
            )
            destination = eligible[np.lexsort((
                batch_ties[eligible],
                batch_mass[eligible],
            ))[0]]
            batches[int(destination)].append(int(item_index))
            batch_mass[destination] += self.item_weight_mass[item_index]

        for batch in batches:
            rng.shuffle(batch)
        rng.shuffle(batches)
        return batches

    def __iter__(self):
        batches = self._build_batches(self.epoch)
        self.epoch += 1
        yield from batches

    def audit(self):
        batches = self._build_batches(0)
        masses = np.asarray([
            self.item_weight_mass[np.asarray(batch, dtype=np.int64)].sum()
            for batch in batches
        ])
        return {
            "policy": "supervision_weight_mass_balanced_without_resampling",
            "batch_count": int(len(batches)),
            "batch_size_limit": self.batch_size,
            "item_count": int(len(self.item_weight_mass)),
            "batch_weight_mass_min": float(masses.min()),
            "batch_weight_mass_mean": float(masses.mean()),
            "batch_weight_mass_max": float(masses.max()),
            "batch_weight_mass_coefficient_of_variation": float(
                masses.std(ddof=0) / masses.mean()
            ),
            "batch_weight_mass_max_to_mean_ratio": float(
                masses.max() / masses.mean()
            ),
            "sampling_with_replacement": False,
            "every_optimizer_item_once_per_epoch": True,
        }


def make_sparse_loader(
    factor_paths: Sequence[str],
    regions_path: str,
    positive_points: np.ndarray,
    negative_points: np.ndarray,
    allowed_regions: Sequence[int],
    normalizer: FrozenMinMaxNormalizer | FrozenFoldFeatureTransformer,
    crop_size: int,
    batch_size: int,
    num_workers: int,
    train: bool,
    seed: int,
    region_balance: bool = False,
    region_balance_power: float = 1.0,
    augmentation_mode: str = "aspect_safe_d4",
    aspect_period: float = 1.0,
    training_context_views: int = 1,
    max_supervised_points_per_training_tile: int = 0,
):
    allowed_set = set(map(int, allowed_regions))
    for name, sample_points in (
        ("positive", positive_points),
        ("negative", negative_points),
    ):
        if sample_points.ndim != 2 or sample_points.shape[1] < 3:
            raise ValueError(f"{name} points must include row, col, and region ID.")
        unexpected = set(map(int, np.unique(sample_points[:, 2]))) - allowed_set
        if unexpected:
            raise ValueError(
                f"{name} samples from disallowed regions would cross split boundaries: "
                f"{sorted(unexpected)}"
            )
    points = np.concatenate((positive_points[:, :2], negative_points[:, :2]), axis=0)
    labels = np.concatenate(
        (np.ones(len(positive_points), dtype=np.int64), np.zeros(len(negative_points), dtype=np.int64))
    )
    if train and region_balance:
        sample_weights, weighting_audit = region_class_balanced_weights(
            positive_points,
            negative_points,
            allowed_regions,
            balance_power=region_balance_power,
        )
    else:
        sample_weights = np.ones(len(points), dtype=np.float32)
        weighting_audit = {
            "policy": "uniform_sample_weight",
            "normalization": "all_weights_equal_one",
        }
    dataset = SparseRasterDataset(
        factor_paths,
        regions_path,
        points,
        labels,
        allowed_regions,
        normalizer,
        sample_weights=sample_weights,
        crop_size=crop_size,
        train=train,
        augmentation_mode=augmentation_mode,
        aspect_period=aspect_period,
        training_context_views=training_context_views,
        max_supervised_points_per_training_tile=(
            max_supervised_points_per_training_tile
        ),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)

    if train:
        batch_sampler = SupervisionMassBatchSampler(dataset, batch_size, seed)
        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            generator=generator,
            worker_init_fn=seed_numpy_worker,
        )
        batch_mass_audit = batch_sampler.audit()
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            generator=generator,
            worker_init_fn=seed_numpy_worker,
        )
        batch_mass_audit = {
            "policy": "fixed_sequential_evaluation_batches",
            "batch_count": int(len(loader)),
            "sampling_with_replacement": False,
        }
    loader.region_weighting_audit = weighting_audit
    loader.batch_mass_audit = batch_mass_audit
    dataset.supervision_audit["optimizer_batching"] = batch_mass_audit
    loader.supervision_audit = dataset.supervision_audit
    return loader
