"""Build continuous, terrain-derived macro-regions without using labels.

The regionalization is deliberately separated from the landslide inventory and
from model outputs.  Terrain variables are summarized on a coarse grid, scaled
with robust percentiles, and clustered with Ward linkage under a four-neighbour
spatial connectivity constraint.  Every resulting region is therefore a
single spatially connected unit on the regionalization grid.

These products are *terrain-derived physiographic macro-regions*.  They are not
tectonic units or standard watersheds unless an authoritative external data set
is supplied instead to the regional hold-out workflow.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import warnings
import xml.etree.ElementTree as ET
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
from scipy import ndimage, sparse
from sklearn.cluster import AgglomerativeClustering

from .progress import configure_progress, console, timed_task, track, window_count


DEFAULT_TERRAIN_FACTORS = (
    "DEM",
    "Slope",
    "Relief",
    "Roughness",
    "TPI",
    "TWI",
)
FOUR_NEIGHBOURS = ndimage.generate_binary_structure(2, 1)


def normalize_path(value: str) -> str:
    value = os.path.expanduser(str(value).strip())
    if os.name != "nt":
        value = value.replace("\\", "/")
    return value


def read_xml_params(xml_path: str) -> dict[str, str]:
    root = ET.parse(xml_path).getroot()
    params: dict[str, str] = {}
    for param in root.findall("param"):
        name = param.find("name")
        value = param.find("value")
        if name is not None and value is not None and name.text and value.text is not None:
            params[name.text.strip()] = normalize_path(value.text)
    return params


def parse_bool(value, default: bool = False) -> bool:
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


def parse_name_list(value) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_TERRAIN_FACTORS
    if isinstance(value, (list, tuple)):
        names = [str(item).strip() for item in value]
    else:
        names = [item.strip().strip("'\"") for item in str(value).strip("[]").split(",")]
    names = [name for name in names if name]
    if not names:
        raise ValueError("macro_region_factors must contain at least one raster name.")
    if len({name.casefold() for name in names}) != len(names):
        raise ValueError("macro_region_factors contains duplicate names.")
    return tuple(names)


def resolve_output_template(template: str, region_count: int) -> str:
    return normalize_path(template).replace(
        "{macro_region_count}", str(region_count)
    ).replace("{count}", str(region_count))


@dataclass(frozen=True)
class TerrainRegionConfig:
    factors_dir: str
    output_path: str
    region_count: int = 5
    factor_names: tuple[str, ...] = DEFAULT_TERRAIN_FACTORS
    downsample: int = 64
    spatial_weight: float = 1.5
    minimum_region_fraction: float = 0.03
    maximum_excluded_coarse_fraction: float = 0.01
    maximum_removed_full_fraction: float = 0.005
    write_plot: bool = True

    @classmethod
    def from_params(cls, params: Mapping[str, str]) -> "TerrainRegionConfig":
        count = int(params.get("macro_region_count", 5))
        factors_dir = normalize_path(params["input_factors_dir"])
        default_output = str(
            Path(factors_dir).parent / "macro_regions" / f"terrain_macro_regions_k{count}.tif"
        )
        output_template = params.get("macro_region_output", default_output)
        return cls(
            factors_dir=factors_dir,
            output_path=resolve_output_template(output_template, count),
            region_count=count,
            factor_names=parse_name_list(params.get("macro_region_factors")),
            downsample=int(params.get("macro_region_downsample", 64)),
            spatial_weight=float(params.get("macro_region_spatial_weight", 1.5)),
            minimum_region_fraction=float(params.get("macro_region_minimum_fraction", 0.03)),
            maximum_excluded_coarse_fraction=float(
                params.get("macro_region_maximum_excluded_fraction", 0.01)
            ),
            maximum_removed_full_fraction=float(
                params.get("macro_region_maximum_disconnected_fraction", 0.005)
            ),
            write_plot=parse_bool(params.get("macro_region_write_plot"), True),
        )

    def validate(self) -> None:
        if self.region_count != 5:
            raise ValueError("The manuscript protocol uses only K=5 macro-regions.")
        if self.downsample < 1:
            raise ValueError("macro_region_downsample must be at least 1.")
        if self.spatial_weight < 0:
            raise ValueError("macro_region_spatial_weight cannot be negative.")
        for name, value in (
            ("macro_region_minimum_fraction", self.minimum_region_fraction),
            ("macro_region_maximum_excluded_fraction", self.maximum_excluded_coarse_fraction),
            (
                "macro_region_maximum_disconnected_fraction",
                self.maximum_removed_full_fraction,
            ),
        ):
            if not 0 <= value < 1:
                raise ValueError(f"{name} must be in [0, 1).")


def config_from_xml(xml_path: str) -> TerrainRegionConfig:
    return TerrainRegionConfig.from_params(read_xml_params(xml_path))


def _iter_windows(height: int, width: int, size: int = 1024) -> Iterable[Window]:
    for row_off in range(0, height, size):
        for col_off in range(0, width, size):
            yield Window(
                col_off,
                row_off,
                min(size, width - col_off),
                min(size, height - row_off),
            )


def _same_transform(left, right, atol: float = 1e-8) -> bool:
    return bool(np.allclose(tuple(left), tuple(right), rtol=0.0, atol=atol))


def resolve_factor_paths(factors_dir: str, factor_names: Sequence[str]) -> list[str]:
    directory = Path(factors_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Factor directory does not exist: {factors_dir}")
    available: dict[str, list[Path]] = {}
    for path in directory.iterdir():
        if path.is_file() and path.suffix.casefold() in {".tif", ".tiff"}:
            available.setdefault(path.stem.casefold(), []).append(path)

    resolved = []
    missing = []
    for name in factor_names:
        matches = available.get(Path(name).stem.casefold(), [])
        if len(matches) == 1:
            resolved.append(str(matches[0]))
        elif not matches:
            missing.append(name)
        else:
            raise ValueError(f"Multiple rasters match terrain factor {name!r}: {matches}")
    if missing:
        raise FileNotFoundError(
            f"Terrain factor rasters not found in {factors_dir}: {', '.join(missing)}"
        )
    return resolved


def validate_factor_grids(paths: Sequence[str]) -> dict:
    if not paths:
        raise ValueError("At least one terrain factor raster is required.")
    with rasterio.open(paths[0]) as reference:
        grid = {
            "height": reference.height,
            "width": reference.width,
            "transform": reference.transform,
            "crs": reference.crs,
            "profile": reference.profile.copy(),
        }
    for path in track(
        list(paths[1:]),
        total=max(len(paths) - 1, 0),
        desc="Checking terrain-factor grids",
        unit="raster",
    ):
        with rasterio.open(path) as source:
            problems = []
            if (source.height, source.width) != (grid["height"], grid["width"]):
                problems.append(
                    f"shape {source.height}x{source.width} != "
                    f"{grid['height']}x{grid['width']}"
                )
            if source.crs != grid["crs"]:
                problems.append(f"CRS {source.crs} != {grid['crs']}")
            if not _same_transform(source.transform, grid["transform"]):
                problems.append("affine transform differs")
            if problems:
                raise ValueError(f"Raster grid mismatch for {path}: {'; '.join(problems)}")
    return grid


def _read_coarse_features(paths: Sequence[str], downsample: int, grid: Mapping) -> dict:
    coarse_height = max(1, int(np.ceil(grid["height"] / downsample)))
    coarse_width = max(1, int(np.ceil(grid["width"] / downsample)))
    arrays = []
    valid = np.ones((coarse_height, coarse_width), dtype=bool)
    for path in track(
        list(paths),
        total=len(paths),
        desc="Reading low-resolution terrain factors",
        unit="factor",
    ):
        with rasterio.open(path) as source:
            data = source.read(
                1,
                out_shape=(coarse_height, coarse_width),
                masked=True,
                resampling=Resampling.average,
            )
        values = np.asarray(data.data, dtype=np.float64)
        factor_valid = np.isfinite(values) & ~np.ma.getmaskarray(data)
        valid &= factor_valid
        arrays.append(values)

    feature_cube = np.stack(arrays, axis=-1)
    components, component_count = ndimage.label(valid, structure=FOUR_NEIGHBOURS)
    if component_count == 0:
        raise RuntimeError("No common finite terrain cells were found.")
    component_sizes = np.bincount(components.ravel())[1:]
    largest_component = int(np.argmax(component_sizes) + 1)
    domain = components == largest_component
    excluded_cells = int(valid.sum() - domain.sum())
    excluded_fraction = excluded_cells / max(int(valid.sum()), 1)
    return {
        "feature_cube": feature_cube,
        "valid": valid,
        "domain": domain,
        "height": coarse_height,
        "width": coarse_width,
        "component_count": int(component_count),
        "component_sizes": [int(value) for value in sorted(component_sizes, reverse=True)],
        "excluded_cells": excluded_cells,
        "excluded_fraction": float(excluded_fraction),
    }


def _robust_feature_matrix(feature_cube: np.ndarray, domain: np.ndarray,
                           spatial_weight: float, factor_names: Sequence[str]) -> tuple[np.ndarray, dict]:
    raw = feature_cube[domain]
    medians = np.nanpercentile(raw, 50, axis=0)
    lower = np.nanpercentile(raw, 5, axis=0)
    upper = np.nanpercentile(raw, 95, axis=0)
    scales = upper - lower
    constant = scales <= np.finfo(np.float64).eps
    safe_scales = scales.copy()
    safe_scales[constant] = 1.0
    standardized = np.clip((raw - medians) / safe_scales, -2.0, 2.0)
    standardized[:, constant] = 0.0

    rows, cols = np.nonzero(domain)
    coordinate_features = np.column_stack(
        (
            rows / max(domain.shape[0] - 1, 1),
            cols / max(domain.shape[1] - 1, 1),
        )
    )
    if spatial_weight > 0:
        matrix = np.column_stack((standardized, spatial_weight * coordinate_features))
    else:
        matrix = standardized
    if not np.all(np.isfinite(matrix)):
        raise RuntimeError("Non-finite values remain after terrain feature scaling.")

    scaling = {
        name: {
            "median": float(median),
            "percentile_05": float(low),
            "percentile_95": float(high),
            "scale": float(scale),
            "constant": bool(is_constant),
        }
        for name, median, low, high, scale, is_constant in zip(
            factor_names, medians, lower, upper, scales, constant
        )
    }
    return matrix.astype(np.float64, copy=False), scaling


def _grid_connectivity(domain: np.ndarray) -> sparse.csr_matrix:
    node_ids = np.full(domain.shape, -1, dtype=np.int64)
    node_ids[domain] = np.arange(int(domain.sum()), dtype=np.int64)
    horizontal = domain[:, :-1] & domain[:, 1:]
    vertical = domain[:-1, :] & domain[1:, :]
    h_rows, h_cols = np.nonzero(horizontal)
    v_rows, v_cols = np.nonzero(vertical)
    left = node_ids[h_rows, h_cols]
    right = node_ids[h_rows, h_cols + 1]
    top = node_ids[v_rows, v_cols]
    bottom = node_ids[v_rows + 1, v_cols]
    first = np.concatenate((left, top))
    second = np.concatenate((right, bottom))
    rows = np.concatenate((first, second))
    cols = np.concatenate((second, first))
    values = np.ones(rows.size, dtype=np.uint8)
    connectivity = sparse.coo_matrix(
        (values, (rows, cols)), shape=(int(domain.sum()), int(domain.sum()))
    ).tocsr()
    graph_components, _ = sparse.csgraph.connected_components(connectivity)
    if graph_components != 1:
        raise RuntimeError(
            f"Regionalization domain unexpectedly has {graph_components} graph components."
        )
    return connectivity


def _ordered_region_labels(raw_labels: np.ndarray, domain: np.ndarray) -> np.ndarray:
    rows, cols = np.nonzero(domain)
    centroids = []
    for label in np.unique(raw_labels):
        selected = raw_labels == label
        centroids.append((float(rows[selected].mean()), float(cols[selected].mean()), int(label)))
    order = sorted(centroids)
    mapping = {old_label: new_label for new_label, (_, _, old_label) in enumerate(order, start=1)}
    labels = np.zeros(domain.shape, dtype=np.uint16)
    labels[domain] = np.asarray([mapping[int(label)] for label in raw_labels], dtype=np.uint16)
    return labels


def cluster_continuous_regions(matrix: np.ndarray, connectivity: sparse.csr_matrix,
                               domain: np.ndarray, region_count: int) -> tuple[np.ndarray, dict]:
    if region_count >= matrix.shape[0]:
        raise ValueError(
            f"region_count={region_count} must be smaller than the {matrix.shape[0]} valid cells."
        )
    model = AgglomerativeClustering(
        n_clusters=region_count,
        linkage="ward",
        connectivity=connectivity,
        compute_full_tree=True,
    )
    raw_labels = model.fit_predict(matrix)
    labels = _ordered_region_labels(raw_labels, domain)
    component_counts = {}
    sizes = {}
    for region_id in range(1, region_count + 1):
        _, component_count = ndimage.label(labels == region_id, structure=FOUR_NEIGHBOURS)
        component_counts[region_id] = int(component_count)
        sizes[region_id] = int(np.count_nonzero(labels == region_id))
    if any(count != 1 for count in component_counts.values()):
        raise RuntimeError(f"Spatial constraint failed: component counts={component_counts}")
    total = max(int(domain.sum()), 1)
    return labels, {
        "coarse_component_counts": component_counts,
        "coarse_cell_counts": sizes,
        "coarse_fractions": {
            region_id: count / total for region_id, count in sizes.items()
        },
    }


def requested_outputs(config: TerrainRegionConfig) -> dict[int, str]:
    return {5: config.output_path}


def diagnostic_path(raster_path: str) -> str:
    path = Path(raster_path)
    return str(path.with_name(f"{path.stem}_diagnostics.json"))


def summary_path(raster_path: str) -> str:
    path = Path(raster_path)
    return str(path.with_name(f"{path.stem}_summary.csv"))


def names_path(raster_path: str) -> str:
    path = Path(raster_path)
    return str(path.with_name(f"{path.stem}_region_names.csv"))


def plot_path(raster_path: str) -> str:
    path = Path(raster_path)
    return str(path.with_name(f"{path.stem}_layout.png"))


def _source_signature(paths: Sequence[str]) -> list[dict]:
    result = []
    for path in paths:
        stat = os.stat(path)
        result.append({
            "path": os.path.abspath(path),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        })
    return result


def _configuration_signature(config: TerrainRegionConfig, paths: Sequence[str],
                             region_count: int) -> dict:
    signature = {
        "method": "spatially_constrained_ward_on_terrain",
        "algorithm_version": 1,
        "region_count": int(region_count),
        "factor_names": list(config.factor_names),
        "downsample": int(config.downsample),
        "spatial_weight": float(config.spatial_weight),
        "minimum_region_fraction": float(config.minimum_region_fraction),
        "maximum_excluded_coarse_fraction": float(
            config.maximum_excluded_coarse_fraction
        ),
        "maximum_removed_full_fraction": float(config.maximum_removed_full_fraction),
        "source_rasters": _source_signature(paths),
    }
    canonical = json.dumps(signature, sort_keys=True, ensure_ascii=False).encode("utf-8")
    signature["sha256"] = hashlib.sha256(canonical).hexdigest()
    return signature


def _existing_output_matches(path: str, signature: Mapping) -> bool:
    if not os.path.exists(path) or not os.path.exists(diagnostic_path(path)):
        return False
    try:
        with open(diagnostic_path(path), encoding="utf-8") as handle:
            diagnostics = json.load(handle)
    except (OSError, ValueError, TypeError):
        return False
    return diagnostics.get("configuration", {}).get("sha256") == signature.get("sha256")


def _full_valid_mask(sources: Sequence[rasterio.DatasetReader], window: Window) -> np.ndarray:
    shape = (int(window.height), int(window.width))
    valid = np.ones(shape, dtype=bool)
    for source in sources:
        data = source.read(1, window=window, masked=True)
        valid &= ~np.ma.getmaskarray(data)
        valid &= np.isfinite(np.asarray(data.data, dtype=np.float64))
    return valid


def _write_full_resolution_rasters(paths: Sequence[str], grid: Mapping,
                                   coarse_labels: Mapping[int, np.ndarray],
                                   outputs: Mapping[int, str]) -> dict[int, str]:
    profile = grid["profile"].copy()
    profile.update(
        driver="GTiff",
        count=1,
        dtype="uint16",
        nodata=0,
        compress="deflate",
        predictor=2,
        zlevel=4,
        tiled=True,
        blockxsize=512,
        blockysize=512,
        BIGTIFF="IF_SAFER",
    )
    temporary_paths = {
        count: str(Path(path).with_name(f".{Path(path).name}.building.tif"))
        for count, path in outputs.items()
    }
    for path in outputs.values():
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    try:
        with ExitStack() as stack:
            sources = [stack.enter_context(rasterio.open(path)) for path in paths]
            destinations = {
                count: stack.enter_context(rasterio.open(temporary_paths[count], "w", **profile))
                for count in outputs
            }
            windows = _iter_windows(grid["height"], grid["width"])
            for window in track(
                windows,
                total=window_count(grid["height"], grid["width"], 1024),
                desc="Writing full-resolution macro-regions",
                unit="tile",
            ):
                row_start = int(window.row_off)
                col_start = int(window.col_off)
                row_centres = np.arange(row_start, row_start + int(window.height)) + 0.5
                col_centres = np.arange(col_start, col_start + int(window.width)) + 0.5
                valid = _full_valid_mask(sources, window)
                for count, labels in coarse_labels.items():
                    coarse_rows = np.minimum(
                        (row_centres * labels.shape[0] / grid["height"]).astype(np.int64),
                        labels.shape[0] - 1,
                    )
                    coarse_cols = np.minimum(
                        (col_centres * labels.shape[1] / grid["width"]).astype(np.int64),
                        labels.shape[1] - 1,
                    )
                    block = labels[np.ix_(coarse_rows, coarse_cols)].copy()
                    block[~valid] = 0
                    destinations[count].write(block, 1, window=window)
        return temporary_paths
    except Exception:
        for path in temporary_paths.values():
            try:
                Path(path).unlink(missing_ok=True)
            except OSError:
                pass
        raise


def _enforce_full_connectivity(path: str, region_count: int,
                               maximum_removed_fraction: float) -> dict:
    with rasterio.open(path, "r+") as dataset:
        data = dataset.read(1)
        initial_valid = int(np.count_nonzero(data))
        component_counts = {}
        removed_by_region = {}
        for region_id in range(1, region_count + 1):
            components, count = ndimage.label(data == region_id, structure=FOUR_NEIGHBOURS)
            component_counts[region_id] = int(count)
            if count == 0:
                raise RuntimeError(f"Region {region_id} has no full-resolution cells.")
            sizes = np.bincount(components.ravel())[1:]
            largest = int(np.argmax(sizes) + 1)
            disconnected = (components > 0) & (components != largest)
            removed = int(disconnected.sum())
            removed_by_region[region_id] = removed
            if removed:
                data[disconnected] = 0
        total_removed = int(sum(removed_by_region.values()))
        removed_fraction = total_removed / max(initial_valid, 1)
        if removed_fraction > maximum_removed_fraction:
            raise RuntimeError(
                f"Full-resolution connectivity would exclude {removed_fraction:.3%} of valid "
                f"cells, above the configured {maximum_removed_fraction:.3%}. Use a finer "
                "macro_region_downsample or an authoritative external region raster."
            )
        if total_removed:
            dataset.write(data, 1)

        final_counts = {
            region_id: int(np.count_nonzero(data == region_id))
            for region_id in range(1, region_count + 1)
        }
        final_components = {}
        for region_id in range(1, region_count + 1):
            _, count = ndimage.label(data == region_id, structure=FOUR_NEIGHBOURS)
            final_components[region_id] = int(count)
        if any(count != 1 for count in final_components.values()):
            raise RuntimeError(f"Full-resolution connectivity repair failed: {final_components}")
    return {
        "component_counts_before_cleanup": component_counts,
        "removed_cells_by_region": removed_by_region,
        "total_removed_cells": total_removed,
        "removed_fraction": float(removed_fraction),
        "component_counts_after_cleanup": final_components,
        "full_cell_counts": final_counts,
    }


def _region_rows(labels: np.ndarray, feature_cube: np.ndarray, factor_names: Sequence[str],
                 grid: Mapping, full_counts: Mapping[int, int]) -> list[dict]:
    coarse_transform = grid["transform"] * rasterio.Affine.scale(
        grid["width"] / labels.shape[1], grid["height"] / labels.shape[0]
    )
    total_coarse = max(int(np.count_nonzero(labels)), 1)
    total_full = max(int(sum(full_counts.values())), 1)
    rows = []
    for region_id in range(1, int(labels.max()) + 1):
        mask = labels == region_id
        raster_rows, raster_cols = np.nonzero(mask)
        centroid_row = float(raster_rows.mean())
        centroid_col = float(raster_cols.mean())
        centroid_x, centroid_y = rasterio.transform.xy(
            coarse_transform, centroid_row, centroid_col, offset="center"
        )
        row = {
            "region_id": region_id,
            "region_name": f"terrain_region_{region_id}",
            "coarse_cells": int(mask.sum()),
            "coarse_fraction": float(mask.sum() / total_coarse),
            "full_cells": int(full_counts[region_id]),
            "full_fraction": float(full_counts[region_id] / total_full),
            "centroid_x": float(centroid_x),
            "centroid_y": float(centroid_y),
        }
        for factor_index, factor_name in enumerate(factor_names):
            row[f"mean_{factor_name}"] = float(np.mean(feature_cube[..., factor_index][mask]))
        rows.append(row)
    return rows


def _write_csv_outputs(raster_path: str, rows: Sequence[Mapping]) -> None:
    if not rows:
        return
    with open(summary_path(raster_path), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(names_path(raster_path), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["region_id", "region_name"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"region_id": row["region_id"], "region_name": row["region_name"]})


def _write_layout_plot(path: str, labels: np.ndarray, region_count: int) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import BoundaryNorm, ListedColormap
        from matplotlib.patches import Patch
    except ImportError:
        warnings.warn("matplotlib is unavailable; macro-region layout plot was skipped.")
        return
    display = np.ma.masked_equal(labels, 0)
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, region_count))
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(0.5, region_count + 1.5), region_count)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(display, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_title("Terrain-derived continuous macro-regions (no inventory used)")
    ax.set_xlabel("Coarse-grid column")
    ax.set_ylabel("Coarse-grid row")
    ax.legend(
        handles=[Patch(color=colors[index - 1], label=f"terrain_region_{index}")
                 for index in range(1, region_count + 1)],
        loc="best",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_terrain_macro_regions(config: TerrainRegionConfig, overwrite: bool = False) -> dict:
    """Build the single K=5 terrain-derived regionalization."""
    config.validate()
    console(
        "Region-building configuration: "
        f"K={config.region_count}，"
        f"factors={len(config.factor_names)}, downsample={config.downsample}"
    )
    factor_paths = resolve_factor_paths(config.factors_dir, config.factor_names)
    grid = validate_factor_grids(factor_paths)
    outputs = requested_outputs(config)
    signatures = {
        count: _configuration_signature(config, factor_paths, count)
        for count in outputs
    }
    pending = {}
    skipped = {}
    for count, path in outputs.items():
        if not os.path.exists(path):
            pending[count] = path
        elif not overwrite and _existing_output_matches(path, signatures[count]):
            skipped[count] = path
        elif overwrite:
            pending[count] = path
        else:
            raise FileExistsError(
                f"Existing macro-region raster does not match the requested configuration: {path}. "
                "Inspect it or rebuild explicitly with --force-regions/--force."
            )
    if not pending:
        console(f"All region products already exist with matching configuration: {sorted(skipped)}")
        return {"built": {}, "skipped": skipped, "outputs": outputs}

    prepared = _read_coarse_features(factor_paths, config.downsample, grid)
    if prepared["excluded_fraction"] > config.maximum_excluded_coarse_fraction:
        raise RuntimeError(
            f"The common terrain mask has {prepared['excluded_fraction']:.3%} outside its "
            f"largest connected component, above the configured "
            f"{config.maximum_excluded_coarse_fraction:.3%}."
        )
    matrix, scaling = _robust_feature_matrix(
        prepared["feature_cube"],
        prepared["domain"],
        config.spatial_weight,
        config.factor_names,
    )
    connectivity = _grid_connectivity(prepared["domain"])

    labels_by_count = {}
    cluster_audits = {}
    for count in track(
        list(pending),
        total=len(pending),
        desc="Spatially constrained clustering",
        unit="K",
    ):
        with timed_task(f"Fitting K={count} Ward spatial clustering"):
            labels, audit = cluster_continuous_regions(
                matrix, connectivity, prepared["domain"], count
            )
        labels_by_count[count] = labels
        cluster_audits[count] = audit
        small = [
            region_id for region_id, fraction in audit["coarse_fractions"].items()
            if fraction < config.minimum_region_fraction
        ]
        if small:
            warnings.warn(
                f"K={count} contains regions below the configured "
                f"{config.minimum_region_fraction:.1%} area warning threshold: {small}."
            )

    temporary = _write_full_resolution_rasters(
        factor_paths, grid, labels_by_count, pending
    )
    built = {}
    try:
        for count, final_path in track(
            list(pending.items()),
            total=len(pending),
            desc="Repairing region connectivity and writing outputs",
            unit="product",
        ):
            full_audit = _enforce_full_connectivity(
                temporary[count], count, config.maximum_removed_full_fraction
            )
            rows = _region_rows(
                labels_by_count[count],
                prepared["feature_cube"],
                config.factor_names,
                grid,
                full_audit["full_cell_counts"],
            )
            diagnostic = {
                "method": "spatially_constrained_ward_on_terrain",
                "scientific_interpretation": "terrain-derived physiographic macro-regions",
                "authoritative_tectonic_or_watershed_units": False,
                "landslide_inventory_used": False,
                "model_output_used": False,
                "configuration": signatures[count],
                "grid": {
                    "height": int(grid["height"]),
                    "width": int(grid["width"]),
                    "crs": str(grid["crs"]),
                    "transform": list(grid["transform"]),
                },
                "coarse_grid": {
                    "height": int(prepared["height"]),
                    "width": int(prepared["width"]),
                    "valid_cells_before_component_filter": int(prepared["valid"].sum()),
                    "regionalization_cells": int(prepared["domain"].sum()),
                    "input_component_count": prepared["component_count"],
                    "input_component_sizes": prepared["component_sizes"],
                    "excluded_cells": prepared["excluded_cells"],
                    "excluded_fraction": prepared["excluded_fraction"],
                },
                "feature_scaling": scaling,
                "cluster_connectivity": cluster_audits[count],
                "full_resolution_connectivity": full_audit,
                "regions": rows,
                "preregistration_note": (
                    "Freeze this raster before reading fold outcomes. Do not tune boundaries or K "
                    "using landslide labels, DWSS results, validation metrics, or test metrics."
                ),
            }
            os.replace(temporary[count], final_path)
            _write_csv_outputs(final_path, rows)
            with open(diagnostic_path(final_path), "w", encoding="utf-8") as handle:
                json.dump(diagnostic, handle, indent=2, ensure_ascii=False)
            if config.write_plot:
                _write_layout_plot(plot_path(final_path), labels_by_count[count], count)
            built[count] = final_path
            console(
                f"K={count} completed: {final_path}; "
                f"valid pixels={sum(full_audit['full_cell_counts'].values()):,}"
            )
    except Exception:
        for path in temporary.values():
            try:
                Path(path).unlink(missing_ok=True)
            except OSError:
                pass
        raise
    return {"built": built, "skipped": skipped, "outputs": outputs}


def build_from_xml(xml_path: str, overwrite: bool = False) -> dict:
    return build_terrain_macro_regions(config_from_xml(xml_path), overwrite=overwrite)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build label-independent continuous terrain macro-regions."
    )
    parser.add_argument("xml", help="Path to Landslide_susceptibility_mapping.xml")
    parser.add_argument("--force", action="store_true", help="Replace existing generated products.")
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show live terminal progress bars (default: enabled).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    configure_progress(arguments.progress)
    result = build_from_xml(arguments.xml, overwrite=arguments.force)
    for count, path in sorted(result["built"].items()):
        console(f"Built K={count}: {path}")
    for count, path in sorted(result["skipped"].items()):
        console(f"Validated K={count}: {path}")
