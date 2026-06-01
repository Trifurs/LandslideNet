import os

import numpy as np
from rasterio.warp import transform as transform_coords


def setup_plot_style(plt):
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#30343b",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "#d8dde6",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.7,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#d0d5dd",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
    })


def to_wgs84(xs, ys, crs):
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if crs is None:
        raise ValueError("Raster CRS is missing; WGS84 map plots require a defined CRS.")
    try:
        if crs.to_epsg() == 4326:
            return xs, ys
    except AttributeError:
        pass
    lon, lat = transform_coords(crs, "EPSG:4326", xs.tolist(), ys.tolist())
    return np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)


def add_wgs84_axes(ax, title=None):
    ax.set_xlabel("Longitude (WGS84)")
    ax.set_ylabel("Latitude (WGS84)")
    if title:
        ax.set_title(title, pad=10, weight="bold")
    ax.tick_params(axis="both", labelsize=9)
    ax.set_axisbelow(True)


def add_panel_label(ax, label):
    ax.text(
        0.015,
        0.985,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "#d0d5dd", "alpha": 0.95},
    )


def save_figure(fig, path, dpi=350):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi)
