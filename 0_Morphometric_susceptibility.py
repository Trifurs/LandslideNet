#!/usr/bin/env python3
# Susceptibility_pysheds_lightfix.py
"""
Lightweight fixes to Morphometric Susceptibility pipeline:
- tile-level fast depression filling (iterative min-filter-based)
- small deterministic jitter to break flats
- use filled DEM for D8 flow direction
- keep tiled slope, memmap, pointer-jumping, β, and Ps_L logic unchanged
Author: adapted (2025-11-18)
"""
import os
import math
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
from tqdm import tqdm
import rasterio
from rasterio.windows import Window
from scipy.ndimage import convolve, gaussian_filter, minimum_filter
from scipy.stats import rankdata

# try numba (optional)
_USE_NUMBA = False
try:
    import numba
    from numba import njit, prange
    _USE_NUMBA = True
    print("numba detected — enabling nb-accelerated routines.")
except Exception:
    print("numba not detected — running pure-NumPy fallback.")

# ----------------------------
# utils
# ----------------------------
def ensure_dir(d):
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def print_progress(s):
    print(s, flush=True)

# D8 offsets order: N, NE, E, SE, S, SW, W, NW
_OFFSETS_DI = np.array([-1, -1, 0, 1, 1, 1, 0, -1], dtype=np.int32)
_OFFSETS_DJ = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int32)

# ----------------------------
# Sobel slope (tile)
# ----------------------------
_KX = np.array([[1,0,-1],[2,0,-2],[1,0,-1]],dtype=np.float32)
_KY = _KX.T

def slope_from_dem_tile_sobel(dem_tile, transform_tile):
    """
    Compute local slope in degrees for a tile (handles NaNs by local averaging).
    dem_tile: 2D float array, may contain np.nan
    transform_tile: rasterio transform for pixel size
    """
    nan_mask = np.isnan(dem_tile)
    if np.any(nan_mask):
        dem_valid = dem_tile.copy()
        dem_valid[nan_mask] = 0.0
        w = (~nan_mask).astype(np.float32)
        kernel = np.ones((3,3), dtype=np.float32)
        denom = convolve(w, kernel, mode='mirror')
        numer = convolve(dem_valid, kernel, mode='mirror')
        avg = np.where(denom>0, numer/denom, 0.0)
        dem_fill = dem_tile.copy()
        dem_fill[nan_mask] = avg[nan_mask]
    else:
        dem_fill = dem_tile

    dx = abs(transform_tile.a)
    dy = abs(transform_tile.e)
    gx = convolve(dem_fill, _KX, mode='mirror') / (8.0 * dx)
    gy = convolve(dem_fill, _KY, mode='mirror') / (8.0 * dy)
    slope_rad = np.arctan(np.hypot(gx, gy))
    slope_deg = np.degrees(slope_rad)
    slope_deg[np.isnan(dem_tile)] = np.nan
    return slope_deg

# ----------------------------
# simple local depression filling (tile-level) + flat resolver
# ----------------------------
def fill_depressions_simple(dem_tile, max_iter=200, epsilon=1e-6):
    """
    Lightweight iterative minimum-filter-based depression filling.
    - dem_tile: 2D float array with np.nan for nodata
    Returns dem_filled (float32) with NaNs preserved where whole-window NaN.
    Notes:
      * Fast and robust for many practical hillslope DEMs.
      * Not a full priority-flood, but reduces common pits and flats.
    """
    dem = dem_tile.astype(np.float64).copy()
    nan_mask = np.isnan(dem)

    # If entire tile is NaN, return as-is
    if nan_mask.all():
        return dem.astype(np.float32)

    # 1) Fill isolated NaNs with local average (so minimum_filter works)
    if np.any(nan_mask):
        w = (~nan_mask).astype(np.float64)
        kernel = np.ones((3,3), dtype=np.float64)
        denom = convolve(w, kernel, mode='mirror')
        numer = convolve(np.where(nan_mask, 0.0, dem), kernel, mode='mirror')
        avg = np.where(denom>0, numer/denom, np.nan)
        # only fill NaNs where avg is finite; leave truly isolated as NaN
        fill_idx = nan_mask & np.isfinite(avg)
        dem[fill_idx] = avg[fill_idx]
        # if still NaNs (very isolated), replace with nearest finite using minimum_filter on mask:
        if np.any(np.isnan(dem)):
            # replace remaining NaNs by global min (safe fallback)
            global_min = np.nanmin(dem)
            dem[np.isnan(dem)] = global_min

    # 2) Iteratively raise pits: dem_new = max(dem, min_neighbor(dem))
    for it in range(max_iter):
        neigh_min = minimum_filter(dem, size=3, mode='mirror')
        new_dem = np.maximum(dem, neigh_min)
        # convergence
        if np.allclose(new_dem, dem, equal_nan=True):
            dem = new_dem
            break
        dem = new_dem

    # 3) Resolve remaining flats deterministically by adding tiny tile-based jitter
    #    The jitter is deterministic (not random) to keep reproducible results.
    rows, cols = dem.shape
    rr, cc = np.indices((rows, cols))
    jitter = ((rr + 2*cc) % 8).astype(np.float64) * epsilon
    dem = dem + jitter

    # 4) restore NaN where original tile was fully nodata (we used fallback earlier)
    dem[nan_mask] = np.nan

    return dem.astype(np.float32)

# ----------------------------
# fast vectorized D8 (tile) - uses filled DEM
# ----------------------------
def flow_dir_from_dem_tile_fast(dem_tile_filled):
    """
    Compute D8 flow direction array (uint8) for a filled DEM tile.
    Returns:
      flow: uint8 array with values 0..7 for the 8 directions, and 255 for no-flow/nodata.
    """
    rows, cols = dem_tile_filled.shape
    padded = np.pad(dem_tile_filled, pad_width=1, mode='edge')
    center = padded[1:-1,1:-1]
    neigh_stack = np.empty((8, rows, cols), dtype=np.float32)
    neigh_stack[0] = padded[0:-2,1:-1]   # N
    neigh_stack[1] = padded[0:-2,2:]     # NE
    neigh_stack[2] = padded[1:-1,2:]     # E
    neigh_stack[3] = padded[2:,2:]       # SE
    neigh_stack[4] = padded[2:,1:-1]     # S
    neigh_stack[5] = padded[2:,0:-2]     # SW
    neigh_stack[6] = padded[1:-1,0:-2]   # W
    neigh_stack[7] = padded[0:-2,0:-2]   # NW

    # Compute drops to neighbors (center - neighbor)
    diffs = center[np.newaxis,:,:] - neigh_stack  # shape (8,rows,cols)
    # deal with NaNs: treat diffs where either center or neighbor NaN as -inf (no drop)
    diffs[np.isnan(diffs)] = -np.inf
    # max drop and direction
    max_drop = np.max(diffs, axis=0)
    argmax = np.argmax(diffs, axis=0).astype(np.uint8)
    # pixels with no positive drop or NaN center -> no flow
    no_flow = (max_drop <= 0) | np.isnan(center)
    flow = argmax
    flow[no_flow] = 255
    return flow

# ----------------------------
# receiver computation (flat -> 1D receiver indices)
# ----------------------------
def receivers_from_flowdir_numpy(flow_dir_mm):
    rows, cols = flow_dir_mm.shape
    n = rows*cols
    fd = np.asarray(flow_dir_mm, dtype=np.uint8).ravel()
    ii = np.arange(n)//cols
    jj = np.arange(n)%cols
    receiver = np.full(n, -1, dtype=np.int32)
    mask = fd != 255
    if np.any(mask):
        dirs = fd[mask].astype(np.int32)
        ri = ii[mask] + _OFFSETS_DI[dirs]
        rj = jj[mask] + _OFFSETS_DJ[dirs]
        inb = (ri >=0)&(ri<rows)&(rj>=0)&(rj<cols)
        rec = np.full(mask.sum(), -1, dtype=np.int32)
        rec[inb] = (ri[inb]*cols + rj[inb]).astype(np.int32)
        receiver[mask] = rec
    return receiver

# ----------------------------
# pointer-jumping (numba accelerated if available)
# ----------------------------
if _USE_NUMBA:
    @njit(cache=True, parallel=True)
    def pointer_jumping_numba(receiver):
        n = receiver.size
        cur = receiver.copy()
        invalid = cur < 0
        idx = np.arange(n, dtype=np.int32)
        cur[invalid] = idx[invalid]
        changed = True
        it = 0
        while changed and it < 1000:
            changed = False
            for i in prange(n):
                newv = cur[cur[i]]
                if newv != cur[i]:
                    cur[i] = newv
                    changed = True
            it += 1
        cur[invalid] = -1
        return cur
else:
    def pointer_jumping_labels_numpy(receiver):
        n = receiver.size
        cur = receiver.copy()
        invalid = cur < 0
        idx = np.arange(n, dtype=np.int32)
        cur[invalid] = idx[invalid]
        it = 0
        while True:
            cur2 = cur[cur]
            it +=1
            if np.array_equal(cur2, cur) or it>1000:
                break
            cur = cur2
        cur[invalid] = -1
        return cur

# ----------------------------
# memmap & tiling (main tile loop uses fill_depressions_simple)
# ----------------------------
def create_memmap_files(tmp_dir, rows, cols):
    ensure_dir(tmp_dir)
    flow_dir_path = os.path.join(tmp_dir, "flow_dir_u8.dat")
    slope_path = os.path.join(tmp_dir, "slope_f32.dat")
    fd_mm = np.memmap(flow_dir_path, dtype=np.uint8, mode='w+', shape=(rows, cols))
    sl_mm = np.memmap(slope_path, dtype=np.float32, mode='w+', shape=(rows, cols))
    fd_mm[:] = 255
    sl_mm[:] = np.nan
    return flow_dir_path, slope_path, fd_mm, sl_mm

def tile_generator(width,height,tile_size,overlap):
    for row_off in range(0,height,tile_size):
        for col_off in range(0,width,tile_size):
            i0 = max(0,row_off-overlap)
            j0 = max(0,col_off-overlap)
            i1 = min(height,row_off+tile_size+overlap)
            j1 = min(width,col_off+tile_size+overlap)
            read_win = Window(j0,i0,j1-j0,i1-i0)
            int_i0 = row_off
            int_j0 = col_off
            int_i1 = min(row_off+tile_size,height)
            int_j1 = min(col_off+tile_size,width)
            write_win = Window(int_j0,int_i0,int_j1-int_j0,int_i1-int_i0)
            yield read_win, write_win, (i0,i1,j0,j1),(int_i0,int_i1,int_j0,int_j1)

def compute_tiled_slope_and_flowdir(dem_path,tmp_dir,tile_size=2048,overlap=64):
    print_progress("Step 1: tiled slope & flow_dir ...")
    with rasterio.open(dem_path) as src:
        height = src.height
        width = src.width
        profile = src.profile.copy()
        nodata = src.nodata if src.nodata is not None else -9999
        flow_dir_path, slope_path, fd_mm, sl_mm = create_memmap_files(tmp_dir,height,width)
        total_tiles = math.ceil(width/tile_size)*math.ceil(height/tile_size)
        pbar = tqdm(total=total_tiles, desc="tiles")
        for read_win, write_win, read_bounds, write_bounds in tile_generator(width,height,tile_size,overlap):
            i0,i1,j0,j1 = read_bounds
            int_i0,int_i1,int_j0,int_j1 = write_bounds
            dem_tile = src.read(1, window=read_win, out_dtype=np.float32)
            # mask nodata -> nan
            dem_tile = dem_tile.astype(np.float32)
            dem_tile[dem_tile==nodata] = np.nan

            # 1) Light depression fill & flat resolution (tile)
            dem_filled = fill_depressions_simple(dem_tile, max_iter=200, epsilon=1e-6)

            # 2) slope using original transform for tile
            transform_tile = rasterio.windows.transform(read_win, src.transform)
            slope_tile = slope_from_dem_tile_sobel(dem_filled, transform_tile)

            # 3) flowdir from filled DEM
            flow_tile = flow_dir_from_dem_tile_fast(dem_filled)

            # copy central (non-overlap) region into memmap
            off_i0 = int_i0 - i0
            off_i1 = off_i0 + (int_i1 - int_i0)
            off_j0 = int_j0 - j0
            off_j1 = off_j0 + (int_j1 - int_j0)

            sl_mm[int_i0:int_i1, int_j0:int_j1] = slope_tile[off_i0:off_i1, off_j0:off_j1]
            fd_mm[int_i0:int_i1, int_j0:int_j1] = flow_tile[off_i0:off_i1, off_j0:off_j1]
            pbar.update(1)
        pbar.close()
        fd_mm.flush()
        sl_mm.flush()
    print_progress("tiled slope & flow_dir done.")
    return flow_dir_path,slope_path,profile,nodata,height,width

# ----------------------------
# simplified flow accumulation (with weights)
# ----------------------------
def compute_flow_accumulation(flow_dir_mm, weights, height, width):
    """Compute flow accumulation with custom weights (weights must be 1D flat array length n)."""
    n = height * width
    receiver = receivers_from_flowdir_numpy(flow_dir_mm)
    # initialize accumulation with weights (ensure float64)
    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.size != n:
        raise ValueError("weights length must equal n (height*width)")
    acc = w.copy()
    acc[~np.isfinite(acc)] = 0.0

    # indegree
    indeg = np.bincount(receiver[receiver >= 0], minlength=n).astype(np.int32)

    # queue of sources (indeg==0)
    q = np.nonzero((indeg == 0))[0]
    head, tail = 0, q.size
    queue = np.empty(n, dtype=np.int32)
    if tail > 0:
        queue[:tail] = q

    # propagate accumulation upstream -> down receiver chain
    while head < tail:
        u = queue[head]
        head += 1
        v = receiver[u]
        if v >= 0:
            acc[v] += acc[u]
            indeg[v] -= 1
            if indeg[v] == 0:
                queue[tail] = v
                tail += 1
    return acc

# ----------------------------
# compute β (mean drainage area slope)
# ----------------------------
def compute_beta(flow_dir_path, slope_path, tmp_dir, profile, nodata, height, width):
    print_progress("Step 2: computing β (mean drainage area slope)...")
    fd_mm = np.memmap(flow_dir_path, dtype=np.uint8, mode='r', shape=(height, width))
    slope_mm = np.memmap(slope_path, dtype=np.float32, mode='r', shape=(height, width))

    slope_flat = slope_mm.ravel().astype(np.float64)
    slope_flat[np.isnan(slope_flat)] = 0.0

    flow_acc_slope = compute_flow_accumulation(fd_mm, slope_flat, height, width)
    flow_acc = compute_flow_accumulation(fd_mm, np.ones_like(slope_flat), height, width)

    beta_flat = np.zeros_like(flow_acc, dtype=np.float32)
    valid_mask = flow_acc > 0
    beta_flat[valid_mask] = (flow_acc_slope[valid_mask] / flow_acc[valid_mask]).astype(np.float32)
    beta_flat[~valid_mask] = np.nan

    beta_path = os.path.join(tmp_dir, "beta_f32.dat")
    beta_mm = np.memmap(beta_path, dtype=np.float32, mode='w+', shape=(height, width))
    beta_mm[:] = beta_flat.reshape((height, width))
    beta_mm.flush()
    print_progress("β computed and saved.")
    return beta_path

# ----------------------------
# compute global receiver & flow accumulation (unchanged logic)
# ----------------------------
def compute_global_receiver_and_acc(flow_dir_path, slope_path, tmp_dir, profile, nodata, height, width, use_numba=_USE_NUMBA):
    print_progress("Step 2: global receiver & flow accumulation ...")
    fd_mm = np.memmap(flow_dir_path, dtype=np.uint8, mode='r', shape=(height, width))
    sl_mm = np.memmap(slope_path, dtype=np.float32, mode='r', shape=(height, width))

    # Compute β (mean drainage area slope)
    beta_path = compute_beta(flow_dir_path, slope_path, tmp_dir, profile, nodata, height, width)

    # Compute standard flow accumulation (for denominator)
    flow_acc = compute_flow_accumulation(fd_mm, np.ones(height * width, dtype=np.float64), height, width)

    # Save flow accumulation
    flow_acc_path = os.path.join(tmp_dir, "flow_acc_f64.dat")
    acc_mm = np.memmap(flow_acc_path, dtype=np.float64, mode='w+', shape=(height * width,))
    acc_mm[:] = flow_acc[:]
    acc_mm.flush()

    # Compute receiver array
    receiver = receivers_from_flowdir_numpy(fd_mm)
    receiver_path = os.path.join(tmp_dir, "receiver_i32.dat")
    recv_mm = np.memmap(receiver_path, dtype=np.int32, mode='w+', shape=(height * width,))
    recv_mm[:] = receiver[:]
    recv_mm.flush()

    # pointer jumping -> reps
    if _USE_NUMBA:
        try:
            reps = pointer_jumping_numba(receiver)
        except Exception:
            reps = pointer_jumping_labels_numpy(receiver)
    else:
        reps = pointer_jumping_labels_numpy(receiver)

    valid = reps >= 0
    labels = np.zeros_like(reps, dtype=np.int32)
    if np.any(valid):
        uniques, inv = np.unique(reps[valid], return_inverse=True)
        labels[valid] = inv + 1

    slope_units_path = os.path.join(tmp_dir, "slope_units_i32.dat")
    su_mm = np.memmap(slope_units_path, dtype=np.int32, mode='w+', shape=(height * width,))
    su_mm[:] = labels[:]
    su_mm.flush()

    print_progress("Receiver, flow_acc, slope_units memmapped.")
    return flow_acc_path, slope_units_path, beta_path, receiver_path

# ----------------------------
# save raster utility
# ----------------------------
def save_raster_from_array(arr, profile, out_path, nodata_val=-9999, dtype=rasterio.float32):
    prof = profile.copy()
    prof.update({'dtype': dtype, 'count': 1, 'nodata': nodata_val, 'compress': 'lzw', 'tiled': True, 'blockxsize': 256, 'blockysize': 256})
    out_arr = arr.astype(np.float32).copy()
    out_arr[np.isnan(out_arr)] = nodata_val
    with rasterio.open(out_path, 'w', **prof) as dst:
        dst.write(out_arr, 1)

# ----------------------------
# compute morphometric susceptibility (unchanged)
# ----------------------------
def compute_morphometric_and_save(flow_acc_path, beta_path, slope_units_path, profile, nodata, output_folder,
                                  write_intermediates=True, gaussian_sigma=None):
    print_progress("Step 3: computing morphometric index and saving outputs...")
    height = profile['height']
    width = profile['width']

    flow_acc_mm = np.memmap(flow_acc_path, dtype=np.float64, mode='r', shape=(height * width,))
    flow_acc = flow_acc_mm.reshape((height, width))

    beta_mm = np.memmap(beta_path, dtype=np.float32, mode='r', shape=(height, width))
    slope_units_mm = np.memmap(slope_units_path, dtype=np.int32, mode='r', shape=(height * width,))
    slope_units = slope_units_mm.reshape((height, width))

    ensure_dir(output_folder)
    inter_dir = os.path.join(output_folder, "intermediate")
    if write_intermediates:
        ensure_dir(inter_dir)
        save_raster_from_array(beta_mm, profile, os.path.join(inter_dir, "beta.tif"))
        save_raster_from_array(flow_acc.astype(np.float32), profile, os.path.join(inter_dir, "flow_acc.tif"))

    min_slope = 0
    slope_path = os.path.join(os.path.dirname(beta_path), "slope_f32.dat")
    slope_mm = np.memmap(slope_path, dtype=np.float32, mode='r', shape=(height, width))

    valid_all = (~np.isnan(slope_mm)) & (~np.isnan(flow_acc))
    Ps_L = np.zeros((height, width), dtype=np.float32)
    Ps_L[~valid_all] = np.nan

    steep_mask = valid_all & (slope_mm >= min_slope)

    if np.any(steep_mask):
        labels_flat = slope_units.ravel()
        beta_flat = beta_mm.ravel()
        acc_flat = flow_acc.ravel()
        steep_flat = steep_mask.ravel()

        steep_indices = np.flatnonzero(steep_flat)
        if steep_indices.size > 0:
            steep_beta = beta_flat[steep_indices].astype(np.float64)
            steep_acc = acc_flat[steep_indices].astype(np.float64)

            if steep_indices.size > 1:
                rank_beta = rankdata(steep_beta, method='average')
                rank_acc = rankdata(steep_acc, method='average')
                cdf_beta = (rank_beta - 1) / (steep_indices.size - 1)
                cdf_acc = (rank_acc - 1) / (steep_indices.size - 1)
                joint_steep = (cdf_beta * cdf_acc).astype(np.float32)
            else:
                joint_steep = np.zeros(steep_indices.size, dtype=np.float32)

            steep_labels = labels_flat[steep_indices]
            valid_unit_mask = steep_labels > 0

            if np.any(valid_unit_mask):
                valid_steep_labels = steep_labels[valid_unit_mask]
                valid_joint = joint_steep[valid_unit_mask]

                unit_values = np.zeros(labels_flat.max() + 1, dtype=np.float32)
                for ul, val in zip(valid_steep_labels, valid_joint):
                    if val > unit_values[ul]:
                        unit_values[ul] = val

                Ps_L_flat = Ps_L.ravel()
                for idx, label in enumerate(labels_flat):
                    if label > 0:
                        Ps_L_flat[idx] = unit_values[label]
                Ps_L = Ps_L_flat.reshape((height, width))

    if gaussian_sigma is not None and gaussian_sigma > 0:
        print_progress(f"Applying Gaussian smoothing (sigma={gaussian_sigma})...")
        nan_mask = np.isnan(Ps_L)
        Ps_L_filled = Ps_L.copy()
        Ps_L_filled[nan_mask] = 0
        Ps_L_smooth = gaussian_filter(Ps_L_filled, sigma=gaussian_sigma)
        Ps_L_smooth[nan_mask] = np.nan
        Ps_L = Ps_L_smooth

    out_path = os.path.join(output_folder, "Morphometric_Susceptibility.tif")
    save_raster_from_array(Ps_L, profile, out_path)
    print_progress(f"Saved final: {out_path}")
    if write_intermediates:
        print_progress(f"Intermediate saved to {inter_dir}")
    return out_path

# ----------------------------
# main
# ----------------------------
if __name__ == "__main__":
    # ====== 用户参数 ======
    # dem_path = r"Data/DEM_test.TIF"      # 输入DEM路径
    dem_path = r"D:\lb\myData\RSE_R3\Data\DEM.tif"
    output_folder = r"D:\lb\myCode\LandslideNet\Result\All_LSMs\Morphometric"        # 输出文件夹
    tile_size = 2048                 # 分块大小
    overlap = 64                     # 分块重叠：用于减少边界效应（建议 >= 32）
    # gaussian_sigma = None            # 高斯平滑sigma（None=禁用）
    gaussian_sigma = 2.0             # 高斯平滑sigma（None=禁用）

    tmp_dir = os.path.join(output_folder, "tmp")
    ensure_dir(tmp_dir)

    # Step 1: Compute slope and flow direction (tile-level, with lightweight fill)
    flow_dir_path, slope_path, profile, nodata, height, width = compute_tiled_slope_and_flowdir(
        dem_path, tmp_dir, tile_size, overlap
    )

    # Step 2: Compute β (mean drainage area slope) + flow accumulation
    flow_acc_path, slope_units_path, beta_path, receiver_path = compute_global_receiver_and_acc(
        flow_dir_path, slope_path, tmp_dir, profile, nodata, height, width
    )

    # Step 3: Compute morphometric susceptibility using β
    compute_morphometric_and_save(
        flow_acc_path, beta_path, slope_units_path, profile, nodata, output_folder,
        gaussian_sigma=gaussian_sigma
    )
