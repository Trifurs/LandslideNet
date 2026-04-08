import os
import numpy as np
import rasterio
from tqdm import tqdm
from scipy.stats import gaussian_kde
from sklearn.preprocessing import MinMaxScaler
from multiprocessing import Pool, cpu_count, RawArray
import warnings
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import psutil
import jenkspy
import gc
import time

warnings.filterwarnings("ignore")

dtype = np.float32
output_nodata = -9999
base_block_size = 256
n_workers = max(1, cpu_count() - 2)
memmap_threshold = 4
random_seed = 42

theta_min = 0.55
n_strata = 3
pos_neg_ratio = 1.0

weight_power = 1
min_sample_per_stratum = 5

landslide_path = r"D:\lb\myCode\Landslide_susceptibility_mapping\R3\data\Landslide.tif"
factor_folder = r"D:\lb\myCode\LandslideNet\Data\origin\factors"
output_folder = r"D:\lb\myCode\LandslideNet\Data\origin\DWSS"

os.makedirs(output_folder, exist_ok=True)
np.random.seed(random_seed)

shared_proto_data = None
shared_factors_shape = None
shared_factors_mmap_path = None

def log_memory_usage(desc):
    mem = psutil.virtual_memory()
    print(f"{desc} - Memory Usage: {mem.used/1024**3:.1f}GB / {mem.total/1024**3:.1f}GB")

def auto_block_size(shape):
    element_size = np.dtype(dtype).itemsize
    available_mem = psutil.virtual_memory().available * 0.4
    max_block_mem = 500 * 1024 * 1024
    max_block = int(np.sqrt(max_block_mem / (shape[2] * element_size)))
    return min(max_block, base_block_size)

def create_memmap_array(shape, name):
    mmap_path = os.path.join(output_folder, f"{name}.dat")
    if os.path.exists(mmap_path):
        safe_remove_file(mmap_path)
    return np.memmap(mmap_path, dtype=dtype, mode='w+', shape=shape)

def safe_remove_file(file_path, max_retries=5, delay=1):
    for i in range(max_retries):
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
            return True
        except PermissionError:
            if i < max_retries - 1:
                time.sleep(delay)
                gc.collect()
            else:
                print(f"Warning: Failed to delete temporary file {file_path}, please clean it manually")
                return False

def load_prototype_data():
    with rasterio.open(landslide_path) as src:
        landslide = src.read(1)
        meta = src.meta.copy()
        nodata = src.nodata
        
        valid_mask = (landslide != nodata) & (~np.isnan(landslide))
        prototype_mask = (landslide == 1) & valid_mask
        
        n_prototype = np.sum(prototype_mask)
        print(f"Number of valid landslide prototype points: {n_prototype}")
        if n_prototype == 0:
            raise ValueError("No valid landslide prototype points found, please check input landslide data")
            
    return prototype_mask, meta, valid_mask, nodata, landslide

def load_environmental_factors():
    factor_files = sorted([f for f in os.listdir(factor_folder) if f.endswith('.tif')])
    if len(factor_files) == 0:
        raise ValueError("No tif format data found in environmental factor folder")
    print(f"Number of detected environmental factors: {len(factor_files)}")
    
    with rasterio.open(os.path.join(factor_folder, factor_files[0])) as src:
        base = src.read(1)
        rows, cols = base.shape
        transform = src.transform
        crs = src.crs

    factors = create_memmap_array((rows, cols, len(factor_files)), 'factors')
    global_valid = np.ones((rows, cols), dtype=bool)
    
    for idx, f in enumerate(tqdm(factor_files, desc="Loading environmental factors")):
        with rasterio.open(os.path.join(factor_folder, f)) as src:
            data = src.read(1).astype(dtype)
            nodata = src.nodata
            
            valid = (data != nodata) & (~np.isnan(data))
            data[~valid] = np.nan
            
            col_means = np.nanmean(data, axis=0)
            global_col_mean = np.nanmean(col_means)
            col_means = np.where(np.isnan(col_means), global_col_mean, col_means)
            data = np.where(np.isnan(data), col_means[None, :], data)
            
            scaler = MinMaxScaler(feature_range=(0, 1))
            factors[:, :, idx] = scaler.fit_transform(data.reshape(-1, 1)).reshape(data.shape)
            global_valid &= valid
    
    factors.flush()
    log_memory_usage("Environmental factors loading completed")
    return factors, global_valid, transform, crs, factor_files

def init_worker(proto_data_raw, factors_shape, factors_mmap_path):
    global shared_proto_data, shared_factors_shape, shared_factors_mmap_path
    shared_proto_data = np.frombuffer(proto_data_raw, dtype=dtype).reshape(-1, factors_shape[2])
    shared_factors_shape = factors_shape
    shared_factors_mmap_path = factors_mmap_path
    global shared_kde
    shared_kde = gaussian_kde(shared_proto_data.T, bw_method='scott')

def parallel_kde_worker(args):
    i, j, block_size = args
    rows, cols, n_factors = shared_factors_shape
    
    try:
        factors = np.memmap(shared_factors_mmap_path, dtype=dtype, mode='r', shape=shared_factors_shape)
        i_end = min(i + block_size, rows)
        j_end = min(j + block_size, cols)
        block = factors[i:i_end, j:j_end, :].copy()
        del factors
        gc.collect()
        
        valid_mask = ~np.isnan(block).any(axis=2)
        result = np.zeros(block.shape[:2], dtype=dtype)
        
        if valid_mask.sum() > 0:
            flat_data = block[valid_mask].T
            densities = shared_kde.evaluate(flat_data)
            result[valid_mask] = densities
        
        return (i, j, i_end, j_end, result)
    
    except Exception as e:
        print(f"Failed to process block ({i},{j})-({i_end},{j_end}): {str(e)}")
        return (i, j, i_end, j_end, np.zeros((i_end-i, j_end-j), dtype=dtype))

def calculate_similarity_divergence(factors, proto_mask):
    proto_data = factors[proto_mask].copy()
    valid_proto = ~np.isnan(proto_data).any(axis=1)
    proto_data = proto_data[valid_proto].astype(dtype)
    
    if proto_data.size == 0:
        raise ValueError("Valid prototype data is empty, please check spatial matching between landslide data and environmental factors")
    
    print(f"Prototype data statistics - Factors: {proto_data.shape[1]}, Valid samples: {proto_data.shape[0]}")
    
    block_size = auto_block_size(factors.shape)
    print(f"Auto block size: {block_size}x{block_size}")
    
    rows, cols, n_factors = factors.shape
    tasks = [(i, j, block_size) for i in range(0, rows, block_size) for j in range(0, cols, block_size)]
    
    proto_data_raw = RawArray('f', proto_data.flatten())
    
    print("Starting parallel calculation of prototype similarity (KDE fitted only once)...")
    with Pool(
        processes=n_workers,
        initializer=init_worker,
        initargs=(proto_data_raw, factors.shape, factors.filename)
    ) as pool:
        results = list(tqdm(
            pool.imap_unordered(parallel_kde_worker, tasks, chunksize=2),
            total=len(tasks),
            desc="Similarity calculation progress"
        ))
    
    similarity = np.full((rows, cols), np.nan, dtype=dtype)
    for i, j, i_end, j_end, block in results:
        similarity[i:i_end, j:j_end] = block
    
    max_density = np.nanmax(similarity)
    similarity = similarity / (max_density + 1e-8)
    similarity = np.clip(similarity, 0, 1)
    
    divergence = 1 - similarity
    divergence = np.clip(divergence, 0, 1)
    
    factors.flush()
    del factors, proto_data_raw, proto_data
    gc.collect()
    
    log_memory_usage("Similarity and divergence calculation completed")
    return similarity, divergence

def dwss_sampling(divergence, proto_mask, global_valid, landslide_data, nodata_value):
    pos_points = np.argwhere(proto_mask)
    M_total = len(pos_points)
    n_neg_total = int(M_total * pos_neg_ratio)
    print(f"Total positive samples: {M_total}, Target total negative samples: {n_neg_total}")

    candidate_mask = (landslide_data != 1) & (landslide_data != nodata_value) & global_valid & (divergence >= theta_min)
    candidate_divergence = divergence[candidate_mask].copy()
    candidate_coords = np.argwhere(candidate_mask)
    
    n_candidate = len(candidate_divergence)
    print(f"Negative sample candidate pool size: {n_candidate}")
    if n_candidate < n_neg_total:
        raise ValueError(f"Insufficient negative sample candidates, need {n_neg_total}, only {n_candidate} available, please lower theta_min threshold")

    print(f"Dividing into {n_strata} divergence strata using Jenks natural breaks...")
    sample_size = min(200000, len(candidate_divergence))
    sample_for_break = np.random.choice(candidate_divergence, size=sample_size, replace=False)
    breaks = jenkspy.jenks_breaks(sample_for_break, n_classes=n_strata)
    print(f"Natural break stratification thresholds: {[round(b,4) for b in breaks]}")

    strata_masks = []
    strata_mean_divergence = []
    for k in range(n_strata):
        if k == 0:
            strata_mask = candidate_divergence <= breaks[k+1]
        elif k == n_strata-1:
            strata_mask = candidate_divergence > breaks[k]
        else:
            strata_mask = (candidate_divergence > breaks[k]) & (candidate_divergence <= breaks[k+1])
        
        strata_masks.append(strata_mask)
        mean_div = np.mean(candidate_divergence[strata_mask]) if strata_mask.sum() > 0 else 0
        strata_mean_divergence.append(mean_div)
        print(f"Stratum {k+1} - Samples: {strata_mask.sum()}, Mean divergence: {mean_div:.4f}")

    total_guaranteed = n_strata * min_sample_per_stratum
    if total_guaranteed > n_neg_total:
        raise ValueError(f"Total guaranteed samples {total_guaranteed} exceed target negative samples {n_neg_total}, please lower min_sample_per_stratum")
    remaining_samples = n_neg_total - total_guaranteed
    
    strata_weights = np.array([d ** weight_power for d in strata_mean_divergence])
    sum_weights = strata_weights.sum()
    if sum_weights <= 0:
        raise ValueError("Total stratum weights is 0, cannot calculate sample counts")
    
    strata_sample_counts = []
    for k in range(n_strata):
        base_count = min_sample_per_stratum
        weight_count = round(remaining_samples * (strata_weights[k] / sum_weights))
        total_count = base_count + weight_count
        strata_sample_counts.append(total_count)
    
    count_diff = n_neg_total - sum(strata_sample_counts)
    if count_diff != 0:
        adjust_idx = n_strata - 1
        strata_sample_counts[adjust_idx] += count_diff
    print(f"Final sample counts per stratum: {strata_sample_counts}")

    neg_samples = []
    for k in range(n_strata):
        strata_coords = candidate_coords[strata_masks[k]]
        n_sample = strata_sample_counts[k]
        
        if n_sample <= 0:
            continue
        if len(strata_coords) < n_sample:
            raise ValueError(f"Insufficient samples in stratum {k+1}, need {n_sample}, only {len(strata_coords)} available")
        
        sample_idx = np.random.choice(len(strata_coords), size=n_sample, replace=False)
        neg_samples.append(strata_coords[sample_idx])
    
    neg_samples = np.concatenate(neg_samples, axis=0)
    print(f"DWSS negative sampling completed, final count: {len(neg_samples)}")

    label_raster = np.full_like(landslide_data, 0, dtype=np.int16)
    label_raster[proto_mask] = 1
    label_raster[neg_samples[:, 0], neg_samples[:, 1]] = 2
    label_raster[landslide_data == nodata_value] = nodata_value

    del candidate_divergence, candidate_coords
    gc.collect()
    
    return label_raster, neg_samples, pos_points

def save_raster(data, meta, filename):
    save_meta = meta.copy()
    save_meta.update({
        'dtype': dtype if data.dtype == np.float32 else data.dtype,
        'nodata': output_nodata if data.dtype == np.float32 else meta['nodata'],
        'count': 1,
        'compress': 'lzw'
    })
    
    with rasterio.open(os.path.join(output_folder, filename), 'w', **save_meta) as dst:
        for i in range(0, data.shape[0], base_block_size):
            i_end = min(i + base_block_size, data.shape[0])
            block = data[i:i_end]
            block = np.where(np.isnan(block), save_meta['nodata'], block)
            dst.write(block, 1, window=((i, i_end), (0, data.shape[1])))
    print(f"Raster saved: {filename}")

def save_vector_samples(pos_points, neg_samples, transform, crs):
    def coords_to_gdf(points, label):
        if len(points) == 0:
            return gpd.GeoDataFrame()
        coords = [Point(transform * (j+0.5, i+0.5)) for i, j in points]
        return gpd.GeoDataFrame(
            geometry=coords,
            data={'label': [label]*len(points), 'type': ['landslide' if label==1 else 'non-landslide']*len(points)},
            crs=crs
        )

    gdf = gpd.GeoDataFrame(pd.concat([coords_to_gdf(pos_points, 1), coords_to_gdf(neg_samples, 2)], ignore_index=True))
    shp_path = os.path.join(output_folder, "dwss_samples.shp")
    if not gdf.empty:
        gdf.to_file(shp_path, encoding='utf-8')
        print(f"Vector sample file saved: dwss_samples.shp")
    del gdf
    gc.collect()

def apply_global_mask(global_valid, meta):
    for fname in os.listdir(output_folder):
        if fname.endswith('.tif') and fname != 'final_labels.tif':
            path = os.path.join(output_folder, fname)
            with rasterio.open(path, 'r+') as src:
                data = src.read(1)
                data[~global_valid] = output_nodata
                src.write(data, 1)
            gc.collect()

if __name__ == "__main__":
    temp_files = []
    try:
        proto_mask, meta, landslide_valid, nodata_value, landslide_data = load_prototype_data()
        
        factors, env_valid, transform, crs, factor_files = load_environmental_factors()
        temp_files.append(factors.filename)
        meta.update({'transform': transform, 'crs': crs})
        
        global_valid = landslide_valid & env_valid
        print(f"Global valid area ratio: {global_valid.mean()*100:.2f}%")
        
        similarity, divergence = calculate_similarity_divergence(factors, proto_mask)
        
        save_raster(similarity, meta, "landslide_reliability.tif")
        save_raster(divergence, meta, "non_landslide_reliability.tif")
        save_raster(divergence, meta, "divergence_degree.tif")
        
        del similarity
        gc.collect()
        
        label_raster, neg_samples, pos_points = dwss_sampling(
            divergence, proto_mask, global_valid, landslide_data, nodata_value
        )
        
        label_meta = meta.copy()
        label_meta.update({'dtype': 'int16', 'nodata': nodata_value})
        save_raster(label_raster, label_meta, "final_labels.tif")
        
        save_vector_samples(pos_points, neg_samples, transform, crs)
        
        apply_global_mask(global_valid, meta)
        
        gc.collect()
        time.sleep(2)
        print("Starting to clean temporary files...")
        for temp_file in temp_files:
            safe_remove_file(temp_file)
        for fname in os.listdir(output_folder):
            if fname.endswith('.dat') and '_old_' in fname:
                safe_remove_file(os.path.join(output_folder, fname))
        
        print("="*60)
        print(f"DWSS sampling processing completed! Results saved in: {output_folder}")
        print(f"Number of positive samples: {len(pos_points)}")
        print(f"umber of negative samples: {len(neg_samples)}")
        print(f"Statistics reverted to mean divergence, weight power: {weight_power}")
        print("="*60)
        
    except Exception as e:
        print(f"Processing failed: {str(e)}")
        gc.collect()
        time.sleep(2)
        for temp_file in temp_files:
            safe_remove_file(temp_file)
        for fname in os.listdir(output_folder):
            if fname.endswith('.dat') and '_old_' in fname:
                safe_remove_file(os.path.join(output_folder, fname))
        raise
