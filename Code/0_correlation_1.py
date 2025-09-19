import os
import sys
import numpy as np
import pandas as pd
from osgeo import gdal
from scipy import stats
from tqdm import tqdm
from sklearn.feature_selection import mutual_info_regression
import dcor
import warnings
from multiprocessing import Pool
import argparse
import xml.etree.ElementTree as ET
from utils import *

# Suppress all warnings
warnings.filterwarnings("ignore")
os.environ['GDAL_NUM_THREADS'] = '1'  # Prevent GDAL multi-thread conflict

#========================= Correlation Functions =========================#
def valid_data(x, y):
    """Data cleaning function"""
    mask = (~np.isnan(x)) & (~np.isnan(y))
    return x[mask].astype(np.float64), y[mask].astype(np.float64)

def pearson(x, y):
    x_clean, y_clean = valid_data(x, y)
    return np.corrcoef(x_clean, y_clean)[0,1] if len(x_clean) > 0 else np.nan

def spearman(x, y):
    x_clean, y_clean = valid_data(x, y)
    return stats.spearmanr(x_clean, y_clean)[0] if len(x_clean) > 1 else np.nan

def kendall(x, y):
    x_clean, y_clean = valid_data(x, y)
    return stats.kendalltau(x_clean, y_clean)[0] if len(x_clean) > 1 else np.nan

def mutual_info(x, y):
    try:
        x_clean, y_clean = valid_data(x, y)
        return mutual_info_regression(x_clean.reshape(-1,1), y_clean)[0]
    except:
        return np.nan

def distance_corr(x, y):
    x_clean, y_clean = valid_data(x, y)
    return dcor.distance_correlation(x_clean, y_clean) if len(x_clean) > 0 else np.nan

def r2_score(x, y):
    try:
        x_clean, y_clean = valid_data(x, y)
        slope, intercept = np.polyfit(x_clean, y_clean, 1)
        y_pred = slope * x_clean + intercept
        ss_res = np.sum((y_clean - y_pred)**2)
        ss_tot = np.sum((y_clean - np.mean(y_clean))**2)
        return 1 - (ss_res / (ss_tot + 1e-9))
    except:
        return np.nan

CORRELATION_FUNCS = {
    'Pearson': pearson,
    'Spearman': spearman,
    'Kendall': kendall,
    'MutualInfo': mutual_info,
    'DistanceCorr': distance_corr,
    'R2_Score': r2_score
}

#========================= Core Processing =========================#
def process_single_file(args):
    """Process single file in parallel"""
    tif_file, result_path, factors_dir, factor_dirs = args
    try:
        result_arr = read_tif(os.path.join(result_path, tif_file))
        
        records = {name: {'tif_name': tif_file} for name in CORRELATION_FUNCS.keys()}
        
        for factor_subdir in factor_dirs:
            factor_path = os.path.join(factors_dir, factor_subdir, tif_file)
            if os.path.exists(factor_path):
                factor_arr = read_tif(factor_path)
                for func_name, func in CORRELATION_FUNCS.items():
                    try:
                        corr_value = func(result_arr.flatten(), factor_arr.flatten())
                        records[func_name][f"{func_name}_{factor_subdir}"] = corr_value
                    except Exception as e:
                        records[func_name][f"{func_name}_{factor_subdir}"] = np.nan
        
        return records
    except Exception as e:
        print(f"Error processing {tif_file}: {str(e)}")
        return {name: {'tif_name': tif_file} for name in CORRELATION_FUNCS.keys()}

def process_directory(result_subdir, result_dir, factors_dir, factor_dirs):
    """Process single subdirectory"""
    result_path = os.path.join(result_dir, result_subdir)
    tif_files = [f for f in os.listdir(result_path) if f.endswith('.tif')]
    
    args_list = [(f, result_path, factors_dir, factor_dirs) for f in tif_files]
    
    with Pool(processes=2) as pool:
        results = list(tqdm(pool.imap(process_single_file, args_list), 
                          total=len(tif_files), 
                          desc=f"Processing {result_subdir}"))
    
    merged_records = {name: [] for name in CORRELATION_FUNCS.keys()}
    for file_records in results:
        for name in CORRELATION_FUNCS.keys():
            merged_records[name].append(file_records[name])
    
    return merged_records

def process_tifs(result_dir, factors_dir, correlation_dir):
    """Main processing function"""
    os.makedirs(correlation_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    factor_dirs = [d for d in os.listdir(factors_dir)
                  if os.path.isdir(os.path.join(factors_dir, d))]
    
    result_tables = {name: [] for name in CORRELATION_FUNCS.keys()}
    
    result_subdirs = [d for d in os.listdir(result_dir) 
                     if os.path.isdir(os.path.join(result_dir, d))]
    
    for result_subdir in tqdm(result_subdirs, desc="Main directories"):
        subdir_records = process_directory(result_subdir, result_dir, factors_dir, factor_dirs)
        for name in CORRELATION_FUNCS.keys():
            result_tables[name].extend(subdir_records[name])
    
    for func_name, data in tqdm(result_tables.items(), desc="Saving results"):
        df = pd.DataFrame(data)
        relevant_cols = ['tif_name'] + [c for c in df.columns if c.startswith(f"{func_name}_")]
        df = df[relevant_cols]
        output_path = os.path.join(correlation_dir, f"{func_name}_correlation.csv")
        df.to_csv(output_path, index=False)

#========================= Utility Functions =========================#
def read_tif(file_path):
    """Read TIFF file as numpy array"""
    ds = gdal.Open(file_path)
    if ds is None:
        raise ValueError(f"Cannot read file: {file_path}")
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    return arr.astype(np.float32)
    
def result_cut(input_lsm_dir, output_lsm_dir, crop_size, overlap):
    """Crop results for analysis"""
    crop_all_rasters(input_lsm_dir, output_lsm_dir, crop_size, overlap)
    move_black_images_in_all_subfolders(output_lsm_dir)
    move_missing_images_to_black(output_lsm_dir)

def get_argv(xml_file):
    """Read parameters from XML configuration"""
    argv_names = [
        'output_factors_dir',
        'correlation_table_dir',
        'LSM_cut_dir',
        'crop_size',
        'overlap',
        'mosaic_map'
    ]
    argv_values = []
    root = ET.parse(xml_file).getroot()
    for argv_name in argv_names:
        for parameter in root.findall('param'):
            name = parameter.find('name').text
            value = parameter.find('value').text
            if name == argv_name:
                argv_values.append(value)
    return argv_values

def main(output_factors_dir, correlation_table_dir, LSM_cut_dir, crop_size, overlap, mosaic_map):
    """Main function"""
    input_lsm_dir = os.path.dirname(mosaic_map)
    output_lsm_dir = LSM_cut_dir
    
    result_cut(input_lsm_dir, output_lsm_dir, int(crop_size), int(overlap))
    process_tifs(output_lsm_dir, output_factors_dir, correlation_table_dir)
    print("\n[+] Correlation analysis completed successfully.")

if __name__ == '__main__':
    try:
        if len(sys.argv) < 2:
            raise RuntimeError("Missing XML config file parameter")
            
        config_path = sys.argv[1]
        params = get_argv(config_path)
        
        if len(params) != 6:
            raise ValueError("Mismatched configuration parameters count")
            
        main(*params)
        
        print('<process_status>0</process_status>')
        print('<process_log>success</process_log>')
        
    except Exception as e:
        error_msg = str(e).replace('\n', ' ').replace('\t', ' ')
        print('<process_status>1</process_status>')
        print(f'<process_log>{error_msg}</process_log>')
