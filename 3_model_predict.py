import os
import sys
import rasterio
import xml.etree.ElementTree as ET
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from rasterio.windows import Window
import csv

from utils import LandslideNet

def get_argv(xml_file):
    argv_names = [
        'train_output', 
        'input_factors_dir',
        'LSM_dir',
        'device_ids',
        'batch_size',
        'crop_size',
        'num_bands',
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
                break
    return argv_values

def predict_landslide_sensitivity(model, factors_dir, output_dir, crop_size=512, overlap_size=128, batch_size=32):
    model.eval()

    os.makedirs(output_dir, exist_ok=True)
    factor_files = sorted([os.path.join(factors_dir, f) for f in os.listdir(factors_dir) if f.endswith('.tif')])

    try:
        src_factors = [rasterio.open(f) for f in factor_files]
    except rasterio.RasterioIOError as e:
        raise FileNotFoundError(f"Error opening factor files: {e}")
        
    if not src_factors:
        raise FileNotFoundError(f"No .tif factor files found in {factors_dir}")

    src_profile = src_factors[0].profile.copy()
    width = src_profile['width']
    height = src_profile['height']
    nodata = src_profile.get('nodata', -9999.0)
    
    prediction_data = np.full((height, width), nodata, dtype=np.float32)

    # --- Window Calculation and Batching Setup ---
    windows_info = []
    stride = crop_size - overlap_size
    
    rows = list(range(0, height - crop_size + stride, stride))
    cols = list(range(0, width - crop_size + stride, stride))
    
    if rows and rows[-1] < height - crop_size: rows.append(height - crop_size)
    if cols and cols[-1] < width - crop_size: cols.append(width - crop_size)
    
    rows = sorted(list(set(rows)))
    cols = sorted(list(set(cols)))

    for row in rows:
        for col in cols:
            win_h = min(crop_size, height - row)
            win_w = min(crop_size, width - col)
            if win_h > 0 and win_w > 0:
                windows_info.append({
                    'r': row,
                    'c': col,
                    'h': win_h,
                    'w': win_w,
                    'window': Window(col, row, win_w, win_h)
                })

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pbar = tqdm(range(0, len(windows_info), batch_size), desc=f"Predicting Overlap {overlap_size}")
    
    for i in pbar:
        batch_info = windows_info[i:i + batch_size]
        batch_inputs = []

        for win_info in batch_info:
            win_h, win_w = win_info['h'], win_info['w']
            window = win_info['window']
            
            factors = []
            for src in src_factors:
                data = src.read(1, window=window)
                data = np.nan_to_num(data, nan=0.0)
                data = np.clip(data, 0, 1)
                factors.append(data)
            
            factors_patch = np.stack(factors, axis=0)
            
            pad_h = crop_size - win_h
            pad_w = crop_size - win_w
            factors_patch = np.pad(factors_patch,
                                   ((0,0), (0, pad_h), (0, pad_w)),
                                   mode='constant')
            
            batch_inputs.append(torch.tensor(factors_patch, dtype=torch.float32).unsqueeze(0))
        
        inputs = torch.cat(batch_inputs, dim=0).to(device)

        with torch.no_grad():
            outputs = model(inputs) # outputs: [B, 2, H, W]
            
            probabilities = F.softmax(outputs, dim=1)[:, 1, :, :].cpu().numpy() # [B, H, W]

        for idx, win_info in enumerate(batch_info):
            prob_map = probabilities[idx] # [H, W]
            r, c, h, w = win_info['r'], win_info['c'], win_info['h'], win_info['w']

            prob_to_write = prob_map[:h, :w]

            prediction_data[r:r+h, c:c+w] = prob_to_write

    for src in src_factors:
        src.close()

    final_output_path = os.path.join(output_dir, f'landslide_sensitivity_map_{overlap_size}.tif')

    with rasterio.open(factor_files[0]) as mask_src:
        global_mask = (mask_src.read(1) != mask_src.nodata)
    
    profile = src_profile.copy() 
    profile.update(dtype=rasterio.float32, count=1, nodata=nodata)

    with rasterio.open(final_output_path, 'w', **profile) as final_dst:
        final_data = np.where(global_mask, prediction_data, nodata).astype(np.float32)
        final_dst.write(final_data, 1)
        
    print(f"Prediction map saved for overlap {overlap_size} at: {final_output_path}")

def mosaic_landslide_sensitivity_maps(input_dir, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    input_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.tif')])
    
    if not input_files:
        print("Warning: No sensitivity maps found for mosaicking.")
        return

    with rasterio.open(input_files[0]) as src:
        profile = src.profile.copy() 
        transform = src.transform
        crs = src.crs
        width = src.width
        height = src.height
        nodata = src.nodata
        base_mask = (src.read(1) != nodata) 

    final_output = np.full((height, width), nodata, dtype=np.float32)
    
    for y in tqdm(range(height), desc="Processing rows for Mosaicking"):
        row_stack = np.full((width, len(input_files)), np.nan, dtype=np.float32)
        
        for i, file in enumerate(input_files):
            with rasterio.open(file) as src_row: 
                row_data = src_row.read(1, window=Window(0, y, width, 1))[0]
                
                valid_mask = (row_data != src_row.nodata) & base_mask[y, :]
                row_stack[:, i] = np.where(valid_mask, row_data, np.nan)
        
        median_row = np.nanmedian(row_stack, axis=1)
        
        final_output[y, :] = median_row

    final_output = np.where(base_mask, final_output, nodata) 
    
    valid_pixels = final_output[base_mask]
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ["Very Low", "Low", "Medium", "High", "Very High"]
    hist, _ = np.histogram(valid_pixels, bins=bins)
    
    total = valid_pixels.size
    stats = []
    for i in range(len(labels)):
        count = hist[i]
        percentage = count / total * 100
        stats.append({
            "class": labels[i],
            "count": count,
            "percentage": f"{percentage:.2f}%"
        })
    
    # Generate CSV
    csv_path = os.path.splitext(output_path)[0] + '.csv'
    csv_path = csv_path.replace('landslide_sensitivity_landslidecnn_mosaic', 'risk_stats')
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["Risk Level", "Pixel Count", "Percentage"])
        writer.writeheader()
        for item in stats:
            writer.writerow({
                "Risk Level": item["class"],
                "Pixel Count": item["count"],
                "Percentage": item["percentage"]
            })
    
    # Console output
    print("\nLandslide Risk Level Statistics:")
    for item in stats:
        print(f"{item['class']} Risk Area: {item['count']} pixels ({item['percentage']})")
    
    # Maintain original file writing
    profile.update(dtype=rasterio.float32, count=1, nodata=nodata)
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(final_output, 1)

def main(train_output, factors_dir, lsm_dir, device_ids, batch_size, crop_size, num_bands, mosaic_map):
    
    model = LandslideNet(int(num_bands))
    
    state_dict = torch.load(os.path.join(train_output, 'best_model_weight.pth'))
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v 
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict) 
    
    device_list = list(map(int, device_ids.strip('[]').split(',')))
    model = nn.DataParallel(model, device_ids=device_list).cuda()
    
    batch_size = int(batch_size)
    crop_size = int(crop_size)
    
    overlap_list = [20, 43, 71, 111, 147]
    list_index = 1
    for overlap_size in overlap_list:
        print(f"\n{list_index}/{len(overlap_list)}: Starting prediction with overlap size {overlap_size}") 
        
        predict_landslide_sensitivity(
            model, factors_dir, lsm_dir, 
            crop_size, overlap_size, 
            batch_size
        )
        list_index += 1

    mosaic_landslide_sensitivity_maps(lsm_dir, mosaic_map)

if __name__ == '__main__':
    try:
        if len(sys.argv) < 2:
            raise RuntimeError("Missing XML config file parameter")
            
        config_path = sys.argv[1]
        params = get_argv(config_path)
        
        if len(params) != 8:
            raise ValueError(f"Mismatched configuration parameters count. Expected 8, got {len(params)}.")
            
        main(*params)
        
        print('<predict_status>0</predict_status>')
        print('<predict_log>success</predict_log>')
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        error_msg = str(e).replace('\n', ' ').replace('\t', ' ')
        print('<predict_status>1</predict_status>')
        print(f'<predict_log>{error_msg}</predict_log>')

