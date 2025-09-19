import os
import sys
import rasterio
import xml.etree.ElementTree as ET
import numpy as np
import torch
import torch.nn as nn
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
    return argv_values

def predict_landslide_sensitivity(model, factors_dir, output_dir, crop_size=512, overlap_size=128, batch_size=32, device_ids=[0, 1]):
    model = nn.DataParallel(model, device_ids=list(map(int, device_ids)))
    model = model.cuda()
    model.eval()

    os.makedirs(output_dir, exist_ok=True)
    factor_files = sorted([os.path.join(factors_dir, f) for f in os.listdir(factors_dir) if f.endswith('.tif')])

    with rasterio.open(factor_files[0]) as src:
        transform = src.transform
        crs = src.crs
        width = src.width
        height = src.height

    temp_output = os.path.join(output_dir, 'temp_prediction.tif')
    final_output = os.path.join(output_dir, f'landslide_sensitivity_map_{overlap_size}.tif')

    with rasterio.open(temp_output, 'w', driver='GTiff',
                     height=height, width=width, count=1,
                     dtype='float32', crs=crs, transform=transform,
                     nodata=-9999) as temp_dst:
        
        for row in tqdm(range(0, height, crop_size - overlap_size)):
            for col in range(0, width, crop_size - overlap_size):
                win_h = min(crop_size, height - row)
                win_w = min(crop_size, width - col)
                window = Window(col, row, win_w, win_h)

                factors = []
                for factor_file in factor_files:
                    with rasterio.open(factor_file) as src:
                        data = src.read(1, window=window)
                        factors.append(np.nan_to_num(np.clip(data, 0, 1)))

                factors_batch = np.stack(factors, axis=0)
                factors_batch = np.pad(factors_batch,
                                      ((0,0), (0, crop_size-win_h), (0, crop_size-win_w)),
                                      mode='constant')
                factors_batch = torch.tensor(factors_batch, dtype=torch.float32).unsqueeze(0).cuda()

                with torch.no_grad():
                    prob = torch.sigmoid(model(factors_batch)[0,0]).cpu().numpy()[:win_h, :win_w]
                    temp_dst.write(prob.astype(np.float32), 1, window=window)

    with rasterio.open(factor_files[0]) as mask_src, \
         rasterio.open(temp_output) as src_temp:
        global_mask = (mask_src.read(1) != mask_src.nodata)
        temp_data = src_temp.read(1)
        
        with rasterio.open(final_output, 'w', driver='GTiff',
                          height=height, width=width, count=1,
                          dtype='float32', crs=crs, transform=transform,
                          nodata=-9999) as final_dst:
            final_dst.write(np.where(global_mask, temp_data, -9999).astype(np.float32), 1)

    os.remove(temp_output)

def mosaic_landslide_sensitivity_maps(input_dir, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    input_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.tif')])
    
    with rasterio.open(input_files[0]) as src:
        transform = src.transform
        crs = src.crs
        width = src.width
        height = src.height
        nodata = src.nodata
        base_mask = (src.read(1) != nodata)

    final_output = np.full((height, width), nodata, dtype=np.float32)
    
    for y in tqdm(range(height), desc="Processing rows"):
        row_stack = np.full((width, len(input_files)), np.nan)
        
        for i, file in enumerate(input_files):
            with rasterio.open(file) as src:
                row_data = src.read(1, window=Window(0, y, width, 1))[0]
                valid_mask = (row_data != src.nodata) & base_mask[y, :]
                row_stack[:, i] = np.where(valid_mask, row_data, np.nan)
        
        final_output[y, :] = np.nanmedian(row_stack, axis=1)

    final_output = np.where(base_mask, 1 - final_output, nodata)
    
    # Statistics module
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
    with rasterio.open(output_path, 'w', driver='GTiff',
                      height=height, width=width, count=1,
                      dtype=np.float32, crs=crs, transform=transform,
                      nodata=nodata) as dst:
        dst.write(final_output, 1)

def main(train_output, factors_dir, lsm_dir, device_ids, batch_size, crop_size, num_bands, mosaic_map):
    model = LandslideNet(int(num_bands))
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(os.path.join(train_output, 'best_model_weight.pth')))
    
    overlap_list = [55, 128, 80, 0, 199, 175]
    list_index = 1
    for overlap_size in overlap_list:
        print(f"{list_index}/{len(overlap_list)}: Processing with overlap size {overlap_size}")   
        predict_landslide_sensitivity(model, factors_dir, lsm_dir, 
                                     int(crop_size), overlap_size, 
                                     int(batch_size), 
                                     list(map(int, device_ids.strip('[]').split(','))))
        list_index += 1

    mosaic_landslide_sensitivity_maps(lsm_dir, mosaic_map)

if __name__ == '__main__':
    try:
        if len(sys.argv) < 2:
            raise RuntimeError("Missing XML config file parameter")
            
        config_path = sys.argv[1]
        params = get_argv(config_path)
        
        if len(params) != 8:
            raise ValueError("Mismatched configuration parameters count")
            
        main(*params)
        
        print('<predict_status>0</predict_status>')
        print('<predict_log>success</predict_log>')
        
    except Exception as e:
        error_msg = str(e).replace('\n', ' ').replace('\t', ' ')
        print('<predict_status>1</predict_status>')
        print(f'<predict_log>{error_msg}</predict_log>')
