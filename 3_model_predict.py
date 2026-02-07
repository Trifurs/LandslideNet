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
    """
    对大尺度影像进行滑动窗口预测，使用批量处理，并将预测结果保存到以 overlap_size 命名的文件中。
    """
    model.eval()

    os.makedirs(output_dir, exist_ok=True)
    factor_files = sorted([os.path.join(factors_dir, f) for f in os.listdir(factors_dir) if f.endswith('.tif')])

    try:
        # 一次性打开所有因子文件，并在预测结束时关闭
        src_factors = [rasterio.open(f) for f in factor_files]
    except rasterio.RasterioIOError as e:
        raise FileNotFoundError(f"Error opening factor files: {e}")
        
    if not src_factors:
        raise FileNotFoundError(f"No .tif factor files found in {factors_dir}")

    # 获取影像基本信息
    src_profile = src_factors[0].profile.copy() # 保存 profile 以供最终写入使用
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
    
    # 批量处理 Patch
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
            
            # 填充到 crop_size
            pad_h = crop_size - win_h
            pad_w = crop_size - win_w
            factors_patch = np.pad(factors_patch,
                                   ((0,0), (0, pad_h), (0, pad_w)),
                                   mode='constant')
            
            batch_inputs.append(torch.tensor(factors_patch, dtype=torch.float32).unsqueeze(0))
        
        # 批量输入模型
        inputs = torch.cat(batch_inputs, dim=0).to(device)

        # 推理
        with torch.no_grad():
            outputs = model(inputs) # outputs: [B, 2, H, W]
            
            # 语义分割模型输出概率 (取类别 1 的概率)
            probabilities = F.softmax(outputs, dim=1)[:, 1, :, :].cpu().numpy() # [B, H, W]

        # 写回结果
        for idx, win_info in enumerate(batch_info):
            prob_map = probabilities[idx] # [H, W]
            r, c, h, w = win_info['r'], win_info['c'], win_info['h'], win_info['w']
            
            # 只取未填充的有效区域
            prob_to_write = prob_map[:h, :w]
            
            # 将预测值存入全局数组
            prediction_data[r:r+h, c:c+w] = prob_to_write

    # 关闭文件句柄
    for src in src_factors:
        src.close()

    # --- 保存最终输出文件 ---
    final_output_path = os.path.join(output_dir, f'landslide_sensitivity_map_{overlap_size}.tif')

    # 读取基础掩膜（需要重新打开文件）
    with rasterio.open(factor_files[0]) as mask_src:
        global_mask = (mask_src.read(1) != mask_src.nodata)
    
    # 写入最终结果，并应用全局掩膜
    profile = src_profile.copy() # 使用保存的 profile
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

    # 【修正 1】: 在此处打开文件，获取 profile 和基本信息，并立即关闭
    with rasterio.open(input_files[0]) as src:
        # 保存 profile 以供最终文件写入使用
        profile = src.profile.copy() 
        transform = src.transform
        crs = src.crs
        width = src.width
        height = src.height
        nodata = src.nodata
        # 获取基础掩膜，用于后续判断有效像素
        base_mask = (src.read(1) != nodata) 
    # src 对象在此处被关闭

    final_output = np.full((height, width), nodata, dtype=np.float32)
    
    # 使用 nanmedian 进行融合
    for y in tqdm(range(height), desc="Processing rows for Mosaicking"):
        row_stack = np.full((width, len(input_files)), np.nan, dtype=np.float32)
        
        for i, file in enumerate(input_files):
            # 确保在每次迭代中都打开和关闭文件
            with rasterio.open(file) as src_row: 
                # 使用读取窗口，只读取当前行
                row_data = src_row.read(1, window=Window(0, y, width, 1))[0]
                
                # 仅考虑非 nodata 且在 base_mask 内的有效像素
                valid_mask = (row_data != src_row.nodata) & base_mask[y, :]
                row_stack[:, i] = np.where(valid_mask, row_data, np.nan)
        
        # 计算每一行的中位数
        median_row = np.nanmedian(row_stack, axis=1)
        
        # 将中位数结果写回最终输出数组
        final_output[y, :] = median_row

    # 最终结果处理
    final_output = np.where(base_mask, final_output, nodata) 
    
    # 统计模块
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
    # 【修正 2】: 使用前面保存的 profile 变量
    profile.update(dtype=rasterio.float32, count=1, nodata=nodata)
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(final_output, 1)

def main(train_output, factors_dir, lsm_dir, device_ids, batch_size, crop_size, num_bands, mosaic_map):
    
    model = LandslideNet(int(num_bands))
    
    # 1. 加载状态字典
    state_dict = torch.load(os.path.join(train_output, 'best_model_weight.pth'))
    
    # 2. 移除 'module.' 前缀（如果存在），确保可以加载到裸模型中
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            # 剥离 'module.' 前缀
            new_state_dict[k[7:]] = v 
        else:
            new_state_dict[k] = v
            
    # 3. 将权重加载到 LandslideNet 裸模型中
    model.load_state_dict(new_state_dict) 
    
    # 4. 包装 DataParallel 并移至 GPU
    device_list = list(map(int, device_ids.strip('[]').split(',')))
    model = nn.DataParallel(model, device_ids=device_list).cuda()
    
    # 参数转换
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
