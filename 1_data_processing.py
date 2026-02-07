import sys
import os
import rasterio
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
import random
import shutil
import multiprocessing
from rasterio.windows import Window 

# 定义分块处理的大小，单位：像素。
BLOCK_SIZE = 2048 

# 定义安全的进程数上限
NUM_PROCESSES = min(multiprocessing.cpu_count(), 16) 

def get_argv(xml_file):
    """提取预处理参数，新增 output_labels_dir"""
    argv_names = [
        'input_factors_dir', 'input_labels_dir', 
        'output_factors_dir', 'output_labels_dir', # 新增标签输出目录
        'crop_size'
    ]
    argv_values = []
    root = ET.parse(xml_file).getroot()
    for argv_name in argv_names:
        for param in root.findall('param'):
            if param.find('name').text == argv_name:
                argv_values.append(param.find('value').text)
                break
        else:
            raise ValueError(f"Parameter {argv_name} not found")
    return argv_values

def find_factor_files(factors_dir):
    """Helper to find all factor TIFF files."""
    factor_files = []
    
    # 尝试在顶层目录查找
    direct_tifs = sorted([os.path.join(factors_dir, f) for f in os.listdir(factors_dir) if f.endswith('.tif')])
    if len(direct_tifs) > 0:
        factor_files = direct_tifs
    else:
        # 尝试在子目录查找（兼容旧结构）
        subdirs = sorted([d for d in os.listdir(factors_dir) if os.path.isdir(os.path.join(factors_dir, d))])
        for subdir in subdirs:
            subdir_path = os.path.join(factors_dir, subdir)
            tifs = sorted([f for f in os.listdir(subdir_path) if f.endswith('.tif')])
            if tifs:
                factor_files.append(os.path.join(subdir_path, tifs[0])) 
            
    if not factor_files:
        raise FileNotFoundError(f"No .tif factor files found in {factors_dir}")
        
    return factor_files

# --- 新的多进程工作函数，用于 FCN 数据集 ---
def save_single_patch_fcn(args):
    """
    工作函数：从Block数据中提取因子Patch和标签Patch (Label Mask)，并保存。
    """
    (block_factors_data, block_label_data, r_abs, c_abs, label, 
     crop_size, factors_save_dir, labels_save_dir, 
     block_row_start, block_col_start, height, width) = args
     
    half_size = crop_size // 2
    
    r_min_abs, r_max_abs = r_abs - half_size, r_abs + half_size
    c_min_abs, c_max_abs = c_abs - half_size, c_abs + half_size

    # 边界检查
    if r_min_abs < 0 or c_min_abs < 0 or r_max_abs > height or c_max_abs > width:
        return 0

    # 计算 Patch 在 block_factors_data 中的相对坐标
    r_min_rel = r_min_abs - block_row_start
    r_max_rel = r_max_abs - block_row_start
    c_min_rel = c_min_abs - block_col_start
    c_max_rel = c_max_abs - block_col_start

    # 1. 提取因子 Patch (C, H, W)
    patch_factors = block_factors_data[:, r_min_rel:r_max_rel, c_min_rel:c_max_rel]
    
    # 2. 提取标签 Patch (H, W)
    patch_label = block_label_data[r_min_rel:r_max_rel, c_min_rel:c_max_rel]
    
    C, H, W = patch_factors.shape
    if H != crop_size or W != crop_size:
        return 0 # 尺寸不匹配，跳过

    # 确保标签 Patch 只有 0 (非滑坡) 和 1 (滑坡)
    # 原标签是 1=滑坡点, 2=非滑坡点。 
    # 我们应该将所有 1 标记的滑坡点 和 2 标记的非滑坡点 周围的 Mask区域保留下来。
    # 现在的 Label Mask (patch_label) 包含 1 或 2，需要将其转换为 0 (非滑坡) 和 1 (滑坡)。
    
    # FCN/U-Net 的训练标签通常是 0/1 Mask
    # 我们基于中心点的标签来决定它属于哪个文件夹。
    # label == 1 (滑坡): 目标文件夹 '1'
    # label == 0 (非滑坡): 目标文件夹 '0'
    
    # 注意：patch_label 中的实际像素值仍是 1 (滑坡点) 和 2 (非滑坡点)
    # 如果要将其作为 FCN 的像素级标签，需要将 1/2 转换为 1/0
    # 由于原始标签只标记了点，Mask 中大部分区域可能为 NoData 或 0。
    # 假设 LSM Label TIF 中: 1 = 滑坡点, 2 = 非滑坡点, 0/NoData = 背景
    # FCN 训练通常需要一个完整的 Mask，这里仍然沿用您中心点采样的方式，
    # 但保存整个 Patch 的标签区域。
    
    # 简化：我们不修改 patch_label 的内容（保留 1 和 2，用于判断是否为滑坡点/非滑坡点），
    # 但 FCN 训练时，需要将 1/2 映射到 1/0。
    # 如果 TIF 中是 1=滑坡点, 2=非滑坡点，那么 FCN 的目标应该是将 1 预测为滑坡。
    # 故我们让 FCN 训练脚本处理 1/2 到 1/0 的映射，这里只保存原始数据。
    
    # 1. 保存因子 Patch
    factors_save_path = os.path.join(factors_save_dir, str(label), f"{r_abs}_{c_abs}.npy")
    np.save(factors_save_path, patch_factors)

    # 2. 保存标签 Patch (Mask)
    labels_save_path = os.path.join(labels_save_dir, str(label), f"{r_abs}_{c_abs}.npy")
    np.save(labels_save_path, patch_label.astype(np.uint8)) # 标签通常用 uint8
    
    return 1 
# ----------------------------------------


def extract_patches_fcn_parallel(factors_dir, label_path, factors_output_dir, labels_output_dir, crop_size, mode='all'):
    """
    FCN 兼容版本：同时提取因子 Patch 和标签 Mask。
    """
    crop_size = int(crop_size)
    
    # 1. 读取标签数据，获取所有样本坐标
    with rasterio.open(label_path) as src:
        label_data = src.read(1)
        height, width = label_data.shape
        # 假设 1=滑坡点，2=非滑坡点 (根据您原有的代码逻辑)
        slide_indices = np.argwhere(label_data == 1)
        non_slide_indices = np.argwhere(label_data == 2)
        
    print(f"Found {len(slide_indices)} landslide samples and {len(non_slide_indices)} non-landslide samples.")

    # 2. 样本均衡与坐标准备 (保持原有的 1:1 均衡和 1/0 标签映射)
    if len(non_slide_indices) > len(slide_indices):
        indices_to_keep = np.random.choice(len(non_slide_indices), len(slide_indices), replace=False)
        non_slide_indices = non_slide_indices[indices_to_keep]
    
    samples = []
    # 标签 1 (滑坡) 映射到 1
    for idx in slide_indices:
        samples.append((idx[0], idx[1], 1)) 
    # 标签 2 (非滑坡) 映射到 0
    for idx in non_slide_indices:
        samples.append((idx[0], idx[1], 0)) 
        
    random.shuffle(samples)
    sample_coords = {(r, c): label for r, c, label in samples}
    
    # 3. 准备输出目录
    factors_save_dir = os.path.join(factors_output_dir, mode)
    labels_save_dir = os.path.join(labels_output_dir, mode)
    
    for save_dir in [factors_save_dir, labels_save_dir]:
        os.makedirs(os.path.join(save_dir, '1'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, '0'), exist_ok=True)
    
    # 4. 获取所有影响因子文件路径
    factor_files = find_factor_files(factors_dir)
    print(f"Detected {len(factor_files)} factor layers.")

    # 5. 定义大块遍历 (外层 I/O 循环)
    print(f"Starting block processing using a persistent pool of {NUM_PROCESSES} processes.")
    total_saved_patches = 0

    # 预先打开所有 factor 文件句柄和标签文件句柄
    src_factors = [rasterio.open(f) for f in factor_files]
    src_label = rasterio.open(label_path) # 打开标签文件

    with multiprocessing.Pool(processes=NUM_PROCESSES) as pool: 
        
        # 遍历大图，按 BLOCK_SIZE 分块
        for row in tqdm(range(0, height, BLOCK_SIZE), desc="Processing Blocks (Row)"):
            for col in range(0, width, BLOCK_SIZE):
                
                win_h = min(BLOCK_SIZE, height - row)
                win_w = min(BLOCK_SIZE, width - col)
                
                if win_h < crop_size or win_w < crop_size:
                    continue

                io_window = Window(col, row, win_w, win_h)
                
                # --- 内存加载当前 Block 数据 ---
                block_factors_data = []
                valid_block = True
                
                # 5.1 加载因子 Block
                for src in src_factors:
                    try:
                        # 归一化和 NaN 处理 (保持与原逻辑一致)
                        data = src.read(1, window=io_window)
                        data = np.nan_to_num(data, nan=0.0)
                        data = np.clip(data, 0, 1) 
                        block_factors_data.append(data)
                    except Exception as e:
                        valid_block = False
                        break
                
                if not valid_block:
                    continue
                    
                block_factors_data = np.stack(block_factors_data, axis=0).astype(np.float32) # (C, H_block, W_block)
                
                # 5.2 加载标签 Block (Label Mask)
                try:
                    block_label_data = src_label.read(1, window=io_window) # (H_block, W_block)
                except Exception as e:
                    print(f"Error reading label block at ({row}, {col}): {e}")
                    del block_factors_data
                    continue
                
                # --- 过滤当前 Block 内的样本点 ---
                pool_args = []
                
                for (r_abs, c_abs), label in sample_coords.items():
                    # 检查样本点是否在当前 Block 内
                    if row <= r_abs < row + win_h and col <= c_abs < col + win_w:
                        
                        pool_args.append((
                            block_factors_data, block_label_data, r_abs, c_abs, label, 
                            crop_size, factors_save_dir, labels_save_dir, 
                            row, col, height, width
                        ))

                # --- 并行提取和保存 ---
                if pool_args:
                    # 使用持久化进程池提交任务
                    results_iterator = pool.imap_unordered(save_single_patch_fcn, pool_args)
                    saved_in_block = sum(tqdm(results_iterator, total=len(pool_args), leave=False, desc="Saving Patches in Block"))
                    total_saved_patches += saved_in_block

                # 内存清理
                del block_factors_data
                del block_label_data
        
    # 关闭文件句柄
    for src in src_factors:
        src.close()
    src_label.close()
        
    print(f"Total saved {total_saved_patches} patches to {factors_save_dir} and {labels_save_dir}")
    return total_saved_patches


def main(input_factors_dir, input_labels_dir, output_factors_dir, output_labels_dir, crop_size):
    crop_size = int(crop_size)
    
    # 清理旧数据并创建目录
    for output_dir in [output_factors_dir, output_labels_dir]:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

    # 稳健的标签文件查找逻辑 (用于获取坐标和 Block 数据)
    label_path = None
    if os.path.isdir(input_labels_dir):
        potential_files = [f for f in os.listdir(input_labels_dir) if f.endswith('.tif')]
        if potential_files:
            label_path = os.path.join(input_labels_dir, potential_files[0])
    
    if label_path is None:
        raise FileNotFoundError(f"Could not find any .tif file in {input_labels_dir}")
        
    print(f"Using label file: {label_path}")

    # --- 1. 使用 FCN 加速版本提取所有样本到临时目录 ---
    temp_factors_dir = os.path.join(output_factors_dir, 'temp')
    temp_labels_dir = os.path.join(output_labels_dir, 'temp')

    extract_patches_fcn_parallel(
        factors_dir=input_factors_dir, 
        label_path=label_path, 
        factors_output_dir=temp_factors_dir,
        labels_output_dir=temp_labels_dir, # 标签输出目录
        crop_size=crop_size, 
        mode='all'
    )
    
    # --- 2. 划分 Train / Val / Test (6:2:2) ---
    print("Splitting into Train/Val/Test (6:2:2)...")
    
    for label_class in ['0', '1']:
        # 因子数据划分
        factors_class_dir = os.path.join(temp_factors_dir, 'all', label_class)
        # 标签数据划分
        labels_class_dir = os.path.join(temp_labels_dir, 'all', label_class)

        if not os.path.exists(factors_class_dir): continue
        
        files = os.listdir(factors_class_dir) # 以因子文件列表为准
        random.shuffle(files)
        
        total_count = len(files)
        
        # 定义划分索引 (60% Train, 20% Val, 20% Test)
        train_end_idx = int(total_count * 0.6)
        val_end_idx = int(total_count * 0.8) 
        
        train_files = files[:train_end_idx]
        val_files = files[train_end_idx:val_end_idx]
        test_files = files[val_end_idx:]
        
        # 定义目标路径
        dest_map = {
            'train': (os.path.join(output_factors_dir, 'train', label_class), os.path.join(output_labels_dir, 'train', label_class)),
            'val': (os.path.join(output_factors_dir, 'val', label_class), os.path.join(output_labels_dir, 'val', label_class)),
            'test': (os.path.join(output_factors_dir, 'test', label_class), os.path.join(output_labels_dir, 'test', label_class)),
        }
        
        for mode_name, (factors_dest, labels_dest) in dest_map.items():
            os.makedirs(factors_dest, exist_ok=True)
            os.makedirs(labels_dest, exist_ok=True)
            
            file_list = {'train': train_files, 'val': val_files, 'test': test_files}[mode_name]
            
            print(f"--- Class {label_class} - {mode_name} ---")
            for f in tqdm(file_list, leave=False, desc=f"Moving {mode_name} Patches"):
                # 移动因子 Patch
                shutil.move(os.path.join(factors_class_dir, f), os.path.join(factors_dest, f))
                # 移动标签 Patch
                shutil.move(os.path.join(labels_class_dir, f), os.path.join(labels_dest, f))

    # 清理临时目录
    shutil.rmtree(temp_factors_dir)
    shutil.rmtree(temp_labels_dir)
    print("Train/Val/Test split and cleanup complete.")

if __name__ == '__main__':
    if sys.platform.startswith('win'):
        multiprocessing.freeze_support()
        
    try:
        if len(sys.argv) < 2:
            raise RuntimeError("Missing config")
        config_path = sys.argv[1]
        params = get_argv(config_path)
        
        if len(params) != 5:
             # params 数量检查，确保获取了 output_labels_dir
             raise ValueError(f"Expected 5 parameters, got {len(params)}. Check XML config for 'output_labels_dir'.")
             
        # main(*params) 包含 output_labels_dir
        main(*params)
        
        print('<process_status>0</process_status>')
        print('<process_log>success</process_log>')
    except Exception as e:
        import traceback
        traceback.print_exc()
        print('<process_status>1</process_status>')
        print(f'<process_log>{str(e)}</process_log>')
        