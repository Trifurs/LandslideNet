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

BLOCK_SIZE = 2048 

NUM_PROCESSES = min(multiprocessing.cpu_count(), 16) 

def get_argv(xml_file):
    argv_names = [
        'input_factors_dir', 'input_labels_dir', 
        'output_factors_dir', 'output_labels_dir',
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
    
    direct_tifs = sorted([os.path.join(factors_dir, f) for f in os.listdir(factors_dir) if f.endswith('.tif')])
    if len(direct_tifs) > 0:
        factor_files = direct_tifs
    else:
        subdirs = sorted([d for d in os.listdir(factors_dir) if os.path.isdir(os.path.join(factors_dir, d))])
        for subdir in subdirs:
            subdir_path = os.path.join(factors_dir, subdir)
            tifs = sorted([f for f in os.listdir(subdir_path) if f.endswith('.tif')])
            if tifs:
                factor_files.append(os.path.join(subdir_path, tifs[0])) 
            
    if not factor_files:
        raise FileNotFoundError(f"No .tif factor files found in {factors_dir}")
        
    return factor_files

def save_single_patch_fcn(args):
    (block_factors_data, block_label_data, r_abs, c_abs, label, 
     crop_size, factors_save_dir, labels_save_dir, 
     block_row_start, block_col_start, height, width) = args
     
    half_size = crop_size // 2
    
    r_min_abs, r_max_abs = r_abs - half_size, r_abs + half_size
    c_min_abs, c_max_abs = c_abs - half_size, c_abs + half_size

    if r_min_abs < 0 or c_min_abs < 0 or r_max_abs > height or c_max_abs > width:
        return 0

    r_min_rel = r_min_abs - block_row_start
    r_max_rel = r_max_abs - block_row_start
    c_min_rel = c_min_abs - block_col_start
    c_max_rel = c_max_abs - block_col_start

    patch_factors = block_factors_data[:, r_min_rel:r_max_rel, c_min_rel:c_max_rel]

    patch_label = block_label_data[r_min_rel:r_max_rel, c_min_rel:c_max_rel]
    
    C, H, W = patch_factors.shape
    if H != crop_size or W != crop_size:
        return 0 

    factors_save_path = os.path.join(factors_save_dir, str(label), f"{r_abs}_{c_abs}.npy")
    np.save(factors_save_path, patch_factors)

    labels_save_path = os.path.join(labels_save_dir, str(label), f"{r_abs}_{c_abs}.npy")
    np.save(labels_save_path, patch_label.astype(np.uint8)) 
    
    return 1 


def extract_patches_fcn_parallel(factors_dir, label_path, factors_output_dir, labels_output_dir, crop_size, mode='all'):
    crop_size = int(crop_size)
    
    with rasterio.open(label_path) as src:
        label_data = src.read(1)
        height, width = label_data.shape
        slide_indices = np.argwhere(label_data == 1)
        non_slide_indices = np.argwhere(label_data == 2)
        
    print(f"Found {len(slide_indices)} landslide samples and {len(non_slide_indices)} non-landslide samples.")

    if len(non_slide_indices) > len(slide_indices):
        indices_to_keep = np.random.choice(len(non_slide_indices), len(slide_indices), replace=False)
        non_slide_indices = non_slide_indices[indices_to_keep]
    
    samples = []
    for idx in slide_indices:
        samples.append((idx[0], idx[1], 1)) 
    for idx in non_slide_indices:
        samples.append((idx[0], idx[1], 0)) 
        
    random.shuffle(samples)
    sample_coords = {(r, c): label for r, c, label in samples}
    
    factors_save_dir = os.path.join(factors_output_dir, mode)
    labels_save_dir = os.path.join(labels_output_dir, mode)
    
    for save_dir in [factors_save_dir, labels_save_dir]:
        os.makedirs(os.path.join(save_dir, '1'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, '0'), exist_ok=True)
    
    factor_files = find_factor_files(factors_dir)
    print(f"Detected {len(factor_files)} factor layers.")

    print(f"Starting block processing using a persistent pool of {NUM_PROCESSES} processes.")
    total_saved_patches = 0

    src_factors = [rasterio.open(f) for f in factor_files]
    src_label = rasterio.open(label_path) 

    with multiprocessing.Pool(processes=NUM_PROCESSES) as pool: 
        
        for row in tqdm(range(0, height, BLOCK_SIZE), desc="Processing Blocks (Row)"):
            for col in range(0, width, BLOCK_SIZE):
                
                win_h = min(BLOCK_SIZE, height - row)
                win_w = min(BLOCK_SIZE, width - col)
                
                if win_h < crop_size or win_w < crop_size:
                    continue

                io_window = Window(col, row, win_w, win_h)

                block_factors_data = []
                valid_block = True
                
                for src in src_factors:
                    try:
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
                
                try:
                    block_label_data = src_label.read(1, window=io_window) # (H_block, W_block)
                except Exception as e:
                    print(f"Error reading label block at ({row}, {col}): {e}")
                    del block_factors_data
                    continue
                
                pool_args = []
                
                for (r_abs, c_abs), label in sample_coords.items():
                    if row <= r_abs < row + win_h and col <= c_abs < col + win_w:
                        
                        pool_args.append((
                            block_factors_data, block_label_data, r_abs, c_abs, label, 
                            crop_size, factors_save_dir, labels_save_dir, 
                            row, col, height, width
                        ))

                if pool_args:
                    results_iterator = pool.imap_unordered(save_single_patch_fcn, pool_args)
                    saved_in_block = sum(tqdm(results_iterator, total=len(pool_args), leave=False, desc="Saving Patches in Block"))
                    total_saved_patches += saved_in_block

                del block_factors_data
                del block_label_data
        
    for src in src_factors:
        src.close()
    src_label.close()
        
    print(f"Total saved {total_saved_patches} patches to {factors_save_dir} and {labels_save_dir}")
    return total_saved_patches


def main(input_factors_dir, input_labels_dir, output_factors_dir, output_labels_dir, crop_size):
    crop_size = int(crop_size)
    
    for output_dir in [output_factors_dir, output_labels_dir]:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

    label_path = None
    if os.path.isdir(input_labels_dir):
        potential_files = [f for f in os.listdir(input_labels_dir) if f.endswith('.tif')]
        if potential_files:
            label_path = os.path.join(input_labels_dir, potential_files[0])
    
    if label_path is None:
        raise FileNotFoundError(f"Could not find any .tif file in {input_labels_dir}")
        
    print(f"Using label file: {label_path}")

    temp_factors_dir = os.path.join(output_factors_dir, 'temp')
    temp_labels_dir = os.path.join(output_labels_dir, 'temp')

    extract_patches_fcn_parallel(
        factors_dir=input_factors_dir, 
        label_path=label_path, 
        factors_output_dir=temp_factors_dir,
        labels_output_dir=temp_labels_dir,
        crop_size=crop_size, 
        mode='all'
    )
    
    print("Splitting into Train/Val/Test (6:2:2)...")
    
    for label_class in ['0', '1']:
        factors_class_dir = os.path.join(temp_factors_dir, 'all', label_class)
        labels_class_dir = os.path.join(temp_labels_dir, 'all', label_class)

        if not os.path.exists(factors_class_dir): continue
        
        files = os.listdir(factors_class_dir) 
        random.shuffle(files)
        
        total_count = len(files)
        
        train_end_idx = int(total_count * 0.6)
        val_end_idx = int(total_count * 0.8) 
        
        train_files = files[:train_end_idx]
        val_files = files[train_end_idx:val_end_idx]
        test_files = files[val_end_idx:]
        
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

                shutil.move(os.path.join(factors_class_dir, f), os.path.join(factors_dest, f))
 
                shutil.move(os.path.join(labels_class_dir, f), os.path.join(labels_dest, f))


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
             raise ValueError(f"Expected 5 parameters, got {len(params)}. Check XML config for 'output_labels_dir'.")
             
        main(*params)
        
        print('<process_status>0</process_status>')
        print('<process_log>success</process_log>')
    except Exception as e:
        import traceback
        traceback.print_exc()
        print('<process_status>1</process_status>')
        print(f'<process_log>{str(e)}</process_log>')

        
