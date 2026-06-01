import os
import rasterio
import argparse
import xml.etree.ElementTree as ET
import numpy as np
import torch
import logging
import warnings
from tqdm import tqdm
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score
from torchvision.ops import DeformConv2d
import shutil
from rasterio.windows import Window
from sklearn.model_selection import train_test_split

def load_config(config_file):
    tree = ET.parse(config_file)
    root = tree.getroot()

    params = {}
    for param in root.findall('param'):
        name = param.find('name').text
        value = param.find('value').text
        param_type = param.find('type').text
        if param_type == 'int':
            params[name] = int(value)
        elif param_type == 'float':
            params[name] = float(value)
        elif param_type == 'bool':
            params[name] = True if value.lower() == 'true' else False
        elif param_type == 'list':
            params[name] = str_to_list(value)     
        else:
            params[name] = value

    return params

def str_to_list(list_str):
    list_str = list_str.strip('[]')
    if not list_str:
        return []
    elements = list_str.split(',')
    result = [int(element.strip()) for element in elements]
    return result

class LandslideDataset(Dataset):
    def __init__(self, factors_dir, labels_dir, crop_size=512, train=False, file_list=None):
        self.factors_dir = factors_dir
        self.labels_dir = labels_dir
        self.crop_size = crop_size
        self.train = train

        self.factors_subdirs = sorted([d for d in os.listdir(factors_dir) if os.path.isdir(os.path.join(factors_dir, d))])
        self.labels_dir_name = os.listdir(labels_dir)[0]
        
        if file_list is not None:
            self.label_files = sorted(file_list)
        else:
            self.label_files = sorted([f for f in os.listdir(os.path.join(labels_dir, self.labels_dir_name)) if f.endswith('.tif')])

        self.valid_files = self._filter_valid_files()

    def _filter_valid_files(self):
        valid_files = []
        for label_file in self.label_files:
            label_path = os.path.join(self.labels_dir, self.labels_dir_name, label_file)
            try:
                with rasterio.open(label_path) as src:
                    label = src.read(1)
                if np.any(label == 1) or np.any(label == 2):
                    valid_files.append(label_file)
            except Exception as e:
                print(f"Warning: Error reading {label_file}, skipping.")
        return valid_files

    def __len__(self):
        return len(self.valid_files)
    
    def __getitem__(self, idx):
        label_file = self.valid_files[idx]
        label_path = os.path.join(self.labels_dir, self.labels_dir_name, label_file)
        
        with rasterio.open(label_path) as src:
            label = src.read(1)

        label_processed = np.full(label.shape, -1, dtype=np.int64)

        label_processed[label == 1] = 1

        label_processed[label == 2] = 0

        factors = []
        masks = []
        for factor_subdir in self.factors_subdirs:
            factor_files = sorted([f for f in os.listdir(os.path.join(self.factors_dir, factor_subdir)) if f.endswith('.tif')])
            try:
                factor_file = next(f for f in factor_files if f == label_file)
            except StopIteration:
                factor_file = factor_files[idx]
            
            factor_path = os.path.join(self.factors_dir, factor_subdir, factor_file)
            
            with rasterio.open(factor_path) as src:
                data = src.read(1)
                nodata_val = src.nodata
                
                if nodata_val is not None:
                    mask = ~np.isclose(data, nodata_val, equal_nan=True)
                else:
                    mask = np.ones_like(data, dtype=bool)
                
                data_clipped = np.clip(data, 0, 1)
                data_clipped[~mask] = 0
                
                factors.append(data_clipped)
                masks.append(mask)

        X = np.stack(factors, axis=0)
        mask = np.stack(masks, axis=0)
        combined_mask = np.all(mask, axis=0)

        X = torch.tensor(X, dtype=torch.float32)
        label = torch.tensor(label_processed, dtype=torch.long)
        combined_mask = torch.tensor(combined_mask, dtype=torch.bool)

        if self.train:
            if torch.rand(1).item() < 0.5:
                X = torch.flip(X, dims=[2])
                label = torch.flip(label, dims=[1])
                combined_mask = torch.flip(combined_mask, dims=[1])
            if torch.rand(1).item() < 0.5:
                X = torch.flip(X, dims=[1])
                label = torch.flip(label, dims=[0])
                combined_mask = torch.flip(combined_mask, dims=[0])
            
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                X = torch.rot90(X, k, dims=[1, 2])
                label = torch.rot90(label, k, dims=[0, 1])
                combined_mask = torch.rot90(combined_mask, k, dims=[0, 1])

        return X, label, combined_mask

class DCSELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.theta = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, 1),
            nn.GroupNorm(1, channel // reduction),
            nn.GELU()
        )
        self.phi = nn.Parameter(torch.randn(channel // reduction, channel))
        nn.init.kaiming_uniform_(self.phi, mode='fan_in', nonlinearity='relu') 
    
    def forward(self, x):
        B, C, H, W = x.size()
        theta = self.theta(x).view(B, -1)
        phi = F.softmax(self.phi, dim=-1)
        dynamic_weights = torch.matmul(theta, phi).view(B, C, 1, 1).sigmoid()
        return x * dynamic_weights.expand_as(x)

class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, groups=8):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, padding=padding, groups=in_ch, bias=False)
        self.gn1 = nn.GroupNorm(groups, in_ch)
        self.act1 = nn.GELU()
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.gn2 = nn.GroupNorm(groups, out_ch)
        self.act2 = nn.GELU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.gn1(x)
        x = self.act1(x)
        x = self.pointwise(x)
        x = self.gn2(x)
        return self.act2(x)

class SpatialPerceptionBlock(nn.Module):
    def __init__(self, in_c, out_c, dropout_p=0.1, groups=8):
        super().__init__()
        self.offset_conv = nn.Conv2d(in_c, 2*3*3, 3, padding=1)
        nn.init.constant_(self.offset_conv.weight, 0)
        if self.offset_conv.bias is not None:
            nn.init.constant_(self.offset_conv.bias, 0)
        
        self.deform_conv = DeformConv2d(in_c, out_c, 3, padding=1) 
        self.gn = nn.GroupNorm(groups, out_c)
        self.dcse = DCSELayer(out_c)
        self.dropout = nn.Dropout2d(p=dropout_p)
        self.act = nn.ReLU(inplace=True)

        self.downsample = None
        if in_c != out_c:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, bias=False),
                nn.GroupNorm(groups, out_c)
            )

    def forward(self, x):
        identity = x
        offsets = self.offset_conv(x)
        out = self.deform_conv(x, offsets)
        out = self.gn(out)
        out = self.dcse(out)
        out = self.dropout(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.act(out)
        return out

class LandslideNet(nn.Module): 
    def __init__(self, num_bands, num_classes=2, groups=8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(num_bands, 64, 3, padding=1, bias=False),
            nn.GroupNorm(groups, 64),
            nn.ReLU(inplace=True)
        )
        self.spb1 = SpatialPerceptionBlock(64, 128, dropout_p=0.1, groups=groups) 
        self.spb2 = SpatialPerceptionBlock(128, 256, dropout_p=0.15, groups=groups) 
        self.spb3 = SpatialPerceptionBlock(256, 512, dropout_p=0.2, groups=groups) 
        
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.dropout = nn.Dropout2d(p=0.3)
        self.dropout_dec = nn.Dropout2d(p=0.2)
        
        self.lat_conv3 = nn.Conv2d(512, 256, 1)
        self.lat_conv2 = nn.Conv2d(256, 256, 1)
        self.lat_conv1 = nn.Conv2d(128, 256, 1)
        self.smooth_conv = nn.Conv2d(256, 256, 3, padding=1)
        
        self.dec_conv1 = nn.Sequential(
            DSConv(256 + 64, 128, groups=groups),
            self.dropout_dec,
            DSConv(128, 128, groups=groups)
        )
        self.dec_conv2 = nn.Sequential(
            DSConv(128 + 64, 64, groups=groups),
            self.dropout_dec,
            DSConv(64, 64, groups=groups)
        )
        self.final_conv = nn.Conv2d(64, num_classes, 1)
        
    def forward(self, x):
        x0 = self.conv(x)
        x0_pool = F.max_pool2d(x0, 2, 2)
        
        x1_pre = self.spb1(x0_pool)
        x1_pre = self.dropout(x1_pre)
        x1 = self.pool1(x1_pre)
        
        x2_pre = self.spb2(x1)
        x2_pre = self.dropout(x2_pre)
        x2 = self.pool2(x2_pre)
        
        x3_pre = self.spb3(x2)
        x3 = self.pool3(x3_pre) 
        
        c3_lat = self.lat_conv3(x3_pre)
        c2_lat = self.lat_conv2(x2_pre)
        c3_up = F.interpolate(c3_lat, size=c2_lat.shape[2:], mode='bilinear', align_corners=True)
        p2 = c3_up + c2_lat
        
        c1_lat = self.lat_conv1(x1_pre)
        p2_up = F.interpolate(p2, size=c1_lat.shape[2:], mode='bilinear', align_corners=True)
        p1 = p2_up + c1_lat
        
        fused_features = self.smooth_conv(p1)
        
        up1 = F.interpolate(fused_features, size=x0_pool.shape[2:], mode='bilinear', align_corners=True)
        concat1 = torch.cat([up1, x0_pool], dim=1)
        dec1 = self.dec_conv1(concat1)
        
        up2 = F.interpolate(dec1, size=x0.shape[2:], mode='bilinear', align_corners=True)
        concat2 = torch.cat([up2, x0], dim=1)
        dec2 = self.dec_conv2(concat2)
        
        output = self.final_conv(dec2)
        return output

def create_dataloaders(factors_dir, labels_dir, batch_size=32, crop_size=512, num_workers=0, seed=20250609):
    labels_dir_name = os.listdir(labels_dir)[0]
    all_label_files = sorted([f for f in os.listdir(os.path.join(labels_dir, labels_dir_name)) if f.endswith('.tif')])
    
    print(f"Scanning dataset for stratified split...")
    file_names = []
    stratify_labels = []
    
    for label_file in tqdm(all_label_files):
        label_path = os.path.join(labels_dir, labels_dir_name, label_file)
        try:
            with rasterio.open(label_path) as src:
                label = src.read(1)
            
            mask_valid = (label == 1) | (label == 2)
            if not np.any(mask_valid):
                continue
            
            count_pos = np.sum(label == 1)
            count_neg = np.sum(label == 2)
            
            ratio = count_pos / (count_pos + count_neg + 1e-6)
            file_names.append(label_file)
            stratify_labels.append(1 if ratio > 0.05 else 0)
                
        except Exception as e:
            print(f"Skipping {label_file} due to error: {e}")

    train_files, temp_files, y_train, y_temp = train_test_split(
        file_names, 
        stratify_labels, 
        test_size=0.7, 
        random_state=seed, 
        stratify=stratify_labels
    )
    
    try:
        val_files, test_files = train_test_split(
            temp_files, 
            test_size=0.5, 
            random_state=seed, 
            stratify=y_temp
        )
    except ValueError:
        val_files, test_files = train_test_split(
            temp_files, 
            test_size=0.5, 
            random_state=seed
        )

    print(f"Data split complete: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")

    train_dataset = LandslideDataset(factors_dir, labels_dir, crop_size, train=True, file_list=train_files)
    val_dataset = LandslideDataset(factors_dir, labels_dir, crop_size, train=False, file_list=val_files)
    test_dataset = LandslideDataset(factors_dir, labels_dir, crop_size, train=False, file_list=test_files)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

def create_directories(base_dir, sub_dirs):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    for sub_dir in sub_dirs:
        sub_dir_path = os.path.join(base_dir, sub_dir)
        if not os.path.exists(sub_dir_path):
            os.makedirs(sub_dir_path)

def crop_raster(input_raster, output_raster, crop_size=512, overlap=128):
    with rasterio.open(input_raster) as src:
        width = src.width
        height = src.height
        step = crop_size - overlap
        
        for i in range(0, height, step):
            for j in range(0, width, step):
                window = Window(j, i, crop_size, crop_size)
                transform = src.window_transform(window)
                data = src.read(window=window, boundless=True)
                output_file = os.path.join(output_raster, f"{i}_{j}.tif")
                
                with rasterio.open(output_file, 'w', 
                                 driver='GTiff',
                                 height=window.height,
                                 width=window.width,
                                 count=src.count,
                                 dtype=data.dtype,
                                 crs=src.crs,
                                 transform=transform) as dst:
                    dst.write(data)

def crop_all_rasters(input_dir, output_dir, crop_size=512, overlap=128):
    tif_files = [f for f in os.listdir(input_dir) if f.endswith('.tif')]
    for tif_file in tif_files:
        print(f"Processing {tif_file}...")
        input_raster = os.path.join(input_dir, tif_file)
        output_raster_dir = os.path.join(output_dir, tif_file.replace('.tif', ''))
        create_directories(output_dir, [tif_file.replace('.tif', '')])
        crop_raster(input_raster, output_raster_dir, crop_size, overlap)
        print(f"Processing {tif_file} done.")

def create_black_folder(subdir_path):
    black_folder = os.path.join(subdir_path, 'black')
    if not os.path.exists(black_folder):
        os.makedirs(black_folder)
    return black_folder

def collect_black_images(subdir_path):
    black_images = []
    for file_name in os.listdir(subdir_path):
        file_path = os.path.join(subdir_path, file_name)
        if file_name.endswith('.tif'):
            with rasterio.open(file_path) as src:
                data = src.read(1)
                if np.any(np.abs(data) >= 3):
                    black_images.append(file_path)
    return black_images

def move_black_images(black_images, black_folder):
    for file_path in black_images:
        file_name = os.path.basename(file_path)
        black_file_path = os.path.join(black_folder, file_name)
        shutil.move(file_path, black_file_path)

def move_black_images_in_all_subfolders(output_dir):
    for subdir_name in os.listdir(output_dir):
        subdir_path = os.path.join(output_dir, subdir_name)
        if os.path.isdir(subdir_path):
            print(f"Processing subfolder: {subdir_name}")
            black_folder = create_black_folder(subdir_path)
            black_images = collect_black_images(subdir_path)
            if black_images:
                move_black_images(black_images, black_folder)
            print(f"Processing subfolder: {subdir_name} done.")

def collect_image_names_and_paths(subdir_path):
    image_names = []
    image_paths = []
    for file_name in os.listdir(subdir_path):
        file_path = os.path.join(subdir_path, file_name)
        if file_name.endswith('.tif'):
            image_names.append(file_name)
            image_paths.append(file_path)
    return image_names, image_paths

def collect_intersection_image_names(output_dir):
    all_image_names = None
    for subdir_name in os.listdir(output_dir):
        subdir_path = os.path.join(output_dir, subdir_name)
        if os.path.isdir(subdir_path):
            image_names, _ = collect_image_names_and_paths(subdir_path)
            if all_image_names is None:
                all_image_names = set(image_names)
            else:
                all_image_names.intersection_update(image_names)
    return all_image_names

def move_to_black_folder(image_path, black_folder):
    file_name = os.path.basename(image_path)
    black_file_path = os.path.join(black_folder, file_name)
    shutil.move(image_path, black_file_path)
    print(f"Moved image: {file_name} to {black_folder}")

def move_missing_images_to_black(output_dir):
    common_image_names = collect_intersection_image_names(output_dir)
    print(f"Common images in all subfolders: {common_image_names}")
    for subdir_name in os.listdir(output_dir):
        subdir_path = os.path.join(output_dir, subdir_name)
        if os.path.isdir(subdir_path):
            print(f"Processing subfolder: {subdir_name}")
            image_names, image_paths = collect_image_names_and_paths(subdir_path)
            for image_name, image_path in zip(image_names, image_paths):
                if image_name not in common_image_names:
                    print(f"Image {image_name} is extra, moving to black folder.")
                    black_folder = create_black_folder(subdir_path)
                    move_to_black_folder(image_path, black_folder)
