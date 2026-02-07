import os
import xml.etree.ElementTree as ET
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
import csv
import logging 

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


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

class LandslidePatchDataset(Dataset):
    def __init__(self, factors_base_dir, labels_base_dir, mode='train'):
        self.factors_dir = os.path.join(factors_base_dir, mode)
        self.labels_dir = os.path.join(labels_base_dir, mode)
        self.samples = []

        for class_label in ['0', '1']:
            factors_class_dir = os.path.join(self.factors_dir, class_label)
            labels_class_dir = os.path.join(self.labels_dir, class_label)
            
            if not os.path.exists(factors_class_dir):
                logging.warning(f"Factors directory not found: {factors_class_dir}")
                continue

            file_names = sorted([f for f in os.listdir(factors_class_dir) if f.endswith('.npy')])
            
            for file_name in file_names:
                factor_path = os.path.join(factors_class_dir, file_name)
                label_path = os.path.join(labels_class_dir, file_name)

                if os.path.exists(label_path):
                    self.samples.append((factor_path, label_path))

        if not self.samples:
            raise FileNotFoundError(f"No .npy patches found in {self.factors_dir} and {self.labels_dir}")
        
        logging.info(f"Loaded {len(self.samples)} samples for {mode} set.")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        factor_path, label_path = self.samples[idx]

        X = np.load(factor_path) 
        
        label_mask = np.load(label_path)

        target = -1 * np.ones_like(label_mask, dtype=np.int64)

        target[label_mask == 1] = 1 

        target[label_mask == 2] = 0

        X = torch.as_tensor(X, dtype=torch.float32)
        target = torch.as_tensor(target, dtype=torch.long) 

        return X, target


class DCSELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.theta = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, 1),
            
            nn.BatchNorm2d(channel // reduction), 
            
            nn.GELU()
        )

        self.phi = nn.Parameter(torch.randn(channel // reduction, channel))

        nn.init.kaiming_uniform_(self.phi, mode='fan_in', nonlinearity='relu') 
    
    def forward(self, x):
        B, C, H, W = x.size()

        theta = self.theta(x).view(B, -1) # [B, C/r]

        phi = F.softmax(self.phi, dim=-1) # [C/r, C]

        dynamic_weights = torch.matmul(theta, phi) # [B, C]

        dynamic_weights = dynamic_weights.view(B, C, 1, 1).sigmoid()

        return x * dynamic_weights.expand_as(x)

class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, padding=padding, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)

class SpatialPerceptionBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        
        self.offset_conv = nn.Conv2d(in_c, 2*3*3, 3, padding=1)
        self.deform_conv = DeformConv2d(in_c, out_c, 3, padding=1) 
        self.bn = nn.BatchNorm2d(out_c)
        self.dcse = DCSELayer(out_c)
        self.act = nn.ReLU(inplace=True)

        self.downsample = None
        if in_c != out_c:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        identity = x

        offsets = self.offset_conv(x)
        out = self.deform_conv(x, offsets)
        out = self.bn(out)
        out = self.dcse(out) 

        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity # Feature Addition
        out = self.act(out)
        
        return out 

class LandslideNet(nn.Module): 
    def __init__(self, num_bands, num_classes=2):
        super(LandslideNet, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(num_bands, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.spb1 = SpatialPerceptionBlock(64, 128) 
        self.spb2 = SpatialPerceptionBlock(128, 256) 
        self.spb3 = SpatialPerceptionBlock(256, 512) 

        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.dropout = nn.Dropout2d(p=0.2) 
        self.dropout_dec = nn.Dropout2d(p=0.1)

        self.lat_conv3 = nn.Conv2d(512, 256, 1) 
        self.lat_conv2 = nn.Conv2d(256, 256, 1) 
        self.lat_conv1 = nn.Conv2d(128, 256, 1) 
        self.smooth_conv = nn.Conv2d(256, 256, 3, padding=1)

        self.dec_conv1 = nn.Sequential(
            DSConv(256 + 64, 128),
            self.dropout_dec, 
            DSConv(128, 128)
        )
        
        # Decoder 2: 128 + 64 (skip) -> 64
        self.dec_conv2 = nn.Sequential(
            DSConv(128 + 64, 64),
            self.dropout_dec, 
            DSConv(64, 64)
        )

        self.final_conv = nn.Conv2d(64, num_classes, 1)
        
    def forward(self, x):
        x0 = self.conv(x)                   # [B,64,H,W]
        x0_pool = F.max_pool2d(x0, 2, 2)     # [B,64,H/2,W/2]

        x1_pre = self.spb1(x0_pool)          # [B,128,H/2,W/2]
        x1_pre = self.dropout(x1_pre)
        x1 = self.pool1(x1_pre)              # [B,128,H/4,W/4]
        
        x2_pre = self.spb2(x1)               # [B,256,H/4,W/4]
        x2_pre = self.dropout(x2_pre)
        x2 = self.pool2(x2_pre)              # [B,256,H/8,W/8]
        
        x3_pre = self.spb3(x2)               # [B,512,H/8,W/8]
        x3 = self.pool3(x3_pre)              # [B,512,H/16,W/16] 

        c3_lat = self.lat_conv3(x3) 
        c2_lat = self.lat_conv2(x2)

        c3_up = F.interpolate(c3_lat, size=c2_lat.shape[2:], mode='bilinear', align_corners=True)
        p2 = c3_up + c2_lat                  # [B,256,H/8,W/8]
        
        c1_lat = self.lat_conv1(x1)
        p2_up = F.interpolate(p2, size=c1_lat.shape[2:], mode='bilinear', align_corners=True)
        p1 = p2_up + c1_lat                  # [B,256,H/4,W/4]

        fused_features = self.smooth_conv(p1) # [B,256,H/4,W/4]

        # Stage 1: H/4 -> H/2
        up1 = F.interpolate(fused_features, size=x0_pool.shape[2:], mode='bilinear', align_corners=True)
        concat1 = torch.cat([up1, x0_pool], dim=1)
        dec1 = self.dec_conv1(concat1)
        
        # Stage 2: H/2 -> H
        up2 = F.interpolate(dec1, size=x0.shape[2:], mode='bilinear', align_corners=True)
        concat2 = torch.cat([up2, x0], dim=1)
        dec2 = self.dec_conv2(concat2)

        output = self.final_conv(dec2)
        
        return output

def create_dataloaders(factors_dir, labels_dir, batch_size=32, crop_size=512, num_workers=0, only_test=False):    
    num_workers_opt = max(0, int(num_workers)) 

    test_dataset = LandslidePatchDataset(factors_dir, labels_dir, mode='test') 
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers_opt,
        pin_memory=True
    ) 

    if only_test:
        return None, None, test_loader

    train_dataset = LandslidePatchDataset(factors_dir, labels_dir, mode='train')
    val_dataset = LandslidePatchDataset(factors_dir, labels_dir, mode='val')
    
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers_opt, 
        pin_memory=True,
        drop_last=True
    ) 
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers_opt,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader

