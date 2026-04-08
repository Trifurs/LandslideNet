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

