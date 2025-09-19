import os
import sys
import xml.etree.ElementTree as ET
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import csv
from sklearn.metrics import precision_score, recall_score, f1_score
from utils import LandslideDataset, LandslideNet, create_dataloaders

def get_argv(xml_file):
    argv_names = [
        'train_output',
        'output_factors_dir',
        'output_labels_dir',
        'batch_size',
        'crop_size',
        'num_bands',
        'num_workers',
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

def calculate_feature_importance(model, dataloader, num_bands, accumulation_steps=10):
    model.eval()
    importance_matrix = torch.zeros(num_bands).cuda()
    accumulated_importance = torch.zeros(num_bands).cuda()
    num_samples = 0

    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(tqdm(dataloader)):
            inputs = inputs.cuda()
            batch_size, num_channels, height, width = inputs.shape
            original_outputs = model(inputs)
            original_probs = torch.softmax(original_outputs, dim=1)[:, 0]

            for band in range(num_bands):
                feature_removed_inputs = inputs.clone()
                feature_removed_inputs[:, band, :, :] = 0
                removed_outputs = model(feature_removed_inputs)
                removed_probs = torch.softmax(removed_outputs, dim=1)[:, 0]
                prob_diff = torch.abs(original_probs - removed_probs)
                accumulated_importance[band] += torch.sum(prob_diff)
                torch.cuda.empty_cache()

            if (batch_idx + 1) % accumulation_steps == 0:
                importance_matrix += accumulated_importance
                accumulated_importance.zero_()
            num_samples += batch_size

    if accumulated_importance.sum() > 0:
        importance_matrix += accumulated_importance
    return importance_matrix / num_samples

def main(train_output, factors_dir, labels_dir, batch_size, crop_size, num_bands, num_workers, mosaic_map):
    model = LandslideNet(int(num_bands))
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(os.path.join(train_output, 'best_model_weight.pth')))
    model = model.cuda()

    train_loader, val_loader = create_dataloaders(
        factors_dir,
        labels_dir,
        batch_size=int(batch_size),
        crop_size=int(crop_size),
        num_workers=int(num_workers)
    )

    importance = calculate_feature_importance(model, val_loader, int(num_bands))
    normalized_importance = (importance.cpu().numpy() / np.sum(importance.cpu().numpy()))

    dataset = LandslideDataset(factors_dir, labels_dir)
    
    csv_dir = os.path.dirname(mosaic_map)
    csv_path = os.path.join(csv_dir, 'feature_importance.csv')
    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Factor', 'Importance'])
        for factor_name, importance in zip(dataset.factors_subdirs, normalized_importance):
            print(f"Factor: {factor_name.ljust(25)} Importance: {importance:.4f}")
            writer.writerow([factor_name, f"{importance:.4f}"])

if __name__ == '__main__':
    try:
        if len(sys.argv) < 2:
            raise RuntimeError("Missing XML config file parameter")

        config_path = sys.argv[1]
        params = get_argv(config_path)
        
        if len(params) != 8:
            raise ValueError(f"Expected 8 parameters, got {len(params)}")
            
        main(*params)
        
        print('<analysis_status>0</analysis_status>')
        print('<analysis_log>success</analysis_log>')
        
    except Exception as e:
        error_msg = str(e).replace('\n', ' ').replace('\t', ' ')
        print('<analysis_status>1</analysis_status>')
        print(f'<analysis_log>{error_msg}</analysis_log>')
