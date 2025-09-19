import os
import sys
import rasterio
import xml.etree.ElementTree as ET
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import shap
import matplotlib.pyplot as plt
import logging
import random
import warnings
from tqdm import tqdm
from rasterio.windows import Window
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from utils import load_config, LandslideDataset, LandslideNet, create_dataloaders

# 配置日志
def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger('feature_importance')
    logger.setLevel(logging.INFO)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)
    
    # 文件处理器
    file_handler = logging.FileHandler(os.path.join(output_dir, 'importance_log.txt'))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(file_handler)
    
    return logger

def get_argv(xml_file):
    argv_names = [
        'train_output',
        'output_factors_dir',
        'output_labels_dir', 
        'importance_dir',
        'analysis_times',
        'device_ids',
        'batch_size'
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

def calculate_feature_importance(model_path, factors_dir, labels_dir, output_dir, times=5, device_ids=None, batch_size=32):
    """Multi-cycle feature importance calculation with optimized memory management"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 存储每轮的重要性分数
    all_epoch_scores = np.zeros((times, 20))
    accumulated_scores = np.zeros(20)
    
    # 初始化日志
    logger = setup_logger(output_dir)
    logger.info("=== Start feature importance analysis ===")
    
    for epoch in range(times):
        # print(f"\n=== Processing Cycle {epoch+1}/{times} ===")
        logger.info(f"\n=== Processing Cycle {epoch+1}/{times} ===")
        
        model = LandslideNet(num_bands=20)
        model = nn.DataParallel(model, device_ids=list(map(int, device_ids.strip('[]').split(',')))).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        train_loader, _, test_loader = create_dataloaders(factors_dir, labels_dir, batch_size=int(batch_size), seed=random.randint(1, 1000000))
        dataset = LandslideDataset(factors_dir, labels_dir)
        
        epoch_scores = single_epoch_analysis(model, test_loader, device)
        all_epoch_scores[epoch] = epoch_scores
        accumulated_scores += epoch_scores
        
        # 计算当前平均分数
        current_avg = accumulated_scores / (epoch + 1)
        # print("\nCurrent average importance scores after {} cycles:".format(epoch + 1))
        logger.info("\nCurrent average importance scores after {} cycles:".format(epoch + 1))
        
        for i, score in enumerate(current_avg):
            factor_name = dataset.factors_subdirs[i].split('_')[0]
            # print(f"Factor {factor_name}: {score:.6f}")
            logger.info(f"Factor {factor_name}: {score:.6f}")
        
        del model, train_loader, test_loader
        torch.cuda.empty_cache()
    
    average_scores = accumulated_scores / times
    # 绘制最终排名图
    visualize_importance(average_scores, dataset.factors_subdirs, output_dir)
    # 绘制折线图
    visualize_importance_trend(all_epoch_scores, dataset.factors_subdirs, output_dir)
    
    return average_scores

def single_epoch_analysis(model, test_loader, device):
    """Single cycle analysis"""
    base_accuracy = evaluate_model(model, test_loader, device)
    importance_scores = []
    
    for band_idx in tqdm(range(20), desc="Analyzing features"):
        modified_loader = perturb_band(test_loader, band_idx, device)
        perturbed_accuracy = evaluate_model(model, modified_loader, device)
        importance_scores.append(base_accuracy - perturbed_accuracy)
    
    return np.array(importance_scores)

def evaluate_model(model, loader, device):
    """Memory-efficient accuracy evaluation"""
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            valid_mask = (labels != -1)
            correct += (predicted[valid_mask] == labels[valid_mask]).sum().item()
            total += valid_mask.sum().item()
    
    return correct / total if total != 0 else 0

def perturb_band(loader, band_idx, device):
    """Data perturbation with noise replacement"""
    class ModifiedDataset(Dataset):
        def __init__(self, original_loader):
            self.samples = []
            for batch in original_loader:
                inputs, labels = batch
                for i in range(inputs.shape[0]):
                    sample = inputs[i].unsqueeze(0)
                    self.samples.append((sample, labels[i]))
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            input, label = self.samples[idx]
            noisy = input.clone()
            # 生成与被扰动波段相同形状的[0,1]均匀分布噪声
            noise = torch.rand_like(noisy[:, band_idx])
            noisy[:, band_idx] = noise  # 用噪声替换原波段数据
            return noisy.squeeze(0), label

    modified_dataset = ModifiedDataset(loader)
    return DataLoader(modified_dataset, batch_size=32, collate_fn=pad_collate)

def pad_collate(batch):
    """Custom padding handler"""
    inputs, labels = zip(*batch)
    max_h = max([x.shape[1] for x in inputs])
    max_w = max([x.shape[2] for x in inputs])
    
    padded_inputs = []
    for x in inputs:
        pad_h = max_h - x.shape[1]
        pad_w = max_w - x.shape[2]
        x_padded = F.pad(x, (0, pad_w, 0, pad_h))
        padded_inputs.append(x_padded)
    
    return torch.stack(padded_inputs), torch.stack(labels)

def visualize_importance(scores, factor_names, output_dir):
    """Visualization of final feature importance ranking"""
    factor_names = [name.split('_')[0] for name in factor_names]
    indices = np.argsort(scores)
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(scores)), scores[indices], color='#1f77b4')
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                 f'{width:.4f}', ha='left', va='center')
    
    plt.yticks(range(len(scores)), np.array(factor_names)[indices], fontsize=12)
    plt.xlabel('Average Importance Score', fontsize=14)
    plt.title(f'Feature Importance Ranking (N={len(scores)})', fontsize=16, pad=20)
    plt.xlim(left=min(scores)-0.05, right=max(scores)+0.1)
    plt.tight_layout()
    
    plt.savefig(f"{output_dir}/importance_ranking.png", dpi=300, bbox_inches='tight')
    np.save(f"{output_dir}/average_scores.npy", scores)
    plt.close()

def visualize_importance_trend(epoch_scores, factor_names, output_dir):
    """Visualization of feature importance trend across cycles"""
    factor_names = [name.split('_')[0] for name in factor_names]
    n_cycles = epoch_scores.shape[0]
    
    plt.figure(figsize=(14, 10))
    x = np.arange(1, n_cycles + 1)  # 循环次数从1开始
    
    for i in range(20):
        # 计算累积平均（每轮后更新）
        cumulative_avg = np.cumsum(epoch_scores[:, i]) / (x)
        plt.plot(x, cumulative_avg, label=factor_names[i], linewidth=2)
    
    plt.xlabel('Cycle Number', fontsize=14)
    plt.ylabel('Average Importance Score', fontsize=14)
    plt.title('Feature Importance Trend Across Cycles', fontsize=16, pad=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 调整图例
    if len(factor_names) <= 20:
        plt.legend(loc='best', fontsize=10, ncol=2)
    else:
        plt.legend(loc='upper right', fontsize=8, ncol=3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/importance_trend.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存每轮分数
    np.save(f"{output_dir}/all_epoch_scores.npy", epoch_scores)

def main(train_output, factors_dir, labels_dir, importance_dir, analysis_times=5, device_ids=None, batch_size=32):
    model_path = os.path.join(train_output, 'best_model_weight.pth')
    os.makedirs(importance_dir, exist_ok=True)
    calculate_feature_importance(model_path, factors_dir, labels_dir, importance_dir,
                               times=int(analysis_times),
                               device_ids=device_ids,
                               batch_size=batch_size)

if __name__ == '__main__':
    try:
        if len(sys.argv) < 2:
            raise RuntimeError("Missing XML configuration file parameter")
            
        config_path = sys.argv[1]
        params = get_argv(config_path)
        
        if len(params) != 7:
            raise ValueError("Configuration parameters count mismatch")
            
        main(*params)
        print('<feature_importance_status>0</feature_importance_status>')
        print('<feature_importance_log>success</feature_importance_log>')
        
    except Exception as e:
        error_msg = str(e).replace('\n', ' ').replace('\t', ' ')
        print('<feature_importance_status>1</feature_importance_status>')
        print(f'<feature_importance_log>{error_msg}</feature_importance_log>')
