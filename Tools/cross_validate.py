import sys
import os
import xml.etree.ElementTree as ET
import torch
import logging
import warnings
import copy
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import (
    precision_score, recall_score, f1_score, jaccard_score,
    cohen_kappa_score, accuracy_score, confusion_matrix
)
from datetime import datetime

from Network.utils import load_config, LandslideNet, LandslideDataset, create_dataloaders

def get_argv(xml_file):
    param_names = [
        'train_output', 'num_epochs', 'lr',
        'device_ids', 'patience', 'output_factors_dir',
        'output_labels_dir', 'batch_size', 'crop_size',
        'num_workers', 'num_bands', 'weight_decay'
    ]
    params = []
    root = ET.parse(xml_file).getroot()
    
    for name in param_names:
        for param in root.findall('param'):
            if param.find('name').text == name:
                params.append(param.find('value').text)
                break
        else:
            raise ValueError(f"Parameter {name} not found in config")
    return params

def setup_logger(output_dir, log_file='training.log'):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    os.makedirs(output_dir, exist_ok=True)
    
    log_path = os.path.join(output_dir, log_file)
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def initialize_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def evaluate_model(model, data_loader, phase='Validation'):
    model.eval()
    running_loss = 0.0
    all_labels, all_preds = [], []
    
    with torch.no_grad():
        for inputs, labels, mask in data_loader:
            inputs, labels, mask = inputs.cuda(), labels.cuda(), mask.cuda()
            
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels, ignore_index=-1, reduction='none')
            valid_mask = (labels != -1) & mask
            loss = (loss * valid_mask).sum() / valid_mask.sum().clamp(min=1)
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels[valid_mask].cpu().numpy())
            all_preds.extend(preds[valid_mask].cpu().numpy())
    
    avg_loss = running_loss / len(data_loader)
    
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    precision = precision_score(all_labels, all_preds, average='binary', pos_label=1, zero_division=1)
    recall = recall_score(all_labels, all_preds, average='binary', pos_label=1, zero_division=1)
    f1 = f1_score(all_labels, all_preds, average='binary', pos_label=1, zero_division=1)
    iou = jaccard_score(all_labels, all_preds, average='binary', pos_label=1, zero_division=1)
    oa = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)
    specificity = tn / (tn + fp + 1e-6)
    
    result_str = (f"\n[{phase} Results]\n"
                  f"Loss: {avg_loss:.4f}\n"
                  f"OA: {oa:.4f}\n"
                  f"Kappa: {kappa:.4f}\n"
                  f"Precision: {precision:.4f}\n"
                  f"Recall: {recall:.4f}\n"
                  f"Specificity: {specificity:.4f}\n"
                  f"F1 Score: {f1:.4f}\n"
                  f"IoU: {iou:.4f}\n"
                  f"Confusion Matrix:\n{cm}")
    
    return result_str, {
        'loss': avg_loss, 'oa': oa, 'kappa': kappa,
        'precision': precision, 'recall': recall, 'specificity': specificity,
        'f1': f1, 'iou': iou, 'cm': cm
    }

def train_model(model, train_loader, val_loader, test_loader, num_epochs=10, lr=0.00001, 
               device_ids=[0, 1], patience=20, output_dir='output', weight_decay=1e-4, logger=None):
    if logger is None:
        logger = setup_logger(output_dir)
        
    model.apply(initialize_weights)
    model = nn.DataParallel(model, device_ids=device_ids).cuda()
    
    criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
    optimizer = optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True, min_lr=1e-7
    )

    best_val_kappa = 0.0
    no_improvement_counter = 0
    best_model_epoch = 0
    best_weights = model.state_dict()
    best_test_metrics = None

    for epoch in range(int(num_epochs)):
        model.train()
        running_loss = 0.0
        all_labels, all_preds = [], []
        
        for inputs, labels, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels, mask = inputs.cuda(), labels.cuda(), mask.cuda()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            valid_mask = (labels != -1) & mask
            loss = (loss * valid_mask).sum() / valid_mask.sum().clamp(min=1)
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels[valid_mask].cpu().numpy())
            all_preds.extend(preds[valid_mask].cpu().numpy())

        avg_train_loss = running_loss / len(train_loader)
        train_precision = precision_score(all_labels, all_preds, average='binary', pos_label=1, zero_division=1)
        train_recall = recall_score(all_labels, all_preds, average='binary', pos_label=1, zero_division=1)
        train_f1 = f1_score(all_labels, all_preds, average='binary', pos_label=1, zero_division=1)
        train_oa = accuracy_score(all_labels, all_preds)
        
        log_msg = (f"\n[Epoch {epoch+1}/{num_epochs}] "
                   f"Train Loss: {avg_train_loss:.4f} | OA: {train_oa:.4f} | "
                   f"Precision: {train_precision:.4f} | Recall: {train_recall:.4f} | F1: {train_f1:.4f}")
        print(log_msg)
        logger.info(log_msg)

        val_result_str, val_metrics = evaluate_model(model, val_loader)
        val_kappa = val_metrics['kappa']
        
        print(val_result_str)
        logger.info(val_result_str)
        
        logger.info(
            f"Epoch {epoch+1} | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | "
            f"Train/Val OA: {train_oa:.4f}/{val_metrics['oa']:.4f} | "
            f"Train/Val Kappa: {train_oa:.4f}/{val_metrics['kappa']:.4f} | "
            f"Train/Val Precision: {train_precision:.4f}/{val_metrics['precision']:.4f} | "
            f"Train/Val Recall: {train_recall:.4f}/{val_metrics['recall']:.4f} | "
            f"Train/Val F1: {train_f1:.4f}/{val_metrics['f1']:.4f} | "
            f"Val IoU: {val_metrics['iou']:.4f}"
        )

        if val_kappa > best_val_kappa:
            best_val_kappa = val_kappa
            best_weights = model.state_dict()
            no_improvement_counter = 0
            best_model_epoch = epoch + 1
            
            torch.save(model.state_dict(), os.path.join(output_dir, f"best_model_epoch_{epoch+1}.pth"))
            save_msg = f"\nNew best model saved at epoch {epoch+1}, evaluating on test set..."
            print(save_msg)
            logger.info(save_msg)
            
            test_result_str, test_metrics = evaluate_model(model, test_loader, 'Test')
            best_test_metrics = test_metrics
            print(test_result_str)
            logger.info(f"Test results for best model (epoch {epoch+1}): {test_result_str}")
        else:
            no_improvement_counter += 1

        scheduler.step(val_kappa)

        if no_improvement_counter >= int(patience//2) and epoch > 10:
            logger.info(f"Early stopping warning at epoch {epoch+1} (no Kappa improvement for {int(patience//2)} epochs)")
        if no_improvement_counter >= int(patience):
            logger.info(f"Final early stopping at epoch {epoch+1} (no Kappa improvement for {patience} epochs)")
            break

    torch.save(best_weights, os.path.join(output_dir, "best_model_weight.pth"))
    logger.info(f"Best model saved from epoch {best_model_epoch} (val_kappa={best_val_kappa:.4f})")
    
    return best_test_metrics

def setup_fold_logger(output_dir, fold_idx):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"fold_{fold_idx}.log")
    
    logger = logging.getLogger(f'CV_Fold_{fold_idx}')
    logger.setLevel(logging.INFO)
    
    if logger.handlers:
        logger.handlers.clear()
    
    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter(f'%(asctime)s - [Fold {fold_idx}] - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(f'[Fold {fold_idx}] %(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def analyze_cv_results(metrics_list, save_dir):
    if not metrics_list:
        print("No valid training results obtained.")
        return

    df_data = []
    for m in metrics_list:
        m_copy = {k: v for k, v in m.items() if k != 'cm'}
        df_data.append(m_copy)
    
    df = pd.DataFrame(df_data)
    
    summary_stats = df.describe().transpose()
    
    csv_path_detailed = os.path.join(save_dir, "cv_results_detailed.csv")
    csv_path_summary = os.path.join(save_dir, "cv_results_summary.csv")
    
    df.to_csv(csv_path_detailed, index=False)
    summary_stats.to_csv(csv_path_summary)
    
    print("\n" + "="*60)
    print("          K-FOLD CROSS VALIDATION FINAL REPORT          ")
    print("="*60)
    print(f"Detailed data saved to: {csv_path_detailed}")
    print(f"Summary statistics saved to: {csv_path_summary}")
    
    target_metrics = ['oa', 'kappa', 'f1', 'iou', 'precision', 'recall']
    
    print(f"\n{'Metric':<12} | {'Mean':<10} | {'Std':<10} | {'Min':<10} | {'Max':<10}")
    print("-" * 70)
    
    for metric in target_metrics:
        if metric in df.columns:
            mean_val = df[metric].mean()
            std_val = df[metric].std()
            min_val = df[metric].min()
            max_val = df[metric].max()
            
            print(f"{metric:<12} | {mean_val:<10.4f} | {std_val:<10.4f} | {min_val:<10.4f} | {max_val:<10.4f}")
    
    print("-" * 70)
    
    if 'f1' in df.columns:
        f1_cv = (df['f1'].std() / df['f1'].mean()) * 100 if df['f1'].mean() != 0 else 0
        print("\n[Model Stability Evaluation]")
        if f1_cv < 2.0:
            print(">> High Robustness: Model performance is insensitive to data splitting.")
        elif f1_cv < 5.0:
            print(">> Good Stability: Performance fluctuation is within acceptable range.")
        else:
            print(f">> Fluctuation Exists: F1 coefficient of variation is {f1_cv:.2f}%. Please check data distribution or regularization strength.")

def run_cross_validation(config_path, k_folds=10):
    print(f"Reading configuration file: {config_path}")
    params = get_argv(config_path)
    
    base_output_dir = params[0]
    num_epochs = int(params[1])
    lr = float(params[2])
    device_ids = list(map(int, params[3].strip('[]').split(',')))
    patience = int(params[4])
    factors_dir = params[5]
    labels_dir = params[6]
    batch_size = int(params[7])
    crop_size = int(params[8])
    num_workers = int(params[9])
    num_bands = int(params[10])
    weight_decay = float(params[11])
    
    cv_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    cv_root_dir = os.path.join(base_output_dir, f"CV_Results_{cv_timestamp}")
    os.makedirs(cv_root_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Starting {k_folds}-Fold Cross Validation")
    print(f"Results root directory: {cv_root_dir}")
    print(f"{'='*60}\n")
    
    print("Loading full dataset...")
    full_dataset = LandslideDataset(
        factors_dir=factors_dir,
        labels_dir=labels_dir,
        crop_size=crop_size
    )
    print(f"Full dataset loaded, valid samples: {len(full_dataset)}")
    
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=20250609)
    
    all_fold_metrics = []
    
    for fold_idx, (train_indices, val_indices) in enumerate(kf.split(full_dataset)):
        current_fold_num = fold_idx + 1
        
        print(f"\n>>> Starting Fold {current_fold_num}/{k_folds} <<<")
        
        fold_dir = os.path.join(cv_root_dir, f"Fold_{current_fold_num}")
        os.makedirs(fold_dir, exist_ok=True)
        
        logger = setup_fold_logger(fold_dir, current_fold_num)
        logger.info(f"Starting Fold {current_fold_num}/{k_folds}")
        
        train_subset = Subset(full_dataset, train_indices)
        val_subset = Subset(full_dataset, val_indices)
        
        logger.info(f"Train samples: {len(train_subset)}, Val samples: {len(val_subset)}")
        
        train_loader = DataLoader(
            train_subset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers
        )
        val_loader = DataLoader(
            val_subset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers
        )
        
        model = LandslideNet(num_bands=num_bands)
        
        try:
            fold_test_metrics = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=val_loader,
                num_epochs=num_epochs,
                lr=lr,
                device_ids=device_ids,
                patience=patience,
                output_dir=fold_dir,
                weight_decay=weight_decay,
                logger=logger
            )
            
            if fold_test_metrics:
                fold_test_metrics['fold'] = current_fold_num
                all_fold_metrics.append(fold_test_metrics)
                logger.info(f"Fold {current_fold_num} completed successfully.")
            else:
                logger.warning(f"Fold {current_fold_num} finished but no metrics were returned.")
                
        except Exception as e:
            logger.error(f"Error occurred in Fold {current_fold_num}: {str(e)}", exc_info=True)
        
        del model, train_loader, val_loader
        torch.cuda.empty_cache()
        
        logging.getLogger(f'CV_Fold_{current_fold_num}').handlers.clear()

    print(f"\n{'='*60}")
    print("All folds training completed, generating analysis report...")
    analyze_cv_results(all_fold_metrics, cv_root_dir)
    print(f"{'='*60}\n")

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning, message=".*NCCL.*")
    
    try:
        if len(sys.argv) < 2:
            raise RuntimeError("Missing configuration file path")
            
        config_path = sys.argv[1]
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        run_cross_validation(config_path, k_folds=10)
        
        print('<cv_status>0</cv_status>')
        print('<cv_log>Cross validation completed successfully</cv_log>')
        
    except Exception as e:
        error_msg = str(e).replace('\n', ' ').replace('\t', ' ')
        print('<cv_status>1</cv_status>')
        print(f'<cv_log>{error_msg}</cv_log>')
        import traceback
        traceback.print_exc()
        