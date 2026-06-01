import sys
import os
import copy
import xml.etree.ElementTree as ET
import torch
import logging
import warnings
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import (
    precision_score, recall_score, f1_score, jaccard_score,
    cohen_kappa_score, accuracy_score, confusion_matrix
)
from utils import load_config, LandslideNet, create_dataloaders
from datetime import datetime
import numpy as np

def normalize_path(value):
    value = str(value).strip()
    if os.name != 'nt' and (value.startswith('/') or value.startswith('~') or value.startswith('.')):
        value = os.path.expanduser(value).replace('\\', '/')
    return value

def resolve_device(device_ids):
    if not torch.cuda.is_available():
        return torch.device('cpu'), []
    
    available = torch.cuda.device_count()
    valid_device_ids = [gpu_id for gpu_id in device_ids if 0 <= gpu_id < available]
    if not valid_device_ids:
        valid_device_ids = [0]
    return torch.device(f'cuda:{valid_device_ids[0]}'), valid_device_ids

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
                params.append(normalize_path(param.find('value').text))
                break
        else:
            raise ValueError(f"Parameter {name} not found in config")
    return params

def setup_logger(output_dir, log_file='training.log'):
    logger = logging.getLogger()
    if not logger.handlers:
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
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets, valid_mask=None):
        probs = F.softmax(logits, dim=1)
        pos_probs = probs[:, 1, :, :]
        pos_targets = (targets == 1).float()
        
        if valid_mask is not None:
            pos_probs = pos_probs * valid_mask.float()
            pos_targets = pos_targets * valid_mask.float()
        
        intersection = (pos_probs * pos_targets).sum()
        union = pos_probs.sum() + pos_targets.sum()
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice

def evaluate_model(model, data_loader, device, phase='Validation'):
    model.eval()
    running_loss = 0.0
    all_labels, all_preds = [], []
    
    with torch.no_grad():
        for inputs, labels, mask in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            mask = mask.to(device)
            
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
                  f"Confusion Matrix (TN, FP, FN, TP): [{tn}, {fp}, {fn}, {tp}]")
    
    return result_str, {
        'loss': avg_loss, 'oa': oa, 'kappa': kappa,
        'precision': precision, 'recall': recall, 'specificity': specificity,
        'f1': f1, 'iou': iou, 'cm': cm, 'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
    }

def train_model(model, train_loader, val_loader, test_loader, num_epochs=10, lr=0.00001, 
               device_ids=[0, 1], patience=20, output_dir='output', weight_decay=1e-4):
    model.apply(initialize_weights)
    device, valid_device_ids = resolve_device(device_ids)
    model = model.to(device)
    if device.type == 'cuda' and len(valid_device_ids) > 1:
        model = nn.DataParallel(model, device_ids=valid_device_ids)
    print(f"Using device: {device}; device_ids: {valid_device_ids}")
    
    criterion_ce = nn.CrossEntropyLoss(ignore_index=-1, reduction='none', label_smoothing=0.05)
    criterion_dice = DiceLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay) * 0.5)
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=2, eta_min=1e-7
    )
    
    logger = setup_logger(output_dir)

    best_val_kappa = -np.inf
    no_improvement_counter = 0
    best_model_epoch = 0
    best_weights = copy.deepcopy(model.state_dict())
    best_test_metrics = None

    for epoch in range(int(num_epochs)):
        model.train()
        running_loss = 0.0
        all_labels, all_preds = [], []
        
        for inputs, labels, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            mask = mask.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss_ce = criterion_ce(outputs, labels)
            valid_mask = (labels != -1) & mask
            loss_ce = (loss_ce * valid_mask).sum() / valid_mask.sum().clamp(min=1)
            
            loss_dice = criterion_dice(outputs, labels, valid_mask)
            loss = loss_ce + 0.3 * loss_dice
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
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
        
        val_result_str, val_metrics = evaluate_model(model, val_loader, device)
        val_kappa = val_metrics['kappa']
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\n[Epoch {epoch+1}/{num_epochs}] [LR: {current_lr:.2e}] [No Imp: {no_improvement_counter}/{patience}]")
        print(f"Train | Loss: {avg_train_loss:.4f} | OA: {train_oa:.4f} | P: {train_precision:.4f} | R: {train_recall:.4f}")
        print(f"Val   | Loss: {val_metrics['loss']:.4f} | Kappa: {val_metrics['kappa']:.4f} | IoU: {val_metrics['iou']:.4f}")
        print(f"Val   | P: {val_metrics['precision']:.4f} | R: {val_metrics['recall']:.4f} | CM: TN={val_metrics['tn']} FP={val_metrics['fp']} FN={val_metrics['fn']} TP={val_metrics['tp']}")
        
        is_best = val_kappa > best_val_kappa
        
        if is_best:
            best_val_kappa = val_kappa
            best_weights = copy.deepcopy(model.state_dict())
            no_improvement_counter = 0
            best_model_epoch = epoch + 1
            
            print(f"New Best Model (Kappa: {best_val_kappa:.4f})! Saving & Testing...")
            
            torch.save(model.state_dict(), os.path.join(output_dir, f"best_model_epoch_{epoch+1}.pth"))
            
            test_result_str, test_metrics = evaluate_model(model, test_loader, device, 'Test')
            best_test_metrics = test_metrics
            print(test_result_str)
            logger.info(f"Epoch {epoch+1} BEST | Val Kappa: {val_kappa:.4f} | Test Results: {test_result_str}")
        else:
            no_improvement_counter += 1
            print(f"No improvement. Best Kappa: {best_val_kappa:.4f} @ Epoch {best_model_epoch}")

        logger.info(
            f"Epoch {epoch+1:03d} | LR: {current_lr:.2e} | NoImp: {no_improvement_counter} | "
            f"T_Loss: {avg_train_loss:.4f} | V_Loss: {val_metrics['loss']:.4f} | "
            f"V_Kappa: {val_metrics['kappa']:.4f} | V_P: {val_metrics['precision']:.4f} | V_R: {val_metrics['recall']:.4f}"
        )

        scheduler.step()

        if no_improvement_counter >= int(patience):
            print(f"\nEarly Stopping triggered after {patience} epochs.")
            break

    torch.save(best_weights, os.path.join(output_dir, "best_model_weight.pth"))
    
    if best_test_metrics:
        print("\n" + "="*60)
        print("FINAL TEST RESULTS:")
        print(f"OA: {best_test_metrics['oa']:.4f} | Kappa: {best_test_metrics['kappa']:.4f}")
        print(f"P: {best_test_metrics['precision']:.4f} | R: {best_test_metrics['recall']:.4f} | F1: {best_test_metrics['f1']:.4f}")
        print("="*60)
    
    return model

def main(params):
    warnings.filterwarnings("ignore", category=UserWarning, message=".*NCCL.*")
    
    device_ids = list(map(int, params[3].strip('[]').split(',')))
    
    train_loader, val_loader, test_loader = create_dataloaders(
        factors_dir=params[5],
        labels_dir=params[6],
        batch_size=int(params[7]),
        crop_size=int(params[8]),
        num_workers=int(params[9])
    )
    
    model = LandslideNet(num_bands=int(params[10]))
    
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=int(params[1]),
        lr=float(params[2]),
        device_ids=device_ids,
        patience=int(params[4]),
        output_dir=params[0],
        weight_decay=float(params[11])
    )

if __name__ == '__main__':
    try:
        if len(sys.argv) < 2:
            raise RuntimeError("Missing configuration file path")
            
        config_path = sys.argv[1]
        parameters = get_argv(config_path)
        
        if len(parameters) != 12:
            raise ValueError("Incomplete configuration parameters")
            
        main(parameters)
        
        print('<training_status>0</training_status>')
        print('<training_log>success</training_log>')
        
    except Exception as e:
        error_msg = str(e).replace('\n', ' ').replace('\t', ' ')
        print('<training_status>1</training_status>')
        print(f'<training_log>{error_msg}</training_log>')
        
