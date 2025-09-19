import sys
import os
import xml.etree.ElementTree as ET
import torch
import logging
import warnings
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score
from utils import load_config, LandslideNet, create_dataloaders
from datetime import datetime  # 新增导入

def get_argv(xml_file):
    """Extract training parameters from XML config"""
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
    """Initialize logging system"""
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
    """Xavier初始化提升收敛稳定性"""
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def evaluate_model(model, data_loader, phase='Validation'):  # 重命名并修改原validate_model
    """通用评估函数"""
    model.eval()
    running_loss = 0.0
    all_labels, all_preds = [], []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels, ignore_index=-1)
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            mask = labels != -1
            all_labels.extend(labels[mask].cpu().numpy())
            all_preds.extend(preds[mask].cpu().numpy())
    
    avg_loss = running_loss / len(data_loader)
    precision = precision_score(all_labels, all_preds, average='binary', pos_label=1, zero_division=1)
    recall = recall_score(all_labels, all_preds, average='binary', pos_label=1, zero_division=1)
    f1 = f1_score(all_labels, all_preds, average='binary', pos_label=1, zero_division=1)
    
    # 格式化结果输出
    result_str = (f"\n[{phase} Results]\n"
                  f"Loss: {avg_loss:.4f}\n"
                  f"Precision: {precision:.4f}\n"
                  f"Recall: {recall:.4f}\n"
                  f"F1 Score: {f1:.4f}")
    return result_str, {'loss': avg_loss, 'precision': precision, 'recall': recall, 'f1': f1}

def train_model(model, train_loader, val_loader, num_epochs=10, lr=0.00001, 
               device_ids=[0, 1], patience=20, output_dir='output', weight_decay=1e-4):
    """Main training procedure with multi-GPU support"""
    model.apply(initialize_weights)
    model = nn.DataParallel(model, device_ids=device_ids).cuda()
    
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.AdamW(model.parameters(), 
                          lr=float(lr), 
                          weight_decay=float(weight_decay))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    logger = setup_logger(output_dir)

    best_val_loss = float('inf')
    best_val_f1 = 0
    no_improvement_counter = 0
    best_model_epoch = 0

    for epoch in range(int(num_epochs)):
        # Training phase
        model.train()
        running_loss = 0.0
        all_labels, all_preds = [], []
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.cuda(), labels.cuda()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs, labels) 
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            mask = labels != -1
            all_labels.extend(labels[mask].cpu().numpy())
            all_preds.extend(preds[mask].cpu().numpy())

        # 训练指标计算
        avg_train_loss = running_loss / len(train_loader)
        train_precision = precision_score(all_labels, all_preds, average='binary', pos_label=1, zero_division=1)
        train_recall = recall_score(all_labels, all_preds, average='binary', pos_label=1, zero_division=1)
        train_f1 = f1_score(all_labels, all_preds, average='binary', pos_label=1, zero_division=1)
        
        print(f"[Epoch {epoch+1}/{num_epochs}] "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Precision: {train_precision:.4f} | "
              f"Recall: {train_recall:.4f} | "
              f"F1: {train_f1:.4f}")

        # 验证阶段
        val_result_str, val_metrics = evaluate_model(model, val_loader)  # 修改调用方式
        val_precision = val_metrics['precision']
        val_recall = val_metrics['recall']
        val_f1 = val_metrics['f1']
        val_loss = val_metrics['loss']
        scheduler.step(val_loss)
        
        print(f"[Validation] "
              f"Loss: {val_loss:.4f} | "
              f"Precision: {val_precision:.4f} | "
              f"Recall: {val_recall:.4f} | "
              f"F1: {val_f1:.4f}")
        
        logger.info(
            f"Epoch {epoch+1} | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Train/Val Precision: {train_precision:.4f}/{val_precision:.4f} | "
            f"Train/Val Recall: {train_recall:.4f}/{val_recall:.4f} | "
            f"Train/Val F1: {train_f1:.4f}/{val_f1:.4f}"
        )

        # 模型保存策略
        if val_loss < best_val_loss or val_f1 > best_val_f1:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                
            best_weights = model.state_dict()
            no_improvement_counter = 0
            best_model_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(output_dir, f"best_model_epoch_{epoch+1}.pth"))
        else:
            no_improvement_counter += 1

        # 早停策略
        if no_improvement_counter >= int(patience//2) and epoch > 10:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
        elif no_improvement_counter >= int(patience):
            logger.info(f"Final early stopping at epoch {epoch+1}")
            break

    # 保存最终模型
    torch.save(best_weights, os.path.join(output_dir, "best_model_weight.pth"))
    logger.info(f"Best model saved from epoch {best_model_epoch} (val_loss={best_val_loss:.4f}, val_f1={best_val_f1:.4f})")
    return model

def main(params):
    """Entry point for training workflow"""
    warnings.filterwarnings("ignore", category=UserWarning, message=".*NCCL.*")
    
    # 参数转换
    device_ids = list(map(int, params[3].strip('[]').split(',')))
    
    # 创建数据加载器（修改接收test_loader）
    train_loader, val_loader, test_loader = create_dataloaders(
        factors_dir=params[5],
        labels_dir=params[6],
        batch_size=int(params[7]),
        crop_size=int(params[8]),
        num_workers=int(params[9])
    )
    
    # 初始化模型
    model = LandslideNet(num_bands=int(params[10]))
    
    # 训练模型
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=int(params[1]),
        lr=float(params[2]),
        device_ids=device_ids,
        patience=int(params[4]),
        output_dir=params[0],
        weight_decay=float(params[11])
    )

    # 测试阶段（新增部分）
    # 加载最佳模型
    best_model = LandslideNet(num_bands=int(params[10]))
    best_model_path = os.path.join(params[0], "best_model_weight.pth")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best model weights not found at {best_model_path}")
    
    best_model = nn.DataParallel(best_model, device_ids=device_ids).cuda()
    best_model.load_state_dict(torch.load(best_model_path))
    
    # 执行测试
    test_result_str, test_metrics = evaluate_model(best_model, test_loader, 'Test')
    
    # 输出到控制台
    print("\n" + "="*50)
    print(test_result_str)
    print("="*50)
    
    # 记录到日志
    logger = logging.getLogger()
    logger.info(test_result_str)
    
    # 写入独立测试日志
    test_log_path = os.path.join(params[0], 'test_results.log')
    with open(test_log_path, 'a') as f:
        f.write(f"\n=== Test at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
        f.write(test_result_str)
    
    return test_metrics

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
