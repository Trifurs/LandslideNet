import sys
import os
import xml.etree.ElementTree as ET
import torch
import logging
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import numpy as np 
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from utils import LandslideNet, create_dataloaders 

CRITERION = nn.CrossEntropyLoss(ignore_index=-1) 

def get_argv(xml_file):
    param_names = [
        'train_output', 'num_epochs', 'lr',
        'device_ids', 'patience', 
        'output_factors_dir', 'output_labels_dir', 
        'batch_size', 'crop_size',
        'num_workers', 'num_bands'
    ]
    params = []
    root = ET.parse(xml_file).getroot()
    for name in param_names:
        for param in root.findall('param'):
            if param.find('name').text == name:
                params.append(param.find('value').text)
                break
        else:
            raise ValueError(f"Parameter {name} not found")
    return params

def setup_logger(output_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, 'training.log')
    
    logger.handlers = [h for h in logger.handlers if not isinstance(h, logging.FileHandler)]
        
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    return logger

def train_model(model, train_loader, val_loader, num_epochs, lr, device_ids, patience, output_dir):
    global CRITERION
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your GPU setup.")

    model = nn.DataParallel(model, device_ids=device_ids).cuda()
    
    criterion = CRITERION 
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=5, 
        min_lr=1e-6, 
        verbose=True
    )
    
    logger = setup_logger(output_dir)

    best_val_f1 = float('-inf') 
    no_improvement = 0
    best_epoch = 0

    logger.info("Starting training...")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_labels_valid = [] 
        all_preds_valid = [] 
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)")
        for inputs, labels in pbar:
            # inputs: [N, C, H, W], labels: [N, H, W]
            inputs, labels = inputs.cuda(), labels.cuda()
            
            optimizer.zero_grad()
            outputs = model(inputs) # outputs: [N, C, H, W]
            
            loss = criterion(outputs, labels) 
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            _, preds = torch.max(outputs, 1) # preds: [N, H, W]
            
            valid_mask = (labels != -1)

            labels_valid = labels[valid_mask].cpu().numpy()
            preds_valid = preds[valid_mask].cpu().numpy()
            
            all_labels_valid.extend(labels_valid.flatten())
            all_preds_valid.extend(preds_valid.flatten())
            # --------------------------------------------------
            
            pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})

        epoch_loss = running_loss / len(train_loader)
        
        if all_labels_valid:
            precision = precision_score(all_labels_valid, all_preds_valid, average='binary', pos_label=1, zero_division=0)
            recall = recall_score(all_labels_valid, all_preds_valid, average='binary', pos_label=1, zero_division=0)
            f1 = f1_score(all_labels_valid, all_preds_valid, average='binary', pos_label=1, zero_division=0)
        else:
            precision, recall, f1 = 0.0, 0.0, 0.0

        val_metrics = validate_model(model.module, val_loader, device_ids[0]) 
        val_precision, val_recall, val_f1, val_loss = val_metrics
        
        scheduler.step(val_f1)
       
        print(f"Epoch {epoch+1} | Train Loss: {epoch_loss:.4f} P: {precision:.4f} R: {recall:.4f} F1: {f1:.4f} | Val Loss: {val_loss:.4f} P: {val_precision:.4f} R: {val_recall:.4f} F1: {val_f1:.4f}")
        logger.info(f"Epoch {epoch+1}: Train Loss={epoch_loss:.4f}, Train P={precision:.4f}, Train R={recall:.4f}, Train F1={f1:.4f} | Val Loss={val_loss:.4f}, Val P={val_precision:.4f}, Val R={val_recall:.4f}, Val F1={val_f1:.4f}, LR={optimizer.param_groups[0]['lr']:.6f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.module.state_dict(), os.path.join(output_dir, "best_model_weight.pth"))
            no_improvement = 0
            best_epoch = epoch + 1
            logger.info(f"Checkpoint saved. Best validation F1-score: {best_val_f1:.4f}")
        else:
            no_improvement += 1
            
        if no_improvement >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}. No improvement for {patience} epochs.")
            break
            
    torch.save(model.module.state_dict(), os.path.join(output_dir, "last_model_weight.pth"))
    
    return best_epoch

def validate_model(model, val_loader, main_device):
    global CRITERION 
    
    model = model.to(main_device)
    model.eval()
    running_loss = 0.0
    all_labels_valid = [] 
    all_preds_valid = [] 
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            # inputs: [N, C, H, W], labels: [N, H, W]
            inputs, labels = inputs.to(main_device), labels.to(main_device)
                
            outputs = model(inputs) # outputs: [N, C, H, W]
            loss = CRITERION(outputs, labels)
            running_loss += loss.item()
            
            _, preds = torch.max(outputs, 1) # preds: [N, H, W]
            
            valid_mask = (labels != -1)
            
            labels_valid = labels[valid_mask].cpu().numpy()
            preds_valid = preds[valid_mask].cpu().numpy()
            
            all_labels_valid.extend(labels_valid.flatten())
            all_preds_valid.extend(preds_valid.flatten())
            
    avg_loss = running_loss / len(val_loader)
    
    if all_labels_valid:
        precision = precision_score(all_labels_valid, all_preds_valid, pos_label=1, zero_division=0)
        recall = recall_score(all_labels_valid, all_preds_valid, pos_label=1, zero_division=0)
        f1 = f1_score(all_labels_valid, all_preds_valid, pos_label=1, zero_division=0)
    else:
        precision, recall, f1 = 0.0, 0.0, 0.0
    
    return precision, recall, f1, avg_loss

def test_model(model, test_loader, output_dir, best_epoch):
    
    global CRITERION 
    logger = logging.getLogger()
    
    best_weights_path = os.path.join(output_dir, "best_model_weight.pth")
    if not os.path.exists(best_weights_path):
        logger.error("Best model weights not found! Cannot perform final test.")
        return 0.0, 0.0, 0.0, 0.0, 0.0
        
    model.load_state_dict(torch.load(best_weights_path))
    model = model.cuda() 
    
    model.eval()
    running_loss = 0.0
    all_labels_valid = [] 
    all_preds_valid = [] 
    
    print("\n--- Starting Final Test Evaluation ---")
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        for inputs, labels in pbar:
            # inputs: [N, C, H, W], labels: [N, H, W]
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs) # outputs: [N, C, H, W]
            loss = CRITERION(outputs, labels)
            running_loss += loss.item()
            
            _, preds = torch.max(outputs, 1) # preds: [N, H, W]
            
            valid_mask = (labels != -1)

            labels_valid = labels[valid_mask].cpu().numpy()
            preds_valid = preds[valid_mask].cpu().numpy()
            
            all_labels_valid.extend(labels_valid.flatten())
            all_preds_valid.extend(preds_valid.flatten())
           
    avg_loss = running_loss / len(test_loader)
    
    if all_labels_valid:
        precision = precision_score(all_labels_valid, all_preds_valid, pos_label=1, zero_division=0)
        recall = recall_score(all_labels_valid, all_preds_valid, pos_label=1, zero_division=0)
        f1 = f1_score(all_labels_valid, all_preds_valid, pos_label=1, zero_division=0)
        accuracy = accuracy_score(all_labels_valid, all_preds_valid)
    else:
        precision, recall, f1, accuracy = 0.0, 0.0, 0.0, 0.0

    results_str = f"Final Test Results (Best Epoch: {best_epoch}): Loss={avg_loss:.4f}, Acc={accuracy:.4f}, P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}"
    print(results_str)
    logger.info(results_str)
    
    return precision, recall, f1, avg_loss, accuracy

def main(params):

    if len(params) != 11:
         raise ValueError(f"Expected 11 parameters, but got {len(params)}. 请检查 XML config 和 'get_argv' 函数，确保包含 'output_labels_dir'.")
         
    device_ids_str = params[3].strip('[]')
    if not device_ids_str: 
        raise ValueError("device_ids is empty. Please specify at least one GPU ID.")
        
    device_ids = list(map(int, device_ids_str.split(',')))
    
    output_factors_dir = params[5]
    output_labels_dir = params[6] 
    batch_size = int(params[7])
    crop_size = int(params[8])
    num_workers = int(params[9])
    num_bands = int(params[10])

    if not os.path.exists(os.path.join(output_factors_dir, 'test')):
        raise FileNotFoundError(f"Test set directory not found in {output_factors_dir}. 请确保运行了数据预处理脚本，并生成了'test'子目录。")
        
    train_loader, val_loader, test_loader = create_dataloaders(
        factors_dir=output_factors_dir, 
        labels_dir=output_labels_dir, 
        batch_size=batch_size,
        crop_size=crop_size,
        num_workers=num_workers,
    )

    model = LandslideNet(num_bands=num_bands)
    
    best_epoch = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=int(params[1]),
        lr=float(params[2]),
        device_ids=device_ids,
        patience=int(params[4]),
        output_dir=params[0]
    )
    
    final_model = LandslideNet(num_bands=num_bands) 
    test_model(final_model, test_loader, params[0], best_epoch)


if __name__ == '__main__':
    try:
        if len(sys.argv) < 2:
            raise RuntimeError("Missing config file path")
            
        config_path = sys.argv[1]
        parameters = get_argv(config_path)
        
        if len(parameters) != 11:
             raise ValueError(f"Incomplete configuration parameters. Expected 11 (including 'output_labels_dir'), got {len(parameters)}. Please update your XML config file and 'get_argv' function.")
             
        main(parameters)
        print('<training_status>0</training_status>')
        print('<training_log>success</training_log>')
    except Exception as e:
        import traceback
        traceback.print_exc()
        print('<training_status>1</training_status>')
        print(f'<training_log>{str(e)}</training_log>')

        
