"""Training and metric utilities shared by all dense comparison models."""

from __future__ import annotations

import csv
import copy
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (
    precision_score, recall_score, f1_score, jaccard_score,
    cohen_kappa_score, accuracy_score, confusion_matrix, roc_auc_score,
    average_precision_score,
)

from .progress import TqdmLoggingHandler, metric_line, track


TRAINING_HISTORY_FIELDS = (
    "epoch",
    "total_epochs",
    "learning_rate",
    "train_loss",
    "train_f1",
    "train_precision",
    "train_recall",
    "train_oa",
    "train_kappa",
    "val_loss",
    "val_f1",
    "val_auc",
    "val_pr_auc",
    "val_precision",
    "val_recall",
    "val_oa",
    "val_kappa",
    "val_threshold",
    "selection_metric",
    "selection_score",
    "is_best",
    "best_epoch",
    "best_selection_score",
    "early_stop_counter",
)


def initialize_training_history(output_dir):
    """Create a fresh, machine-readable epoch history for one training run."""
    history_path = os.path.join(output_dir, "training_history.csv")
    with open(history_path, "w", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=TRAINING_HISTORY_FIELDS).writeheader()
    return history_path


def append_training_history(history_path, row):
    """Persist one completed epoch so interrupted runs remain plottable."""
    with open(history_path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRAINING_HISTORY_FIELDS)
        writer.writerow({name: row.get(name, "") for name in TRAINING_HISTORY_FIELDS})

def resolve_device(device_ids):
    if not torch.cuda.is_available():
        return torch.device('cpu'), []
    
    available = torch.cuda.device_count()
    valid_device_ids = [gpu_id for gpu_id in device_ids if 0 <= gpu_id < available]
    if not valid_device_ids:
        valid_device_ids = [0]
    return torch.device(f'cuda:{valid_device_ids[0]}'), valid_device_ids

def setup_logger(output_dir, log_file='training.log'):
    logger = logging.getLogger(f"LandslideNet.training.{os.path.abspath(output_dir)}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    
    os.makedirs(output_dir, exist_ok=True)
    
    log_path = os.path.join(output_dir, log_file)
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    console_handler = TqdmLoggingHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S"))
    logger.addHandler(console_handler)
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

    def forward(self, logits, targets, valid_mask=None, pixel_weights=None):
        probs = F.softmax(logits, dim=1)
        pos_probs = probs[:, 1, :, :]
        pos_targets = (targets == 1).float()

        weights = torch.ones_like(pos_probs)
        if valid_mask is not None:
            weights = weights * valid_mask.float()
        if pixel_weights is not None:
            weights = weights * pixel_weights.float()

        intersection = (pos_probs * pos_targets * weights).sum()
        union = ((pos_probs + pos_targets) * weights).sum()
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice


def unpack_supervised_batch(batch):
    if len(batch) == 4:
        inputs, labels, mask, weights = batch
    elif len(batch) == 3:
        inputs, labels, mask = batch
        weights = torch.ones_like(labels, dtype=torch.float32)
    else:
        raise ValueError(f"Expected a 3- or 4-item supervised batch, got {len(batch)}.")
    return inputs, labels, mask, weights

def predict_probabilities(model, inputs, use_tta=False):
    if not use_tta:
        logits = model(inputs)
        return F.softmax(logits, dim=1)[:, 1], logits

    transforms = [
        (lambda x: x, lambda x: x),
        (lambda x: torch.flip(x, dims=[3]), lambda x: torch.flip(x, dims=[2])),
        (lambda x: torch.flip(x, dims=[2]), lambda x: torch.flip(x, dims=[1])),
        (lambda x: torch.flip(x, dims=[2, 3]), lambda x: torch.flip(x, dims=[1, 2])),
    ]
    probs = []
    logits_original = None
    for forward_transform, inverse_transform in transforms:
        transformed = forward_transform(inputs)
        logits = model(transformed)
        if logits_original is None:
            logits_original = logits
        prob = F.softmax(logits, dim=1)[:, 1]
        probs.append(inverse_transform(prob))
    return torch.stack(probs, dim=0).mean(dim=0), logits_original


def find_best_threshold(labels, probabilities, metric='f1'):
    if len(labels) == 0:
        return 0.5

    thresholds = np.linspace(0.05, 0.95, 181)
    best_threshold = 0.5
    best_score = -np.inf

    for threshold in thresholds:
        preds = (probabilities >= threshold).astype(np.int64)
        if metric == 'kappa':
            score = cohen_kappa_score(labels, preds)
        elif metric == 'iou':
            score = jaccard_score(labels, preds, average='binary', pos_label=1, zero_division=0)
        else:
            score = f1_score(labels, preds, average='binary', pos_label=1, zero_division=0)

        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold


def build_metrics(labels, preds, probabilities, avg_loss, phase, threshold):
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    precision = precision_score(labels, preds, average='binary', pos_label=1, zero_division=0)
    recall = recall_score(labels, preds, average='binary', pos_label=1, zero_division=0)
    f1 = f1_score(labels, preds, average='binary', pos_label=1, zero_division=0)
    iou = jaccard_score(labels, preds, average='binary', pos_label=1, zero_division=0)
    oa = accuracy_score(labels, preds)
    kappa = cohen_kappa_score(labels, preds)
    specificity = tn / (tn + fp + 1e-6)
    if np.unique(labels).size == 2:
        auc = roc_auc_score(labels, probabilities)
        pr_auc = average_precision_score(labels, probabilities)
    else:
        auc = float('nan')
        pr_auc = float('nan')

    result_str = (f"\n[{phase} Results]\n"
                  f"Loss: {avg_loss:.4f}\n"
                  f"Threshold: {threshold:.3f}\n"
                  f"AUC: {auc:.4f}\n"
                  f"PR-AUC: {pr_auc:.4f}\n"
                  f"OA: {oa:.4f}\n"
                  f"Kappa: {kappa:.4f}\n"
                  f"Precision: {precision:.4f}\n"
                  f"Recall: {recall:.4f}\n"
                  f"Specificity: {specificity:.4f}\n"
                  f"F1 Score: {f1:.4f}\n"
                  f"IoU: {iou:.4f}\n"
                  f"Confusion Matrix:\n{cm}")

    return result_str, {
        'loss': avg_loss, 'threshold': threshold, 'auc': auc, 'pr_auc': pr_auc,
        'oa': oa, 'kappa': kappa,
        'precision': precision, 'recall': recall, 'specificity': specificity,
        'f1': f1, 'iou': iou, 'cm': cm,
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }


def stratified_bootstrap_intervals(labels, probabilities, threshold, replicates=1000,
                                   confidence=0.95, seed=20250609,
                                   block_ids=None, progress_desc=None):
    """Bootstrap test metrics by class, optionally resampling spatial blocks."""
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if replicates <= 0:
        return {}
    if not 0 < confidence < 1:
        raise ValueError("bootstrap confidence must be between 0 and 1.")
    class_indices = [np.flatnonzero(labels == value) for value in (0, 1)]
    if any(len(indices) == 0 for indices in class_indices):
        return {
            "bootstrap_replicates": int(replicates),
            "bootstrap_confidence": float(confidence),
            "bootstrap_status": "unavailable_single_class",
        }

    block_groups = None
    if block_ids is not None:
        block_ids = np.asarray(block_ids, dtype=np.int64)
        if block_ids.shape != labels.shape:
            raise ValueError("bootstrap block IDs must align with labels.")
        block_groups = []
        for indices in class_indices:
            class_blocks = block_ids[indices]
            block_groups.append([
                indices[class_blocks == block_id]
                for block_id in np.unique(class_blocks)
            ])

    rng = np.random.default_rng(seed)
    names = ("auc", "pr_auc", "f1", "kappa", "precision", "recall")
    samples = {name: np.empty(replicates, dtype=np.float64) for name in names}
    replicate_range = track(
        range(replicates),
        total=replicates,
        desc=progress_desc or "",
        unit="replicate",
        disable=(progress_desc is None),
    )
    for replicate in replicate_range:
        if block_groups is None:
            indices = np.concatenate([
                rng.choice(class_index, size=len(class_index), replace=True)
                for class_index in class_indices
            ])
        else:
            sampled_groups = []
            for groups in block_groups:
                selected_groups = rng.integers(0, len(groups), size=len(groups))
                sampled_groups.extend(groups[index] for index in selected_groups)
            indices = np.concatenate(sampled_groups)
        replicate_labels = labels[indices]
        replicate_probabilities = probabilities[indices]
        replicate_predictions = (replicate_probabilities >= threshold).astype(np.int64)
        samples["auc"][replicate] = roc_auc_score(
            replicate_labels, replicate_probabilities
        )
        samples["pr_auc"][replicate] = average_precision_score(
            replicate_labels, replicate_probabilities
        )
        samples["f1"][replicate] = f1_score(
            replicate_labels, replicate_predictions, zero_division=0
        )
        samples["kappa"][replicate] = cohen_kappa_score(
            replicate_labels, replicate_predictions
        )
        samples["precision"][replicate] = precision_score(
            replicate_labels, replicate_predictions, zero_division=0
        )
        samples["recall"][replicate] = recall_score(
            replicate_labels, replicate_predictions, zero_division=0
        )

    alpha = (1.0 - confidence) / 2.0
    result = {
        "bootstrap_replicates": int(replicates),
        "bootstrap_confidence": float(confidence),
        "bootstrap_seed": int(seed),
        "bootstrap_policy": (
            "class_stratified_spatial_tile_block_resampling"
            if block_groups is not None
            else "stratified_within_observed_test_classes"
        ),
        "bootstrap_status": "ok",
    }
    if block_groups is not None:
        result["bootstrap_negative_spatial_blocks"] = int(len(block_groups[0]))
        result["bootstrap_positive_spatial_blocks"] = int(len(block_groups[1]))
    for name in names:
        finite = samples[name][np.isfinite(samples[name])]
        if finite.size:
            low, high = np.quantile(finite, [alpha, 1.0 - alpha])
        else:
            low = high = np.nan
        result[f"{name}_ci_low"] = float(low)
        result[f"{name}_ci_high"] = float(high)
    return result


def evaluate_model(model, data_loader, device, phase='Validation', threshold=0.5,
                   tune_threshold=False, threshold_metric='f1', use_tta=False,
                   bootstrap_replicates=0, bootstrap_confidence=0.95,
                   bootstrap_seed=20250609, return_predictions=False,
                   show_progress=False):
    model.eval()
    running_loss = 0.0
    all_labels, all_probs, all_blocks = [], [], []
    next_block_id = 0
    
    with torch.no_grad():
        batches = track(
            data_loader,
            total=len(data_loader),
            desc=f"{phase} 推理",
            unit="batch",
            disable=not show_progress,
        )
        for batch in batches:
            inputs, labels, mask, _weights = unpack_supervised_batch(batch)
            inputs = inputs.to(device)
            labels = labels.to(device)
            mask = mask.to(device)
            
            probabilities, outputs = predict_probabilities(model, inputs, use_tta=use_tta)
            loss = F.cross_entropy(outputs, labels, ignore_index=-1, reduction='none')
            valid_mask = (labels != -1) & mask
            loss = (loss * valid_mask).sum() / valid_mask.sum().clamp(min=1)
            
            running_loss += loss.item()
            valid_cpu = valid_mask.cpu().numpy()
            all_labels.extend(labels[valid_mask].cpu().numpy())
            all_probs.extend(probabilities[valid_mask].cpu().numpy())
            batch_blocks = np.broadcast_to(
                np.arange(
                    next_block_id,
                    next_block_id + labels.shape[0],
                    dtype=np.int64,
                )[:, None, None],
                tuple(labels.shape),
            )
            all_blocks.extend(batch_blocks[valid_cpu])
            next_block_id += int(labels.shape[0])
    
    avg_loss = running_loss / len(data_loader)
    all_labels = np.asarray(all_labels, dtype=np.int64)
    all_probs = np.asarray(all_probs, dtype=np.float64)
    all_blocks = np.asarray(all_blocks, dtype=np.int64)
    if tune_threshold:
        threshold = find_best_threshold(all_labels, all_probs, metric=threshold_metric)
    all_preds = (all_probs >= float(threshold)).astype(np.int64)

    result_str, metrics = build_metrics(
        all_labels,
        all_preds,
        all_probs,
        avg_loss,
        phase,
        float(threshold),
    )
    intervals = stratified_bootstrap_intervals(
        all_labels,
        all_probs,
        float(threshold),
        replicates=int(bootstrap_replicates),
        confidence=float(bootstrap_confidence),
        seed=int(bootstrap_seed),
        block_ids=all_blocks,
        progress_desc=(f"{phase} 空间块 Bootstrap" if show_progress else None),
    )
    metrics.update(intervals)
    if return_predictions:
        metrics["_labels"] = all_labels
        metrics["_probabilities"] = all_probs
        metrics["_block_ids"] = all_blocks
    if intervals.get("bootstrap_status") == "ok":
        bootstrap_label = (
            "class-stratified spatial tile-block bootstrap"
            if intervals.get("bootstrap_policy")
            == "class_stratified_spatial_tile_block_resampling"
            else "stratified bootstrap"
        )
        result_str += (
            f"\n{100 * bootstrap_confidence:.1f}% {bootstrap_label} CIs "
            f"({bootstrap_replicates} replicates): "
            + ", ".join(
                f"{name}=[{intervals[f'{name}_ci_low']:.4f}, "
                f"{intervals[f'{name}_ci_high']:.4f}]"
                for name in ("auc", "pr_auc", "f1", "kappa", "precision", "recall")
            )
        )
    return result_str, metrics

def train_model(model, train_loader, val_loader, test_loader, num_epochs=10, lr=0.00001, 
               device_ids=[0, 1], patience=20, output_dir='output', weight_decay=1e-4,
               logger=None, selection_metric='f1', threshold_metric='f1', use_tta=False,
               test_bootstrap_replicates=0, test_bootstrap_confidence=0.95,
               test_bootstrap_seed=20250609, train_evaluation_loader=None):
    os.makedirs(output_dir, exist_ok=True)
    if logger is None:
        logger = setup_logger(output_dir)
        
    model.apply(initialize_weights)
    device, valid_device_ids = resolve_device(device_ids)
    model = model.to(device)
    if device.type == 'cuda' and len(valid_device_ids) > 1:
        model = nn.DataParallel(model, device_ids=valid_device_ids)
    device_name = (
        torch.cuda.get_device_name(device)
        if device.type == "cuda" else "CPU"
    )
    logger.info(
        f"训练启动 | device={device} ({device_name}) | GPUs={valid_device_ids or 'none'} | "
        f"epochs={int(num_epochs)} | train/val/test batches="
        f"{len(train_loader)}/{len(val_loader)}/{len(test_loader)} | "
        f"lr={float(lr):.3g} | patience={int(patience)}"
    )
    
    criterion_ce = nn.CrossEntropyLoss(ignore_index=-1, reduction='none', label_smoothing=0.05)
    criterion_dice = DiceLoss()
    optimizer = optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay) * 0.5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=2, eta_min=1e-7
    )

    best_val_score = -np.inf
    no_improvement_counter = 0
    best_model_epoch = 0
    best_weights = copy.deepcopy(model.state_dict())
    best_threshold = 0.5
    history_path = initialize_training_history(output_dir)

    epoch_progress = track(
        range(int(num_epochs)),
        total=int(num_epochs),
        desc="模型训练",
        unit="epoch",
        leave=True,
    )
    for epoch in epoch_progress:
        model.train()
        running_loss = 0.0
        all_labels, all_preds = [], []
        
        batch_progress = track(
            train_loader,
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{num_epochs} 训练",
            unit="batch",
        )
        for batch_index, batch in enumerate(batch_progress, start=1):
            inputs, labels, mask, sample_weights = unpack_supervised_batch(batch)
            inputs = inputs.to(device)
            labels = labels.to(device)
            mask = mask.to(device)
            sample_weights = sample_weights.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss_ce = criterion_ce(outputs, labels)
            valid_mask = (labels != -1) & mask
            weighted_valid = valid_mask.float() * sample_weights
            loss_ce = (loss_ce * weighted_valid).sum() / weighted_valid.sum().clamp(min=1e-6)
            loss_dice = criterion_dice(outputs, labels, valid_mask, sample_weights)
            loss = loss_ce + 0.3 * loss_dice
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels[valid_mask].cpu().numpy())
            all_preds.extend(preds[valid_mask].cpu().numpy())
            batch_progress.set_postfix(
                loss=f"{running_loss / batch_index:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                refresh=False,
            )

        avg_train_loss = running_loss / len(train_loader)
        train_precision = precision_score(all_labels, all_preds, average='binary', pos_label=1, zero_division=0)
        train_recall = recall_score(all_labels, all_preds, average='binary', pos_label=1, zero_division=0)
        train_f1 = f1_score(all_labels, all_preds, average='binary', pos_label=1, zero_division=0)
        train_oa = accuracy_score(all_labels, all_preds)
        train_kappa = cohen_kappa_score(all_labels, all_preds)
        
        _val_result_str, val_metrics = evaluate_model(
            model,
            val_loader,
            device,
            threshold=None,
            tune_threshold=True,
            threshold_metric=threshold_metric,
            use_tta=use_tta,
            show_progress=True,
        )
        val_score = val_metrics.get(selection_metric, val_metrics['f1'])

        if val_score > best_val_score:
            best_val_score = val_score
            best_threshold = val_metrics['threshold']
            best_weights = copy.deepcopy(model.state_dict())
            no_improvement_counter = 0
            best_model_epoch = epoch + 1
            
            save_msg = (
                f"新的最佳验证模型：epoch={epoch + 1} | "
                f"val_{selection_metric}={best_val_score:.4f} | "
                f"threshold={best_threshold:.3f}"
            )
            logger.info(save_msg)
        else:
            no_improvement_counter += 1

        epoch_summary = (
            f"Epoch {epoch + 1:03d}/{int(num_epochs):03d} | "
            f"train loss={avg_train_loss:.4f}, F1={train_f1:.4f}, "
            f"Kappa={train_kappa:.4f} | val loss={val_metrics['loss']:.4f}, "
            f"{metric_line(val_metrics)}, threshold={val_metrics['threshold']:.3f} | "
            f"best {selection_metric}={best_val_score:.4f}@{best_model_epoch} | "
            f"early-stop={no_improvement_counter}/{int(patience)}"
        )
        logger.info(epoch_summary)
        append_training_history(
            history_path,
            {
                "epoch": epoch + 1,
                "total_epochs": int(num_epochs),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_loss": avg_train_loss,
                "train_f1": train_f1,
                "train_precision": train_precision,
                "train_recall": train_recall,
                "train_oa": train_oa,
                "train_kappa": train_kappa,
                "val_loss": val_metrics["loss"],
                "val_f1": val_metrics["f1"],
                "val_auc": val_metrics["auc"],
                "val_pr_auc": val_metrics.get("pr_auc", ""),
                "val_precision": val_metrics.get("precision", ""),
                "val_recall": val_metrics.get("recall", ""),
                "val_oa": val_metrics.get("oa", ""),
                "val_kappa": val_metrics.get("kappa", ""),
                "val_threshold": val_metrics["threshold"],
                "selection_metric": selection_metric,
                "selection_score": val_score,
                "is_best": int(best_model_epoch == epoch + 1),
                "best_epoch": best_model_epoch,
                "best_selection_score": best_val_score,
                "early_stop_counter": no_improvement_counter,
            },
        )
        epoch_progress.set_postfix(
            val_f1=f"{val_metrics['f1']:.4f}",
            val_auc=f"{val_metrics['auc']:.4f}",
            best=f"{best_val_score:.4f}",
            wait=f"{no_improvement_counter}/{int(patience)}",
            refresh=False,
        )

        scheduler.step()

        if no_improvement_counter >= int(patience//2) and epoch > 10:
            logger.info(
                f"Early stopping warning at epoch {epoch+1} "
                f"(no validation {selection_metric} improvement for {int(patience//2)} epochs)"
            )
        if no_improvement_counter >= int(patience):
            logger.info(
                f"Final early stopping at epoch {epoch+1} "
                f"(no validation {selection_metric} improvement for {patience} epochs)"
            )
            break

    epoch_progress.close()
    model.load_state_dict(best_weights)
    torch.save(best_weights, os.path.join(output_dir, "best_model_weight.pth"))
    logger.info(
        f"Best model saved from epoch {best_model_epoch} "
        f"(val_{selection_metric}={best_val_score:.4f}, threshold={best_threshold:.3f})"
    )

    success_metrics = {}
    if train_evaluation_loader is not None:
        _success_text, success_metrics = evaluate_model(
            model,
            train_evaluation_loader,
            device,
            "Inner-training success-rate evaluation",
            threshold=best_threshold,
            tune_threshold=False,
            use_tta=use_tta,
            return_predictions=True,
            show_progress=True,
        )
        success_labels = success_metrics.pop("_labels")
        success_probabilities = success_metrics.pop("_probabilities")
        success_blocks = success_metrics.pop("_block_ids")
        np.savez_compressed(
            os.path.join(output_dir, "success_predictions.npz"),
            label=success_labels,
            probability=success_probabilities,
            spatial_block_id=success_blocks,
        )

    # The held-out test region is deliberately evaluated once, only after model,
    # epoch, and decision threshold selection have finished on training/validation
    # regions. This prevents iterative inspection of the test set during training.
    test_result_str, best_test_metrics = evaluate_model(
        model,
        test_loader,
        device,
        'Test',
        threshold=best_threshold,
        tune_threshold=False,
        threshold_metric=threshold_metric,
        use_tta=use_tta,
        bootstrap_replicates=test_bootstrap_replicates,
        bootstrap_confidence=test_bootstrap_confidence,
        bootstrap_seed=test_bootstrap_seed,
        return_predictions=True,
        show_progress=True,
    )
    test_labels = best_test_metrics.pop("_labels")
    test_probabilities = best_test_metrics.pop("_probabilities")
    test_block_ids = best_test_metrics.pop("_block_ids")
    np.savez_compressed(
        os.path.join(output_dir, "test_predictions.npz"),
        label=test_labels,
        probability=test_probabilities,
        spatial_block_id=test_block_ids,
    )
    logger.info(
        f"Single final test evaluation (best epoch {best_model_epoch}, "
        f"selection_{selection_metric}={best_val_score:.4f}, "
        f"threshold={best_threshold:.3f}): {test_result_str}"
    )
    best_test_metrics.update({
        "best_epoch": int(best_model_epoch),
        "validation_selection_metric": selection_metric,
        "validation_selection_score": float(best_val_score),
        "threshold_metric": threshold_metric,
        "threshold_selected_on_validation": float(best_threshold),
        "test_evaluations": 1,
        "success_rate_auc": float(success_metrics.get("auc", np.nan)),
        "success_rate_pr_auc": float(success_metrics.get("pr_auc", np.nan)),
    })
    return best_test_metrics

def setup_fold_logger(output_dir, fold_idx):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"fold_{fold_idx}.log")
    
    logger = logging.getLogger(f'CV_Fold_{fold_idx}')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
    if logger.handlers:
        logger.handlers.clear()
    
    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter(f'%(asctime)s - [Fold {fold_idx}] - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    console_handler = TqdmLoggingHandler()
    console_formatter = logging.Formatter(
        f'%(asctime)s | [Fold {fold_idx}] %(message)s', "%H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
        
