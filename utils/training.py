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
    average_precision_score, balanced_accuracy_score, matthews_corrcoef,
    brier_score_loss,
)

from .progress import TqdmLoggingHandler, metric_line, track


TRAINING_HISTORY_FIELDS = (
    "epoch",
    "total_epochs",
    "learning_rate",
    "train_loss",
    "train_cross_entropy",
    "train_dice_loss",
    "train_class_0_cross_entropy",
    "train_class_1_cross_entropy",
    "observed_negative_to_positive_weight_ratio",
    "configured_negative_to_positive_weight_ratio",
    "objective_normalization",
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
    "val_threshold_max_score",
    "val_threshold_selected_score",
    "val_precision_recall_gap",
    "selection_metric",
    "selection_score",
    "is_best",
    "best_epoch",
    "best_selection_score",
    "early_stop_counter",
    "early_stopping_eligible",
    "domain_risk_penalty",
    "domain_risk_weight",
    "domain_risk_class_terms",
    "validation_weight_source",
    "ema_updates",
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
        if getattr(m, "_landslidenet_zero_init", False):
            nn.init.zeros_(m.weight)
        elif getattr(m, "_landslidenet_classification_head", False):
            # A fan-out initializer is inappropriate for a two-channel logits
            # layer (std ~= 1 for a 1x1 head) and causes saturated initial
            # probabilities. The same stable head initialization is used by
            # every dense comparison and ablation model.
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
        else:
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
    if len(batch) == 5:
        inputs, labels, mask, weights, groups = batch
    elif len(batch) == 4:
        inputs, labels, mask, weights = batch
        groups = torch.full_like(labels, -1, dtype=torch.long)
    elif len(batch) == 3:
        inputs, labels, mask = batch
        weights = torch.ones_like(labels, dtype=torch.float32)
        groups = torch.full_like(labels, -1, dtype=torch.long)
    else:
        raise ValueError(f"Expected a 3-, 4-, or 5-item supervised batch, got {len(batch)}.")
    return inputs, labels, mask, weights, groups

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


def _threshold_table(labels, probabilities):
    """Return exact confusion statistics at every distinct score threshold."""
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.ndim != 1 or probabilities.shape != labels.shape:
        raise ValueError("Threshold labels and probabilities must be aligned 1-D arrays.")
    if labels.size == 0 or np.any(~np.isfinite(probabilities)):
        raise ValueError("Threshold selection needs non-empty finite probabilities.")
    if np.any(~np.isin(labels, [0, 1])):
        raise ValueError("Threshold selection labels must be binary 0/1.")

    order = np.argsort(-probabilities, kind="mergesort")
    sorted_probability = probabilities[order]
    sorted_label = labels[order]
    last_at_score = np.r_[
        sorted_probability[:-1] != sorted_probability[1:],
        True,
    ]
    indices = np.flatnonzero(last_at_score)
    true_positive = np.cumsum(sorted_label == 1)[indices].astype(np.float64)
    false_positive = np.cumsum(sorted_label == 0)[indices].astype(np.float64)
    positive = float(np.count_nonzero(labels == 1))
    negative = float(np.count_nonzero(labels == 0))
    false_negative = positive - true_positive
    true_negative = negative - false_positive
    thresholds = sorted_probability[indices]

    precision = np.divide(
        true_positive,
        true_positive + false_positive,
        out=np.zeros_like(true_positive),
        where=(true_positive + false_positive) > 0,
    )
    recall = np.divide(
        true_positive,
        positive,
        out=np.zeros_like(true_positive),
        where=positive > 0,
    )
    f1 = np.divide(
        2.0 * true_positive,
        2.0 * true_positive + false_positive + false_negative,
        out=np.zeros_like(true_positive),
        where=(2.0 * true_positive + false_positive + false_negative) > 0,
    )
    iou = np.divide(
        true_positive,
        true_positive + false_positive + false_negative,
        out=np.zeros_like(true_positive),
        where=(true_positive + false_positive + false_negative) > 0,
    )
    total = positive + negative
    observed = (true_positive + true_negative) / max(total, 1.0)
    predicted_positive = true_positive + false_positive
    predicted_negative = true_negative + false_negative
    expected = (
        positive * predicted_positive + negative * predicted_negative
    ) / max(total * total, 1.0)
    kappa = np.divide(
        observed - expected,
        1.0 - expected,
        out=np.zeros_like(observed),
        where=np.abs(1.0 - expected) > 1e-12,
    )
    return {
        "threshold": thresholds,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "kappa": kappa,
    }


def find_best_threshold(
    labels,
    probabilities,
    metric='f1',
    *,
    score_tolerance=0.0,
    return_details=False,
):
    """Select an exact validation threshold with an auditable balance tie-break.

    Candidate cut-offs are the distinct validation scores, rather than an
    arbitrary 0.05--0.95 grid.  Among candidates no more than
    ``score_tolerance`` below the maximum requested metric, the threshold with
    the smallest absolute precision--recall gap is selected.  Thus a small,
    discrete validation region cannot win solely by returning the first (lowest)
    threshold on a broad F1 plateau.
    """
    if len(labels) == 0:
        details = {
            "threshold": 0.5,
            "metric": metric,
            "maximum_score": float("nan"),
            "selected_score": float("nan"),
            "score_tolerance": float(score_tolerance),
            "precision": float("nan"),
            "recall": float("nan"),
            "precision_recall_gap": float("nan"),
            "candidate_count": 0,
        }
        return details if return_details else 0.5
    if metric not in {"f1", "kappa", "iou"}:
        raise ValueError(f"Unsupported threshold metric: {metric}")
    if not 0 <= float(score_tolerance) < 1:
        raise ValueError("threshold score_tolerance must be in [0, 1).")

    table = _threshold_table(labels, probabilities)
    scores = table[metric]
    maximum_score = float(np.nanmax(scores))
    eligible = np.flatnonzero(scores >= maximum_score - float(score_tolerance) - 1e-12)
    gaps = np.abs(table["precision"][eligible] - table["recall"][eligible])
    # lexsort uses the final key as primary: balance, then score, then a
    # deterministic preference for a threshold nearer the probability centre.
    ranking = np.lexsort((
        np.abs(table["threshold"][eligible] - 0.5),
        -scores[eligible],
        gaps,
    ))
    selected = int(eligible[int(ranking[0])])
    details = {
        "threshold": float(table["threshold"][selected]),
        "metric": metric,
        "maximum_score": maximum_score,
        "selected_score": float(scores[selected]),
        "score_tolerance": float(score_tolerance),
        "precision": float(table["precision"][selected]),
        "recall": float(table["recall"][selected]),
        "precision_recall_gap": float(
            abs(table["precision"][selected] - table["recall"][selected])
        ),
        "candidate_count": int(len(table["threshold"])),
    }
    return details if return_details else details["threshold"]


def expected_calibration_error(labels, probabilities, bins=10):
    labels = np.asarray(labels, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    result = 0.0
    for index in range(int(bins)):
        selected = (probabilities >= edges[index]) & (
            probabilities < edges[index + 1]
            if index + 1 < int(bins)
            else probabilities <= edges[index + 1]
        )
        if np.any(selected):
            result += selected.mean() * abs(
                probabilities[selected].mean() - labels[selected].mean()
            )
    return float(result)


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
    balanced_accuracy = balanced_accuracy_score(labels, preds)
    mcc = matthews_corrcoef(labels, preds)
    brier = brier_score_loss(labels, probabilities)
    ece = expected_calibration_error(labels, probabilities)
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
                  f"|Precision-Recall|: {abs(precision - recall):.4f}\n"
                  f"Specificity: {specificity:.4f}\n"
                  f"F1 Score: {f1:.4f}\n"
                  f"IoU: {iou:.4f}\n"
                  f"Confusion Matrix:\n{cm}")

    return result_str, {
        'loss': avg_loss, 'threshold': threshold, 'auc': auc, 'pr_auc': pr_auc,
        'oa': oa, 'kappa': kappa,
        'precision': precision, 'recall': recall, 'specificity': specificity,
        'precision_recall_gap': abs(precision - recall),
        'balanced_accuracy': balanced_accuracy, 'mcc': mcc,
        'brier': brier, 'ece': ece,
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
                   threshold_score_tolerance=0.0,
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
            inputs, labels, mask, _weights, _groups = unpack_supervised_batch(batch)
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
    threshold_details = None
    if tune_threshold:
        threshold_details = find_best_threshold(
            all_labels,
            all_probs,
            metric=threshold_metric,
            score_tolerance=threshold_score_tolerance,
            return_details=True,
        )
        threshold = threshold_details["threshold"]
    all_preds = (all_probs >= float(threshold)).astype(np.int64)

    result_str, metrics = build_metrics(
        all_labels,
        all_preds,
        all_probs,
        avg_loss,
        phase,
        float(threshold),
    )
    if threshold_details is not None:
        metrics.update({
            "threshold_metric": threshold_details["metric"],
            "threshold_max_score": threshold_details["maximum_score"],
            "threshold_selected_score": threshold_details["selected_score"],
            "threshold_score_tolerance": threshold_details["score_tolerance"],
            "threshold_validation_precision": threshold_details["precision"],
            "threshold_validation_recall": threshold_details["recall"],
            "threshold_precision_recall_gap": threshold_details[
                "precision_recall_gap"
            ],
            "threshold_candidate_count": threshold_details["candidate_count"],
        })
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


def globally_normalized_sparse_cross_entropy(
    per_pixel_loss,
    labels,
    valid_mask,
    sample_weights,
    class_weight_totals=None,
    num_batches=1,
):
    """Preserve the dataset-level sparse-sample objective across uneven batches.

    A conventional per-batch weighted mean gives a crop containing one labelled
    pixel the same optimizer mass as a crop containing thousands.  Here every
    supervised instance retains its configured global weight: averaging the
    returned loss over all batches is exactly the epoch-wide weighted mean when
    model parameters are held fixed.  The fallback keeps compatibility with
    loaders that do not expose audited dataset totals.
    """
    zero = per_pixel_loss.sum() * 0.0
    numerators = {}
    masses = {}
    for label in (0, 1):
        selected = valid_mask & (labels == label)
        weights = sample_weights[selected]
        masses[label] = weights.sum() if weights.numel() else zero
        numerators[label] = (
            (per_pixel_loss[selected] * weights).sum() if weights.numel() else zero
        )

    configured_totals = None
    if class_weight_totals is not None:
        try:
            configured_totals = {
                label: float(class_weight_totals[label]) for label in (0, 1)
            }
        except (KeyError, TypeError, ValueError):
            configured_totals = None
    if (
        configured_totals is not None
        and all(np.isfinite(value) and value > 0 for value in configured_totals.values())
        and int(num_batches) > 0
    ):
        expected_batch_mass = sum(configured_totals.values()) / int(num_batches)
        loss = sum(numerators.values(), zero) / max(expected_batch_mass, 1e-12)
        normalization = "dataset_global_weight_mass_preserving"
    else:
        batch_mass = sum(masses.values(), zero)
        loss = sum(numerators.values(), zero) / batch_mass.clamp(min=1e-6)
        normalization = "per_batch_weighted_mean_fallback"

    diagnostics = {
        "normalization": normalization,
        "class_weight_mass": {
            label: float(masses[label].detach().item()) for label in (0, 1)
        },
        "class_loss_numerator": {
            label: float(numerators[label].detach().item()) for label in (0, 1)
        },
    }
    return loss, diagnostics


def region_risk_variance(
    per_pixel_loss,
    labels,
    valid_mask,
    sample_weights,
    groups,
):
    """Class-conditional V-REx over represented source macro-regions.

    Computing one risk per region confounds domain difficulty with the region's
    positive/negative composition.  Comparing regions separately within each
    class prevents the regularizer from reducing its variance merely by trading
    precision against recall.
    """
    penalty_terms = []
    for label in (0, 1):
        class_mask = valid_mask & (labels == label)
        selected_groups = torch.unique(groups[class_mask])
        selected_groups = selected_groups[selected_groups >= 0]
        class_risks = []
        for group_id in selected_groups:
            selected = class_mask & (groups == group_id)
            weights = sample_weights[selected]
            if weights.numel():
                class_risks.append(
                    (per_pixel_loss[selected] * weights).sum()
                    / weights.sum().clamp(min=1e-6)
                )
        if len(class_risks) >= 2:
            penalty_terms.append(
                torch.var(torch.stack(class_risks), unbiased=False)
            )
    if not penalty_terms:
        return per_pixel_loss.sum() * 0.0, 0
    return torch.stack(penalty_terms).mean(), len(penalty_terms)


def _clone_ema_model(model):
    ema_model = copy.deepcopy(model)
    ema_model.eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    return ema_model


@torch.no_grad()
def _update_ema_model(ema_model, model, decay):
    """Update parameters and floating buffers without changing model structure."""
    for ema_parameter, parameter in zip(ema_model.parameters(), model.parameters()):
        ema_parameter.mul_(decay).add_(parameter.detach(), alpha=1.0 - decay)
    for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
        if torch.is_floating_point(ema_buffer):
            ema_buffer.mul_(decay).add_(buffer.detach(), alpha=1.0 - decay)
        else:
            ema_buffer.copy_(buffer.detach())
    ema_model.eval()

def train_model(model, train_loader, val_loader, test_loader, num_epochs=10, lr=0.00001, 
               device_ids=[0, 1], patience=20, output_dir='output', weight_decay=1e-4,
               logger=None, selection_metric='auc', threshold_metric='f1', use_tta=False,
               test_bootstrap_replicates=0, test_bootstrap_confidence=0.95,
               test_bootstrap_seed=20250609, train_evaluation_loader=None,
               minimum_epochs=1, threshold_score_tolerance=0.0,
               lr_warmup_epochs=0, lr_plateau_patience=10,
               lr_plateau_factor=0.5, min_lr=1e-6,
               selection_min_delta=0.0,
               domain_risk_variance_weight=0.0,
               domain_risk_warmup_epochs=0,
               ema_decay=0.0,
               ema_start_epoch=1):
    os.makedirs(output_dir, exist_ok=True)
    if logger is None:
        logger = setup_logger(output_dir)
    minimum_epochs = int(minimum_epochs)
    num_epochs = int(num_epochs)
    patience = int(patience)
    lr_warmup_epochs = int(lr_warmup_epochs)
    lr_plateau_patience = int(lr_plateau_patience)
    domain_risk_warmup_epochs = int(domain_risk_warmup_epochs)
    ema_decay = float(ema_decay)
    ema_start_epoch = int(ema_start_epoch)
    if not 1 <= minimum_epochs <= num_epochs:
        raise ValueError("minimum_epochs must be in [1, num_epochs].")
    if patience < 1 or lr_plateau_patience < 1:
        raise ValueError("Early-stopping and LR plateau patience must be positive.")
    if not 0 < float(lr_plateau_factor) < 1:
        raise ValueError("lr_plateau_factor must be in (0, 1).")
    if lr_warmup_epochs < 0 or domain_risk_warmup_epochs < 0:
        raise ValueError("Warm-up epoch counts cannot be negative.")
    if float(selection_min_delta) < 0 or float(domain_risk_variance_weight) < 0:
        raise ValueError("Selection delta and domain-risk weight cannot be negative.")
    if not 0 <= ema_decay < 1 or ema_start_epoch < 1:
        raise ValueError("EMA decay must be in [0, 1), and its start epoch positive.")
        
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
        f"epochs={num_epochs} (minimum={minimum_epochs}) | train/val/test batches="
        f"{len(train_loader)}/{len(val_loader)}/{len(test_loader)} | "
        f"lr={float(lr):.3g} | patience={patience} | "
        f"selection={selection_metric} (min_delta={float(selection_min_delta):.3g}) | "
        f"threshold={threshold_metric} (balance_tolerance="
        f"{float(threshold_score_tolerance):.3g}) | "
        f"class-conditional V-REx={float(domain_risk_variance_weight):.3g} | "
        f"EMA={'disabled' if ema_decay == 0 else f'{ema_decay:.4f}@{ema_start_epoch}'}"
    )
    
    criterion_ce = nn.CrossEntropyLoss(ignore_index=-1, reduction='none', label_smoothing=0.05)
    criterion_dice = DiceLoss()
    base_lr = float(lr)
    initial_lr = (
        base_lr / max(lr_warmup_epochs, 1)
        if lr_warmup_epochs > 0 else base_lr
    )
    optimizer = optim.AdamW(
        model.parameters(),
        lr=initial_lr,
        weight_decay=float(weight_decay),
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(lr_plateau_factor),
        patience=lr_plateau_patience,
        threshold=float(selection_min_delta),
        threshold_mode="abs",
        min_lr=float(min_lr),
    )

    best_val_score = -np.inf
    early_stop_best = -np.inf
    no_improvement_counter = 0
    best_model_epoch = 0
    best_weights = copy.deepcopy(model.state_dict())
    best_threshold = 0.5
    best_threshold_details = {}
    best_weight_source = "raw"
    ema_model = None
    ema_updates = 0
    train_class_weight_totals = getattr(
        getattr(train_loader, "dataset", None),
        "class_weight_totals",
        None,
    )
    if train_class_weight_totals is not None:
        train_class_weight_totals = {
            label: float(train_class_weight_totals[label]) for label in (0, 1)
        }
        configured_class_ratio = (
            train_class_weight_totals[0] / train_class_weight_totals[1]
        )
        objective_normalization = "dataset_global_weight_mass_preserving"
    else:
        configured_class_ratio = float("nan")
        objective_normalization = "per_batch_weighted_mean_fallback"
    logger.info(
        "稀疏监督目标 | normalization=%s | class_weight_totals=%s | N/P=%s",
        objective_normalization,
        train_class_weight_totals,
        (
            f"{configured_class_ratio:.6f}"
            if np.isfinite(configured_class_ratio) else "unavailable"
        ),
    )
    history_path = initialize_training_history(output_dir)

    epoch_progress = track(
        range(num_epochs),
        total=num_epochs,
        desc="模型训练",
        unit="epoch",
        leave=True,
    )
    for epoch in epoch_progress:
        if lr_warmup_epochs > 0 and epoch < lr_warmup_epochs:
            warmup_lr = base_lr * float(epoch + 1) / float(lr_warmup_epochs)
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = warmup_lr
        model.train()
        running_loss = 0.0
        running_cross_entropy = 0.0
        running_dice_loss = 0.0
        running_domain_penalty = 0.0
        domain_penalty_batches = 0
        domain_risk_class_terms = 0
        epoch_class_weight_mass = {0: 0.0, 1: 0.0}
        epoch_class_loss_numerator = {0: 0.0, 1: 0.0}
        all_labels, all_preds = [], []
        if domain_risk_warmup_epochs > 0:
            active_domain_risk_weight = float(domain_risk_variance_weight) * min(
                1.0,
                float(epoch + 1) / float(domain_risk_warmup_epochs),
            )
        else:
            active_domain_risk_weight = float(domain_risk_variance_weight)
        
        batch_progress = track(
            train_loader,
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{num_epochs} 训练",
            unit="batch",
        )
        for batch_index, batch in enumerate(batch_progress, start=1):
            inputs, labels, mask, sample_weights, groups = unpack_supervised_batch(batch)
            inputs = inputs.to(device)
            labels = labels.to(device)
            mask = mask.to(device)
            sample_weights = sample_weights.to(device)
            groups = groups.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss_ce_map = criterion_ce(outputs, labels)
            valid_mask = (labels != -1) & mask
            loss_ce, objective_diagnostics = globally_normalized_sparse_cross_entropy(
                loss_ce_map,
                labels,
                valid_mask,
                sample_weights,
                class_weight_totals=train_class_weight_totals,
                num_batches=len(train_loader),
            )
            loss_dice = criterion_dice(outputs, labels, valid_mask, sample_weights)
            domain_penalty, represented_class_terms = region_risk_variance(
                loss_ce_map,
                labels,
                valid_mask,
                sample_weights,
                groups,
            )
            loss = (
                loss_ce
                + 0.3 * loss_dice
                + active_domain_risk_weight * domain_penalty
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            if ema_decay > 0 and epoch + 1 >= ema_start_epoch:
                if ema_model is None:
                    ema_model = _clone_ema_model(model)
                else:
                    _update_ema_model(ema_model, model, ema_decay)
                ema_updates += 1

            running_loss += loss.item()
            running_cross_entropy += float(loss_ce.detach().item())
            running_dice_loss += float(loss_dice.detach().item())
            for label in (0, 1):
                epoch_class_weight_mass[label] += objective_diagnostics[
                    "class_weight_mass"
                ][label]
                epoch_class_loss_numerator[label] += objective_diagnostics[
                    "class_loss_numerator"
                ][label]
            if represented_class_terms:
                running_domain_penalty += float(domain_penalty.detach().item())
                domain_penalty_batches += 1
                domain_risk_class_terms += represented_class_terms
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels[valid_mask].cpu().numpy())
            all_preds.extend(preds[valid_mask].cpu().numpy())
            batch_progress.set_postfix(
                loss=f"{running_loss / batch_index:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                refresh=False,
            )

        avg_train_loss = running_loss / len(train_loader)
        avg_train_cross_entropy = running_cross_entropy / len(train_loader)
        avg_train_dice_loss = running_dice_loss / len(train_loader)
        class_cross_entropy = {
            label: (
                epoch_class_loss_numerator[label] / epoch_class_weight_mass[label]
                if epoch_class_weight_mass[label] > 0 else float("nan")
            )
            for label in (0, 1)
        }
        observed_class_ratio = (
            epoch_class_weight_mass[0] / epoch_class_weight_mass[1]
            if epoch_class_weight_mass[1] > 0 else float("nan")
        )
        if train_class_weight_totals is not None:
            for label in (0, 1):
                if not np.isclose(
                    epoch_class_weight_mass[label],
                    train_class_weight_totals[label],
                    rtol=1e-4,
                    atol=1e-4,
                ):
                    raise RuntimeError(
                        "Observed sparse supervision mass differs from the audited "
                        f"dataset total for class {label}: "
                        f"{epoch_class_weight_mass[label]} versus "
                        f"{train_class_weight_totals[label]}."
                    )
        avg_domain_penalty = (
            running_domain_penalty / domain_penalty_batches
            if domain_penalty_batches else 0.0
        )
        train_precision = precision_score(all_labels, all_preds, average='binary', pos_label=1, zero_division=0)
        train_recall = recall_score(all_labels, all_preds, average='binary', pos_label=1, zero_division=0)
        train_f1 = f1_score(all_labels, all_preds, average='binary', pos_label=1, zero_division=0)
        train_oa = accuracy_score(all_labels, all_preds)
        train_kappa = cohen_kappa_score(all_labels, all_preds)
        
        validation_model = ema_model if ema_model is not None else model
        validation_weight_source = "ema" if ema_model is not None else "raw"
        _val_result_str, val_metrics = evaluate_model(
            validation_model,
            val_loader,
            device,
            threshold=None,
            tune_threshold=True,
            threshold_metric=threshold_metric,
            threshold_score_tolerance=threshold_score_tolerance,
            use_tta=use_tta,
            show_progress=True,
        )
        val_score = val_metrics.get(selection_metric, val_metrics['f1'])

        is_best = bool(val_score > best_val_score)
        if is_best:
            best_val_score = val_score
            best_threshold = val_metrics['threshold']
            best_threshold_details = {
                key: val_metrics[key]
                for key in (
                    "threshold_metric",
                    "threshold_max_score",
                    "threshold_selected_score",
                    "threshold_score_tolerance",
                    "threshold_validation_precision",
                    "threshold_validation_recall",
                    "threshold_precision_recall_gap",
                    "threshold_candidate_count",
                )
                if key in val_metrics
            }
            best_weights = copy.deepcopy(validation_model.state_dict())
            best_model_epoch = epoch + 1
            best_weight_source = validation_weight_source
            
            save_msg = (
                f"新的最佳验证模型：epoch={epoch + 1} | "
                f"val_{selection_metric}={best_val_score:.4f} | "
                f"threshold={best_threshold:.3f} | weights={best_weight_source}"
            )
            logger.info(save_msg)
        if val_score > early_stop_best + float(selection_min_delta):
            early_stop_best = float(val_score)
            no_improvement_counter = 0
        else:
            no_improvement_counter += 1
        early_stopping_eligible = (epoch + 1) >= minimum_epochs

        epoch_summary = (
            f"Epoch {epoch + 1:03d}/{int(num_epochs):03d} | "
            f"train loss={avg_train_loss:.4f} (CE={avg_train_cross_entropy:.4f}, "
            f"Dice={avg_train_dice_loss:.4f}, N/P={observed_class_ratio:.3f}), "
            f"F1={train_f1:.4f}, "
            f"Kappa={train_kappa:.4f} | val loss={val_metrics['loss']:.4f}, "
            f"{metric_line(val_metrics)}, threshold={val_metrics['threshold']:.3f} | "
            f"best {selection_metric}={best_val_score:.4f}@{best_model_epoch} | "
            f"V-REx={avg_domain_penalty:.4f}*{active_domain_risk_weight:.3g} | "
            f"val-weights={validation_weight_source} | "
            f"early-stop={no_improvement_counter}/{patience} "
            f"({'active' if early_stopping_eligible else f'locked until {minimum_epochs}'})"
        )
        logger.info(epoch_summary)
        append_training_history(
            history_path,
            {
                "epoch": epoch + 1,
                "total_epochs": int(num_epochs),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_loss": avg_train_loss,
                "train_cross_entropy": avg_train_cross_entropy,
                "train_dice_loss": avg_train_dice_loss,
                "train_class_0_cross_entropy": class_cross_entropy[0],
                "train_class_1_cross_entropy": class_cross_entropy[1],
                "observed_negative_to_positive_weight_ratio": observed_class_ratio,
                "configured_negative_to_positive_weight_ratio": configured_class_ratio,
                "objective_normalization": objective_normalization,
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
                "val_threshold_max_score": val_metrics.get("threshold_max_score", ""),
                "val_threshold_selected_score": val_metrics.get(
                    "threshold_selected_score", ""
                ),
                "val_precision_recall_gap": val_metrics.get(
                    "precision_recall_gap", ""
                ),
                "selection_metric": selection_metric,
                "selection_score": val_score,
                "is_best": int(is_best),
                "best_epoch": best_model_epoch,
                "best_selection_score": best_val_score,
                "early_stop_counter": no_improvement_counter,
                "early_stopping_eligible": int(early_stopping_eligible),
                "domain_risk_penalty": avg_domain_penalty,
                "domain_risk_weight": active_domain_risk_weight,
                "domain_risk_class_terms": domain_risk_class_terms,
                "validation_weight_source": validation_weight_source,
                "ema_updates": ema_updates,
            },
        )
        epoch_progress.set_postfix(
            val_f1=f"{val_metrics['f1']:.4f}",
            val_auc=f"{val_metrics['auc']:.4f}",
            best=f"{best_val_score:.4f}",
            wait=f"{no_improvement_counter}/{patience}",
            refresh=False,
        )

        if epoch + 1 >= max(lr_warmup_epochs, 1):
            previous_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(float(val_score))
            current_lr = optimizer.param_groups[0]["lr"]
            if current_lr < previous_lr:
                logger.info(
                    f"Validation {selection_metric} plateau: learning rate reduced "
                    f"from {previous_lr:.3g} to {current_lr:.3g}."
                )

        warning_patience = max(1, int(patience // 2))
        if (
            early_stopping_eligible
            and no_improvement_counter == warning_patience
        ):
            logger.info(
                f"Early stopping warning at epoch {epoch+1} "
                f"(no validation {selection_metric} improvement for "
                f"{warning_patience} epochs)"
            )
        if early_stopping_eligible and no_improvement_counter >= patience:
            logger.info(
                f"Final early stopping at epoch {epoch+1} "
                f"(no validation {selection_metric} improvement for {patience} epochs)"
            )
            break

    epoch_progress.close()
    epochs_completed = epoch + 1
    early_stopped = epochs_completed < num_epochs
    model.load_state_dict(best_weights)
    torch.save(best_weights, os.path.join(output_dir, "best_model_weight.pth"))
    logger.info(
        f"Best model saved from epoch {best_model_epoch} "
        f"(val_{selection_metric}={best_val_score:.4f}, threshold={best_threshold:.3f}, "
        f"weights={best_weight_source})"
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
        "epochs_completed": int(epochs_completed),
        "early_stopped": bool(early_stopped),
        "minimum_epochs": int(minimum_epochs),
        "early_stopping_patience": int(patience),
        "selection_min_delta": float(selection_min_delta),
        "validation_selection_metric": selection_metric,
        "validation_selection_score": float(best_val_score),
        "threshold_metric": threshold_metric,
        "threshold_selected_on_validation": float(best_threshold),
        "domain_risk_variance_weight": float(domain_risk_variance_weight),
        "domain_risk_warmup_epochs": int(domain_risk_warmup_epochs),
        "domain_risk_definition": "class_conditional_source_macro_region_risk_variance",
        "training_objective_normalization": objective_normalization,
        "training_configured_negative_to_positive_weight_ratio": float(
            configured_class_ratio
        ),
        "training_observed_negative_to_positive_weight_ratio": float(
            observed_class_ratio
        ),
        "ema_decay": float(ema_decay),
        "ema_start_epoch": int(ema_start_epoch),
        "ema_updates": int(ema_updates),
        "best_weight_source": best_weight_source,
        "learning_rate_scheduler": "linear_warmup_then_reduce_on_plateau",
        "learning_rate_warmup_epochs": int(lr_warmup_epochs),
        "learning_rate_plateau_patience": int(lr_plateau_patience),
        "learning_rate_plateau_factor": float(lr_plateau_factor),
        "minimum_learning_rate": float(min_lr),
        "test_evaluations": 1,
        "success_rate_auc": float(success_metrics.get("auc", np.nan)),
        "success_rate_pr_auc": float(success_metrics.get("pr_auc", np.nan)),
        **{
            f"validation_{key}": value
            for key, value in best_threshold_details.items()
        },
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
        
