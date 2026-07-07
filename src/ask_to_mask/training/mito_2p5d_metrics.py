"""Pixel, boundary, and object metrics for mitochondria mask proposals."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from scipy.ndimage import binary_dilation, label

from .mito_2p5d_dataset import mask_to_boundary


def pixel_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    eps: float = 1e-6,
) -> dict[str, float]:
    pred = pred.astype(bool) & valid
    target = target.astype(bool) & valid
    tp = float((pred & target).sum())
    fp = float((pred & ~target).sum())
    fn = float((~pred & target).sum())
    union = tp + fp + fn
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (union + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "fp_pixels": fp,
        "fn_pixels": fn,
    }


def boundary_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    boundary_radius_px: int = 2,
    tolerance_px: int = 2,
    eps: float = 1e-6,
) -> dict[str, float]:
    pred_b = mask_to_boundary(pred, boundary_radius_px).astype(bool) & valid
    target_b = mask_to_boundary(target, boundary_radius_px).astype(bool) & valid
    if tolerance_px > 0:
        pred_hit_region = binary_dilation(target_b, iterations=int(tolerance_px)) & valid
        target_hit_region = binary_dilation(pred_b, iterations=int(tolerance_px)) & valid
    else:
        pred_hit_region = target_b
        target_hit_region = pred_b
    precision = (float((pred_b & pred_hit_region).sum()) + eps) / (
        float(pred_b.sum()) + eps
    )
    recall = (float((target_b & target_hit_region).sum()) + eps) / (
        float(target_b.sum()) + eps
    )
    f1 = (2.0 * precision * recall + eps) / (precision + recall + eps)
    return {
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": f1,
    }


def object_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    match_iou_threshold: float = 0.1,
) -> dict[str, float]:
    pred = pred.astype(bool) & valid
    target = target.astype(bool) & valid
    structure = np.ones((3, 3), dtype=np.uint8)
    pred_labels, num_pred = label(pred, structure=structure)
    target_labels, num_target = label(target, structure=structure)

    pred_sizes = np.bincount(pred_labels.ravel(), minlength=num_pred + 1).astype(float)
    target_sizes = np.bincount(target_labels.ravel(), minlength=num_target + 1).astype(float)

    overlap_pixels = (pred_labels > 0) & (target_labels > 0)
    overlaps: dict[tuple[int, int], int] = defaultdict(int)
    pred_to_targets: dict[int, set[int]] = defaultdict(set)
    target_to_preds: dict[int, set[int]] = defaultdict(set)
    if overlap_pixels.any():
        pairs = np.stack(
            [pred_labels[overlap_pixels], target_labels[overlap_pixels]],
            axis=1,
        )
        unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
        for (pred_id, target_id), count in zip(unique_pairs, counts):
            pred_id = int(pred_id)
            target_id = int(target_id)
            count = int(count)
            overlaps[(pred_id, target_id)] = count
            pred_to_targets[pred_id].add(target_id)
            target_to_preds[target_id].add(pred_id)

    scored_pairs = []
    for (pred_id, target_id), intersection in overlaps.items():
        union = pred_sizes[pred_id] + target_sizes[target_id] - intersection
        iou = float(intersection / max(union, 1.0))
        if iou >= match_iou_threshold:
            scored_pairs.append((iou, pred_id, target_id))
    scored_pairs.sort(reverse=True)

    matched_pred: set[int] = set()
    matched_target: set[int] = set()
    for _iou, pred_id, target_id in scored_pairs:
        if pred_id in matched_pred or target_id in matched_target:
            continue
        matched_pred.add(pred_id)
        matched_target.add(target_id)

    if num_target > 0:
        object_recall = len(matched_target) / num_target
    else:
        object_recall = 1.0 if num_pred == 0 else 0.0

    false_positives = num_pred - len(matched_pred)
    missed = num_target - len(matched_target)
    merge_errors = sum(1 for ids in pred_to_targets.values() if len(ids) > 1)
    split_errors = sum(1 for ids in target_to_preds.values() if len(ids) > 1)

    return {
        "objects_gt": float(num_target),
        "objects_pred": float(num_pred),
        "object_recall": float(object_recall),
        "false_positive_objects": float(false_positives),
        "missed_mitochondria": float(missed),
        "merge_errors": float(merge_errors),
        "split_errors": float(split_errors),
    }


def evaluate_mask_arrays(
    pred: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    boundary_radius_px: int = 2,
    boundary_tolerance_px: int = 2,
    object_match_iou: float = 0.1,
) -> dict[str, float]:
    metrics = pixel_metrics(pred, target, valid)
    metrics.update(
        boundary_metrics(
            pred,
            target,
            valid,
            boundary_radius_px=boundary_radius_px,
            tolerance_px=boundary_tolerance_px,
        )
    )
    metrics.update(
        object_metrics(
            pred,
            target,
            valid,
            match_iou_threshold=object_match_iou,
        )
    )
    return metrics


def average_metric_dicts(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    keys = sorted({key for item in items for key in item})
    return {
        key: float(np.mean([item[key] for item in items if key in item]))
        for key in keys
    }


def evaluate_batch_predictions(
    logits: torch.Tensor,
    batch: dict,
    threshold: float = 0.5,
    boundary_radius_px: int = 2,
    boundary_tolerance_px: int = 2,
    object_match_iou: float = 0.1,
) -> dict[str, float]:
    prob = torch.sigmoid(logits[:, 0:1]).detach().float().cpu().numpy()
    pred = prob[:, 0] >= float(threshold)
    target = batch["target"].detach().float().cpu().numpy()[:, 0] > 0.5
    valid = batch["valid_mask"].detach().float().cpu().numpy()[:, 0] > 0
    metrics = [
        evaluate_mask_arrays(
            pred[i],
            target[i],
            valid[i],
            boundary_radius_px=boundary_radius_px,
            boundary_tolerance_px=boundary_tolerance_px,
            object_match_iou=object_match_iou,
        )
        for i in range(pred.shape[0])
    ]
    return average_metric_dicts(metrics)


def evaluate_comparison_masks(
    comparison_masks: dict[str, torch.Tensor],
    batch: dict,
    threshold: float = 0.5,
    boundary_radius_px: int = 2,
    boundary_tolerance_px: int = 2,
    object_match_iou: float = 0.1,
) -> dict[str, dict[str, float]]:
    """Evaluate aligned baseline segmentations (e.g. the existing UNet inference
    pipeline) against the same GT target, if supplied."""
    target = batch["target"].detach().float().cpu().numpy()[:, 0] > 0.5
    valid = batch["valid_mask"].detach().float().cpu().numpy()[:, 0] > 0
    results = {}
    for name, masks in comparison_masks.items():
        arr = masks.detach().float().cpu().numpy()
        if arr.ndim == 4:
            arr = arr[:, 0]
        pred = arr >= float(threshold)
        metrics = [
            evaluate_mask_arrays(
                pred[i],
                target[i],
                valid[i],
                boundary_radius_px=boundary_radius_px,
                boundary_tolerance_px=boundary_tolerance_px,
                object_match_iou=object_match_iou,
            )
            for i in range(pred.shape[0])
        ]
        results[name] = average_metric_dicts(metrics)
    return results
