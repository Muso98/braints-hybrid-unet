"""
Evaluation metrics for tumor segmentation.

Dice and IoU: standard overlap metrics.
Hausdorff distance: boundary distance — important for thesis (tumor boundaries).
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.distance import directed_hausdorff


@torch.no_grad()
def dice_score(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> float:
    """Dice coefficient for a batch. Returns mean over batch."""
    probs = torch.sigmoid(logits)
    pred = (probs > threshold).float()
    target = target.float()

    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)

    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean().item()


@torch.no_grad()
def iou_score(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> float:
    """Intersection-over-Union for a batch."""
    probs = torch.sigmoid(logits)
    pred = (probs > threshold).float()
    target = target.float()

    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)

    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()


def hausdorff_distance_2d(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Hausdorff distance between two 2D binary masks.

    Returns max distance (in pixels) between the two boundary point sets.
    Returns NaN if either mask is empty.
    """
    pred_pts = np.argwhere(pred_mask > 0)
    gt_pts = np.argwhere(gt_mask > 0)
    if pred_pts.size == 0 or gt_pts.size == 0:
        return float("nan")

    d1 = directed_hausdorff(pred_pts, gt_pts)[0]
    d2 = directed_hausdorff(gt_pts, pred_pts)[0]
    return max(d1, d2)


@torch.no_grad()
def evaluate_batch(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> dict:
    """Compute all metrics for a batch and return dict."""
    return {
        "dice": dice_score(logits, target, threshold),
        "iou": iou_score(logits, target, threshold),
    }
