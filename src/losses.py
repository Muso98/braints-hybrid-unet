"""
Loss functions for tumor segmentation.

We use a combination of:
  - Dice loss: handles class imbalance (tumor pixels << background pixels)
  - Focal loss: focuses on hard examples (small tumors)
  - Boundary loss: NEW — sharpens edges for better Hausdorff distance

Combined loss: 0.5 * Dice + 0.3 * Focal + 0.2 * Boundary
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src import config


class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.contiguous().view(probs.size(0), -1)
        target = target.contiguous().view(target.size(0), -1)

        intersection = (probs * target).sum(dim=1)
        union = probs.sum(dim=1) + target.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """Binary focal loss."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * target + (1 - p) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal = alpha_t * (1 - p_t) ** self.gamma * bce
        return focal.mean()


class BoundaryLoss(nn.Module):
    """
    Boundary loss for sharper segmentation edges.

    Idea: penalize errors near the tumor boundary more heavily.
    Implementation:
      1. Find boundary pixels of GT (via Sobel / morphological gradient)
      2. Compute weighted BCE where boundary pixels have higher weight

    Reference: Kervadec et al., "Boundary loss for highly unbalanced
               segmentation" (MIDL 2019), simplified version.
    """

    def __init__(self, theta: float = 5.0) -> None:
        super().__init__()
        self.theta = theta  # boundary weight multiplier

    def _compute_boundary_map(self, target: torch.Tensor) -> torch.Tensor:
        """
        Compute a soft boundary map from binary mask using morphological
        gradient (dilation - erosion approximated via max pooling).
        Returns weight map: boundary pixels get high weight.
        """
        # target: (B, 1, H, W)
        # Erosion ≈ -maxpool(-x); Dilation ≈ maxpool(x)
        kernel = 3
        pad = kernel // 2

        dilation = F.max_pool2d(target, kernel_size=kernel, stride=1, padding=pad)
        erosion = -F.max_pool2d(-target, kernel_size=kernel, stride=1, padding=pad)

        boundary = (dilation - erosion).clamp(0, 1)  # (B, 1, H, W)
        # Weight map: boundary pixels get theta, rest get 1
        weights = 1.0 + self.theta * boundary
        return weights

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weights = self._compute_boundary_map(target)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        weighted_bce = (bce * weights).mean()
        return weighted_bce


class CombinedLoss(nn.Module):
    """Dice + Focal + Boundary — improved loss for better edge accuracy."""

    def __init__(self) -> None:
        super().__init__()
        self.dice = DiceLoss()
        self.focal = FocalLoss(gamma=config.FOCAL_GAMMA)
        self.boundary = BoundaryLoss()
        self.dice_w = config.DICE_WEIGHT
        self.focal_w = config.FOCAL_WEIGHT
        self.boundary_w = getattr(config, "BOUNDARY_WEIGHT", 0.0)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.dice_w * self.dice(logits, target)
        loss = loss + self.focal_w * self.focal(logits, target)
        if self.boundary_w > 0:
            loss = loss + self.boundary_w * self.boundary(logits, target)
        return loss