"""
PyTorch Dataset for preprocessed BraTS 2D slices.

Reads .npy files produced by preprocess.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src import config


class BraTSDataset(Dataset):
    """
    BraTS 2D slice dataset.

    Returns:
        image: torch.FloatTensor of shape (C, H, W)
        mask:  torch.FloatTensor of shape (1, H, W) — values 0 or 1
    """

    def __init__(self, split: str = "train", transform=None) -> None:
        assert split in {"train", "val"}, f"split must be train|val, got {split}"
        self.split = split
        self.img_dir = config.DATA_PROCESSED / split / "images"
        self.mask_dir = config.DATA_PROCESSED / split / "masks"

        if not self.img_dir.exists():
            raise FileNotFoundError(
                f"Processed data not found at {self.img_dir}. "
                "Run `python -m src.preprocess` first."
            )

        self.files = sorted(p.name for p in self.img_dir.glob("*.npy"))
        if not self.files:
            raise RuntimeError(f"No .npy files found in {self.img_dir}")

        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fname = self.files[idx]
        img = np.load(self.img_dir / fname)   # (H, W, C) float32
        mask = np.load(self.mask_dir / fname)  # (H, W) uint8

        if self.transform is not None:
            # albumentations expects (H, W, C) image and (H, W) mask
            transformed = self.transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

        # To tensor: (C, H, W) and (1, H, W)
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).permute(2, 0, 1).float()
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).unsqueeze(0).float()
        else:
            # if albumentations returned a tensor, ensure shape (1,H,W) float
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            mask = mask.float()

        return img, mask
