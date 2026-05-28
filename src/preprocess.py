"""
Preprocess BraTS 2020: convert 3D NIfTI volumes into 2D slice .npy files.

Why 2D:
- RTX 4060 8GB cannot fit 3D models with reasonable batch sizes
- 2D is faster to iterate on for thesis experiments
- Slice-by-slice reconstruction is standard in many BraTS papers

What we do:
1. Read FLAIR + T1ce modalities and segmentation mask per patient
2. Extract middle slices (50..130) — outer slices are mostly empty
3. Z-score normalize each modality on the brain region only
4. Skip slices with too few tumor pixels (< 10) to balance the dataset slightly
5. Save as .npy: image (H, W, C) and mask (H, W) uint8
6. Patient-level train/val split (no slice leakage between sets)

Run:
    python -m src.preprocess
"""
from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

import nibabel as nib
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


def find_patients(raw_dir: Path) -> List[Path]:
    """Return sorted list of BraTS patient directories."""
    if not raw_dir.exists():
        log.error("Raw data directory not found: %s", raw_dir)
        log.error("Did you download BraTS 2020? See README section 2.")
        sys.exit(1)

    patients = sorted(p for p in raw_dir.iterdir() if p.is_dir() and p.name.startswith("BraTS20_Training_"))
    if not patients:
        log.error("No patient folders found in %s", raw_dir)
        sys.exit(1)

    log.info("Found %d patients", len(patients))
    return patients


def load_modality(patient_dir: Path, modality: str) -> np.ndarray:
    """Load a single modality NIfTI as float32. Shape (H, W, D).

    Supports both .nii and .nii.gz extensions.
    """
    pattern = f"*_{modality}.nii*"
    matches = list(patient_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No {modality} file in {patient_dir}")
    return nib.load(str(matches[0])).get_fdata().astype(np.float32)


def load_seg(patient_dir: Path) -> np.ndarray:
    """Load segmentation mask as uint8.

    Supports both .nii and .nii.gz extensions.
    """
    matches = list(patient_dir.glob("*_seg.nii*"))
    if not matches:
        raise FileNotFoundError(f"No seg file in {patient_dir}")
    return nib.load(str(matches[0])).get_fdata().astype(np.uint8)


def zscore_normalize(volume: np.ndarray) -> np.ndarray:
    """Z-score normalize a volume using only non-zero voxels (the brain mask)."""
    brain = volume[volume > 0]
    if brain.size == 0:
        return volume
    mean = brain.mean()
    std = brain.std() + 1e-8
    return (volume - mean) / std


def center_crop(arr: np.ndarray, size: int) -> np.ndarray:
    """Center-crop a 2D array to (size, size). Pads if smaller."""
    h, w = arr.shape[:2]
    if h < size or w < size:
        # pad
        pad_h = max(0, size - h)
        pad_w = max(0, size - w)
        pad = ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2))
        if arr.ndim == 3:
            pad = pad + ((0, 0),)
        arr = np.pad(arr, pad, mode="constant")
        h, w = arr.shape[:2]
    sh = (h - size) // 2
    sw = (w - size) // 2
    return arr[sh : sh + size, sw : sw + size]


def process_patient(patient_dir: Path, out_img_dir: Path, out_mask_dir: Path) -> Tuple[int, int]:
    """
    Slice one patient and save .npy files. Returns (num_saved, num_skipped).
    """
    # Load all modalities + segmentation
    mods = [load_modality(patient_dir, m) for m in config.MODALITIES]
    seg = load_seg(patient_dir)

    # Normalize each modality
    mods = [zscore_normalize(m) for m in mods]

    # Stack channels: shape (H, W, D, C)
    volume = np.stack(mods, axis=-1)

    # Binarize: any non-zero label is "whole tumor"
    if config.BINARY_TUMOR:
        seg = (seg > 0).astype(np.uint8)

    saved, skipped = 0, 0
    pid = patient_dir.name

    for z in range(config.SLICE_START, min(config.SLICE_END, volume.shape[2])):
        mask_slice = seg[:, :, z]
        # Skip mostly-empty slices to balance dataset
        if mask_slice.sum() < 10:
            skipped += 1
            continue

        img_slice = volume[:, :, z, :]  # (H, W, C)

        # Center-crop to IMG_SIZE
        img_slice = center_crop(img_slice, config.IMG_SIZE)
        mask_slice = center_crop(mask_slice, config.IMG_SIZE)

        out_name = f"{pid}_z{z:03d}"
        np.save(out_img_dir / f"{out_name}.npy", img_slice.astype(np.float32))
        np.save(out_mask_dir / f"{out_name}.npy", mask_slice.astype(np.uint8))
        saved += 1

    return saved, skipped


def main() -> None:
    log.info("Starting BraTS 2020 preprocessing")
    log.info("Modalities: %s", config.MODALITIES)
    log.info("Slice range: %d..%d", config.SLICE_START, config.SLICE_END)
    log.info("Image size: %d", config.IMG_SIZE)

    patients = find_patients(config.DATA_RAW)

    # Patient-level split — prevents data leakage
    train_patients, val_patients = train_test_split(
        patients, test_size=config.VAL_RATIO, random_state=config.RANDOM_SEED
    )
    log.info("Train: %d patients, Val: %d patients", len(train_patients), len(val_patients))

    # Clean and create output directories
    if config.DATA_PROCESSED.exists():
        log.info("Wiping previous processed data at %s", config.DATA_PROCESSED)
        shutil.rmtree(config.DATA_PROCESSED)

    splits = {"train": train_patients, "val": val_patients}
    for split_name, patient_list in splits.items():
        img_dir = config.DATA_PROCESSED / split_name / "images"
        mask_dir = config.DATA_PROCESSED / split_name / "masks"
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        total_saved, total_skipped = 0, 0
        for p in tqdm(patient_list, desc=f"Processing {split_name}"):
            try:
                s, sk = process_patient(p, img_dir, mask_dir)
                total_saved += s
                total_skipped += sk
            except Exception as e:
                log.warning("Failed on %s: %s", p.name, e)

        log.info("%s: saved %d slices, skipped %d empty", split_name, total_saved, total_skipped)

    log.info("Done. Processed data is at %s", config.DATA_PROCESSED)


if __name__ == "__main__":
    main()