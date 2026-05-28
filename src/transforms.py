"""
Augmentations for training. We use albumentations because it handles
image+mask jointly and is fast.

Augmentation choices match what BraTS papers commonly do:
- geometric: flips, rotations, elastic transform
- intensity: gaussian noise, brightness/contrast (mild — MRI is already normalized)

This addresses the 'limited data' part of the thesis.
"""
import albumentations as A


def get_train_transforms() -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(0.0, 0.05),
                rotate=(-15, 15),
                p=0.5,
            ),
            A.ElasticTransform(alpha=20, sigma=5, p=0.2),
            A.GaussNoise(p=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        ]
    )


def get_val_transforms():
    # No augmentation at validation — dataset handles None correctly
    return None