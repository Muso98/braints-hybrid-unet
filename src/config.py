"""
Central configuration for the Brain Tumor Segmentation pipeline.
All paths and hyperparameters live here. Tuned for RTX 4060 8GB VRAM.
"""

from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = Path(r"D:\DATASETS\MICCAI_BraTS2020_TrainingData")
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
LOG_DIR = PROJECT_ROOT / "logs"

for d in [DATA_PROCESSED, CHECKPOINT_DIR, RESULTS_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────
MODALITIES = ["flair", "t1ce"]
NUM_INPUT_CHANNELS = len(MODALITIES)

IMG_SIZE = 224

SLICE_START = 50
SLICE_END = 130

VAL_RATIO = 0.2
RANDOM_SEED = 42

NUM_CLASSES = 1
BINARY_TUMOR = True

# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
BATCH_SIZE = 16
BATCH_SIZE_HYBRID = 8

EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# Loss weights (V1 — original setting that gave best results)
DICE_WEIGHT = 0.7
FOCAL_WEIGHT = 0.3
FOCAL_GAMMA = 2.0

# Boundary loss disabled in V1
BOUNDARY_WEIGHT = 0.0

# Deep supervision disabled in V1
USE_DEEP_SUPERVISION = False
DEEP_SUPERVISION_WEIGHTS = [1.0, 0.5, 0.25, 0.125]

OPTIMIZER = "adamw"
SCHEDULER = "cosine"
WARMUP_EPOCHS = 3

USE_AMP = True

PATIENCE = 10

# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
UNET_BASE_CHANNELS = 32

# V1 transformer config (4 blocks)
TRANSFORMER_DIM = 256
TRANSFORMER_HEADS = 8
TRANSFORMER_DEPTH = 4
TRANSFORMER_MLP_RATIO = 4
TRANSFORMER_DROPOUT = 0.1

# ─────────────────────────────────────────────────────────────────────────────
# DataLoader
# ─────────────────────────────────────────────────────────────────────────────
import platform
NUM_WORKERS = 2 if platform.system() != "Windows" else 0
PIN_MEMORY = True

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
LOG_EVERY_N_STEPS = 20
SAVE_PREDICTIONS_EVERY_N_EPOCHS = 5