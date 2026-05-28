"""
Evaluate a trained model on the validation set.

Outputs:
  - results/{model}_metrics.json with mean Dice, IoU, Hausdorff
  - results/{model}_per_slice.csv with per-slice metrics
  - results/{model}_predictions/  with sample mask visualizations

Run:
    python -m src.evaluate --model unet --checkpoint checkpoints/unet_best.pth
    python -m src.evaluate --model hybrid --checkpoint checkpoints/hybrid_best.pth
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import config
from src.dataset import BraTSDataset
from src.metrics import dice_score, hausdorff_distance_2d, iou_score
from src.models.hybrid_unet import HybridUNetTransformer
from src.models.unet import UNet
from src.transforms import get_val_transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def build_model(name: str) -> torch.nn.Module:
    if name == "unet":
        return UNet()
    if name == "hybrid":
        return HybridUNetTransformer()
    raise ValueError(f"Unknown model: {name}")


def visualize_sample(img: np.ndarray, gt: np.ndarray, pred: np.ndarray, save_path: Path, dice: float) -> None:
    """Save a 1x4 figure: input, GT, prediction, overlay."""
    # img is (C, H, W) — show first modality (FLAIR)
    flair = img[0]
    # Normalize for display
    flair_disp = (flair - flair.min()) / (flair.max() - flair.min() + 1e-8)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(flair_disp, cmap="gray")
    axes[0].set_title("FLAIR")
    axes[0].axis("off")

    axes[1].imshow(gt, cmap="hot", vmin=0, vmax=1)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(pred, cmap="hot", vmin=0, vmax=1)
    axes[2].set_title(f"Prediction (Dice={dice:.3f})")
    axes[2].axis("off")

    axes[3].imshow(flair_disp, cmap="gray")
    axes[3].imshow(pred, cmap="Reds", alpha=0.4, vmin=0, vmax=1)
    axes[3].imshow(gt, cmap="Greens", alpha=0.3, vmin=0, vmax=1)
    axes[3].set_title("Overlay (R=Pred, G=GT)")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, model_name: str, num_viz: int = 16) -> dict:
    model.eval()

    all_dice, all_iou, all_hd = [], [], []
    per_slice_records = []

    viz_dir = config.RESULTS_DIR / f"{model_name}_predictions"
    viz_dir.mkdir(parents=True, exist_ok=True)
    num_saved_viz = 0

    pbar = tqdm(loader, desc="Evaluating")
    for batch_idx, (img, mask) in enumerate(pbar):
        img = img.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        logits = model(img)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        # Per-sample metrics
        for i in range(img.size(0)):
            d = dice_score(logits[i:i+1].float(), mask[i:i+1])
            ii = iou_score(logits[i:i+1].float(), mask[i:i+1])
            pred_np = preds[i, 0].cpu().numpy().astype(np.uint8)
            mask_np = mask[i, 0].cpu().numpy().astype(np.uint8)
            hd = hausdorff_distance_2d(pred_np, mask_np)

            all_dice.append(d)
            all_iou.append(ii)
            if not np.isnan(hd):
                all_hd.append(hd)

            per_slice_records.append({
                "batch_idx": batch_idx,
                "in_batch_idx": i,
                "dice": d,
                "iou": ii,
                "hausdorff": hd,
            })

            # Save a few visualizations across the dataset
            if num_saved_viz < num_viz and batch_idx % max(1, len(loader) // num_viz) == 0 and i == 0:
                visualize_sample(
                    img[i].cpu().numpy(),
                    mask_np,
                    pred_np,
                    viz_dir / f"sample_{num_saved_viz:03d}.png",
                    d,
                )
                num_saved_viz += 1

    metrics = {
        "model": model_name,
        "num_slices": len(all_dice),
        "mean_dice": float(np.mean(all_dice)),
        "mean_iou": float(np.mean(all_iou)),
        "mean_hausdorff": float(np.mean(all_hd)) if all_hd else float("nan"),
        "median_dice": float(np.median(all_dice)),
        "median_iou": float(np.median(all_iou)),
        "std_dice": float(np.std(all_dice)),
    }

    # Save metrics
    metrics_path = config.RESULTS_DIR / f"{model_name}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save per-slice CSV
    df = pd.DataFrame(per_slice_records)
    df.to_csv(config.RESULTS_DIR / f"{model_name}_per_slice.csv", index=False)

    log.info("Saved %d sample predictions → %s", num_saved_viz, viz_dir)
    log.info("Metrics → %s", metrics_path)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["unet", "hybrid"], required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-viz", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    val_ds = BraTSDataset(split="val", transform=get_val_transforms())
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    model = build_model(args.model).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    log.info("Loaded checkpoint %s (epoch %d, best Dice %.4f)",
             args.checkpoint, ckpt.get("epoch", -1), ckpt.get("best_dice", float("nan")))

    metrics = evaluate(model, val_loader, device, args.model, num_viz=args.num_viz)

    print("\n" + "=" * 50)
    print(f"  Results — {args.model}")
    print("=" * 50)
    print(f"  Mean Dice:      {metrics['mean_dice']:.4f}")
    print(f"  Median Dice:    {metrics['median_dice']:.4f}")
    print(f"  Mean IoU:       {metrics['mean_iou']:.4f}")
    print(f"  Mean Hausdorff: {metrics['mean_hausdorff']:.2f} px")
    print(f"  N slices:       {metrics['num_slices']}")
    print("=" * 50)


if __name__ == "__main__":
    main()
