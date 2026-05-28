"""
Generate the thesis comparison report from evaluation results.

Reads:
  - results/unet_metrics.json
  - results/hybrid_metrics.json
  - results/unet_per_slice.csv
  - results/hybrid_per_slice.csv

Writes:
  - results/comparison.csv      → table for the thesis
  - results/comparison.png      → bar chart Dice/IoU
  - results/dice_distribution.png → histogram of per-slice Dice

Run:
    python -m src.report
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def load_metrics(model_name: str) -> dict:
    path = config.RESULTS_DIR / f"{model_name}_metrics.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No metrics for {model_name} at {path}. Run "
            f"`python -m src.evaluate --model {model_name} --checkpoint checkpoints/{model_name}_best.pth` first."
        )
    with open(path) as f:
        return json.load(f)


def load_per_slice(model_name: str) -> pd.DataFrame:
    path = config.RESULTS_DIR / f"{model_name}_per_slice.csv"
    return pd.read_csv(path)


def main() -> None:
    unet = load_metrics("unet")
    hybrid = load_metrics("hybrid")

    # ── Build comparison table ──
    rows = [
        {
            "Model": "Baseline U-Net",
            "Mean Dice": unet["mean_dice"],
            "Median Dice": unet["median_dice"],
            "Mean IoU": unet["mean_iou"],
            "Mean Hausdorff (px)": unet["mean_hausdorff"],
            "N slices": unet["num_slices"],
        },
        {
            "Model": "Hybrid U-Net Transformer",
            "Mean Dice": hybrid["mean_dice"],
            "Median Dice": hybrid["median_dice"],
            "Mean IoU": hybrid["mean_iou"],
            "Mean Hausdorff (px)": hybrid["mean_hausdorff"],
            "N slices": hybrid["num_slices"],
        },
    ]

    df = pd.DataFrame(rows)

    # Improvement row
    impr = {
        "Model": "Improvement (Hybrid - U-Net)",
        "Mean Dice": hybrid["mean_dice"] - unet["mean_dice"],
        "Median Dice": hybrid["median_dice"] - unet["median_dice"],
        "Mean IoU": hybrid["mean_iou"] - unet["mean_iou"],
        "Mean Hausdorff (px)": hybrid["mean_hausdorff"] - unet["mean_hausdorff"],
        "N slices": "—",
    }
    df = pd.concat([df, pd.DataFrame([impr])], ignore_index=True)

    out_csv = config.RESULTS_DIR / "comparison.csv"
    df.to_csv(out_csv, index=False, float_format="%.4f")
    log.info("Comparison table → %s", out_csv)
    print("\n" + df.to_string(index=False))

    # ── Bar chart ──
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

    metrics_names = ["Dice", "IoU"]
    unet_vals = [unet["mean_dice"], unet["mean_iou"]]
    hybrid_vals = [hybrid["mean_dice"], hybrid["mean_iou"]]
    x = np.arange(len(metrics_names))
    w = 0.35

    bars1 = ax[0].bar(x - w / 2, unet_vals, w, label="Baseline U-Net", color="#4C72B0")
    bars2 = ax[0].bar(x + w / 2, hybrid_vals, w, label="Hybrid U-Net Transformer", color="#DD8452")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(metrics_names)
    ax[0].set_ylabel("Score")
    ax[0].set_title("Mean Dice & IoU")
    ax[0].set_ylim(0, 1)
    ax[0].legend()
    ax[0].grid(axis="y", alpha=0.3)
    for bars in (bars1, bars2):
        for b in bars:
            ax[0].annotate(f"{b.get_height():.3f}", xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                           xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    # Hausdorff (lower is better)
    ax[1].bar(["U-Net", "Hybrid"],
              [unet["mean_hausdorff"], hybrid["mean_hausdorff"]],
              color=["#4C72B0", "#DD8452"])
    ax[1].set_ylabel("Hausdorff distance (pixels)")
    ax[1].set_title("Mean Hausdorff Distance (lower = better)")
    ax[1].grid(axis="y", alpha=0.3)
    for i, v in enumerate([unet["mean_hausdorff"], hybrid["mean_hausdorff"]]):
        ax[1].annotate(f"{v:.2f}", xy=(i, v), xytext=(0, 3),
                       textcoords="offset points", ha="center", fontsize=9)

    plt.tight_layout()
    fig_path = config.RESULTS_DIR / "comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Comparison figure → %s", fig_path)

    # ── Per-slice Dice distribution ──
    try:
        df_unet = load_per_slice("unet")
        df_hybrid = load_per_slice("hybrid")

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(df_unet["dice"], bins=40, alpha=0.55, label="Baseline U-Net", color="#4C72B0", edgecolor="black")
        ax.hist(df_hybrid["dice"], bins=40, alpha=0.55, label="Hybrid U-Net Transformer", color="#DD8452", edgecolor="black")
        ax.set_xlabel("Dice score per slice")
        ax.set_ylabel("Number of slices")
        ax.set_title("Per-slice Dice distribution")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        dist_path = config.RESULTS_DIR / "dice_distribution.png"
        plt.savefig(dist_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Dice distribution → %s", dist_path)
    except FileNotFoundError as e:
        log.warning("Skipping per-slice distribution plot: %s", e)

    print("\nAll done. Use these files in your thesis:")
    print(f"  - {out_csv}")
    print(f"  - {fig_path}")


if __name__ == "__main__":
    main()
