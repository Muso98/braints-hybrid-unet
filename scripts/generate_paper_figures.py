"""
Generate all figures needed for the LaTeX paper.
Run from project root:
    python scripts/generate_paper_figures.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as FancyArrow
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "paper_figures"
FIGURES.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 150,
})

BLUE  = "#3A7EC6"
ORANGE = "#E07B39"
GREEN  = "#3CA66A"
GRAY   = "#888888"


# ─── Figure 1: Architecture Diagram ──────────────────────────────────────────
def fig_architecture():
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_facecolor("#F8F9FA")
    fig.patch.set_facecolor("#F8F9FA")

    def box(x, y, w, h, color, label, sub="", fontsize=9):
        rect = plt.Rectangle((x, y), w, h, linewidth=1.5,
                              edgecolor="white", facecolor=color, zorder=3, alpha=0.92)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2 + (0.15 if sub else 0), label,
                ha="center", va="center", fontsize=fontsize,
                color="white", fontweight="bold", zorder=4)
        if sub:
            ax.text(x + w / 2, y + h / 2 - 0.25, sub,
                    ha="center", va="center", fontsize=7.5,
                    color="white", alpha=0.85, zorder=4)

    def arrow(x1, x2, y=2.5, color="#444"):
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5), zorder=5)

    def skip_arrow(x, y_from, y_to, color="#aaa"):
        ax.annotate("", xy=(x, y_to), xytext=(x, y_from),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2,
                                   linestyle="dashed"), zorder=5)

    # ── Input
    box(0.1, 2.0, 1.1, 1.0, "#555566", "Input", "B×2×224²", fontsize=8)
    arrow(1.2, 1.55)

    # ── Encoder blocks
    enc_colors = ["#2E6DA4", "#1F5490", "#14407A", "#0D2F5E"]
    enc_labels = ["Enc-1\n32ch\n224²", "Enc-2\n64ch\n112²", "Enc-3\n128ch\n56²", "Enc-4\n256ch\n28²"]
    enc_x = [1.55, 2.65, 3.75, 4.85]
    for i, (xi, col, lbl) in enumerate(zip(enc_x, enc_colors, enc_labels)):
        box(xi, 1.9, 1.0, 1.2, col, lbl, fontsize=7.5)
        if i < 3:
            arrow(xi + 1.0, xi + 1.1)

    # Bottleneck arrow
    arrow(5.85, 6.15)

    # ── Transformer Bottleneck
    box(6.15, 1.7, 1.7, 1.6, "#7B3FA0",
        "Transformer\nBottleneck", "L=4 blocks\n196 tokens, d=256", fontsize=8)
    arrow(7.85, 8.2)

    # ── Decoder blocks
    dec_colors = ["#1A6B45", "#22885A", "#2CA572", "#36C285"]
    dec_labels = ["Dec-4\n256ch\n28²", "Dec-3\n128ch\n56²", "Dec-2\n64ch\n112²", "Dec-1\n32ch\n224²"]
    dec_x = [8.2, 9.3, 10.4, 11.5]
    for i, (xi, col, lbl) in enumerate(zip(dec_x, dec_colors, dec_labels)):
        box(xi, 1.9, 1.0, 1.2, col, lbl, fontsize=7.5)
        if i < 3:
            arrow(xi + 1.0, xi + 1.1)

    # ── Output head
    arrow(12.5, 12.7)
    box(12.7, 2.05, 1.1, 0.9, "#C62828", "Output\n1×1 Conv", "B×1×224²", fontsize=8)

    # ── Skip connections (dashed arcs)
    skip_pairs = [(1.55+0.5, 11.5+0.5), (2.65+0.5, 10.4+0.5),
                  (3.75+0.5, 9.3+0.5),  (4.85+0.5, 8.2+0.5)]
    for xe, xd in skip_pairs:
        y_top = 3.15
        ax.annotate("", xy=(xd, y_top), xytext=(xe, y_top),
                    arrowprops=dict(arrowstyle="-|>", color="#AAAACC",
                                   lw=1.1, linestyle="dashed",
                                   connectionstyle="arc3,rad=-0.25"), zorder=2)

    # ── Auxiliary head markers
    for xi, label in zip([9.3, 10.4, 11.5], ["Aux\n56²", "Aux\n112²", "Aux\n224²"]):
        ax.text(xi + 0.5, 1.65, label, ha="center", va="top",
                fontsize=7, color="#E8A000", fontweight="bold")
        ax.plot([xi + 0.5], [1.82], marker="v", ms=6, color="#E8A000", zorder=5)

    # Legend
    legend_items = [
        mpatches.Patch(color="#2E6DA4", label="CNN Encoder"),
        mpatches.Patch(color="#7B3FA0", label="Transformer Bottleneck"),
        mpatches.Patch(color="#1A6B45", label="CNN Decoder"),
        mpatches.Patch(color="#E8A000", label="Deep Supervision Heads"),
        mpatches.Patch(color="#AAAACC", label="Skip Connections"),
    ]
    ax.legend(handles=legend_items, loc="lower center", ncol=5,
              fontsize=8.5, framealpha=0.9, bbox_to_anchor=(0.5, -0.04))

    ax.set_title("Figure 1: Hybrid U-Net Transformer Architecture with Deep Supervision",
                 fontsize=12, pad=10, fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out = FIGURES / "fig1_architecture.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ─── Figure 2: Quantitative comparison (already exists, copy + reformat) ──────
def fig_metrics():
    models  = ["Baseline\nU-Net", "Hybrid U-Net\nTransformer"]
    dice    = [0.8339, 0.8409]
    iou     = [0.7512, 0.7577]
    hd      = [14.984, 14.266]
    std_d   = [0.1993, 0.1853]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    fig.suptitle("Figure 2: Quantitative Comparison on BraTS 2020 Validation Set (n=1,176 slices)",
                 fontsize=11, fontweight="bold", y=1.01)
    colors = [BLUE, ORANGE]

    # Dice
    bars = axes[0].bar(models, dice, color=colors, width=0.45, edgecolor="white", linewidth=1.2)
    for bar, v in zip(bars, dice):
        axes[0].text(bar.get_x() + bar.get_width()/2, v + 0.002,
                     f"{v:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    axes[0].set_ylim(0.80, 0.86)
    axes[0].set_ylabel("Dice Score")
    axes[0].set_title("Mean Dice (↑ better)", fontweight="bold")
    axes[0].yaxis.grid(True, alpha=0.4)
    axes[0].set_axisbelow(True)

    # IoU
    bars = axes[1].bar(models, iou, color=colors, width=0.45, edgecolor="white", linewidth=1.2)
    for bar, v in zip(bars, iou):
        axes[1].text(bar.get_x() + bar.get_width()/2, v + 0.001,
                     f"{v:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    axes[1].set_ylim(0.73, 0.78)
    axes[1].set_ylabel("IoU Score")
    axes[1].set_title("Mean IoU (↑ better)", fontweight="bold")
    axes[1].yaxis.grid(True, alpha=0.4)
    axes[1].set_axisbelow(True)

    # Hausdorff
    bars = axes[2].bar(models, hd, color=colors, width=0.45, edgecolor="white", linewidth=1.2)
    for bar, v in zip(bars, hd):
        axes[2].text(bar.get_x() + bar.get_width()/2, v + 0.05,
                     f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    axes[2].set_ylim(13.5, 15.5)
    axes[2].set_ylabel("Hausdorff Distance (pixels)")
    axes[2].set_title("Mean Hausdorff Distance (↓ better)", fontweight="bold")
    axes[2].yaxis.grid(True, alpha=0.4)
    axes[2].set_axisbelow(True)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out = FIGURES / "fig2_metrics.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ─── Figure 3: Dice Distribution ──────────────────────────────────────────────
def fig_distribution():
    import pandas as pd
    unet_csv   = RESULTS / "unet_per_slice.csv"
    hybrid_csv = RESULTS / "hybrid_per_slice.csv"
    if not unet_csv.exists() or not hybrid_csv.exists():
        print("Per-slice CSVs not found, skipping distribution figure.")
        return
    unet_df   = pd.read_csv(unet_csv)
    hybrid_df = pd.read_csv(hybrid_csv)
    unet_dice   = unet_df["dice"].values
    hybrid_dice = hybrid_df["dice"].values

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bins = np.linspace(0, 1, 35)
    ax.hist(unet_dice,   bins=bins, alpha=0.65, color=BLUE,   label=f"Baseline U-Net  (μ={unet_dice.mean():.4f}, σ={unet_dice.std():.4f})")
    ax.hist(hybrid_dice, bins=bins, alpha=0.65, color=ORANGE, label=f"Hybrid Transformer (μ={hybrid_dice.mean():.4f}, σ={hybrid_dice.std():.4f})")
    ax.axvline(unet_dice.mean(),   color=BLUE,   linestyle="--", linewidth=1.8, alpha=0.9)
    ax.axvline(hybrid_dice.mean(), color=ORANGE, linestyle="--", linewidth=1.8, alpha=0.9)
    ax.set_xlabel("Per-slice Dice Score")
    ax.set_ylabel("Number of Slices")
    ax.set_title("Figure 3: Per-Slice Dice Score Distribution (n = 1,176 slices)", fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, alpha=0.35)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = FIGURES / "fig3_distribution.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ─── Figure 4: Qualitative comparison (UNet vs Hybrid side-by-side) ───────────
def fig_qualitative():
    sample_ids = [4, 1, 7]   # pick 3 visually good samples
    n = len(sample_ids)
    fig, axes = plt.subplots(n, 8, figsize=(18, 3.5 * n))
    fig.patch.set_facecolor("#1A1A2E")

    col_titles = ["FLAIR Input", "Ground Truth", "U-Net Prediction", "U-Net Overlay",
                  "FLAIR Input", "Ground Truth", "Hybrid Prediction", "Hybrid Overlay"]

    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, color="white", fontsize=9, fontweight="bold", pad=4)

    for row, sid in enumerate(sample_ids):
        unet_img   = np.array(Image.open(RESULTS / "unet_predictions"   / f"sample_{sid:03d}.png"))
        hybrid_img = np.array(Image.open(RESULTS / "hybrid_predictions" / f"sample_{sid:03d}.png"))

        # Each prediction image has 4 panels side-by-side: FLAIR | GT | Pred | Overlay
        W = unet_img.shape[1] // 4
        unet_panels   = [unet_img[:, i*W:(i+1)*W] for i in range(4)]
        hybrid_panels = [hybrid_img[:, i*W:(i+1)*W] for i in range(4)]

        for col, panel in enumerate(unet_panels + hybrid_panels):
            axes[row, col].imshow(panel, cmap="gray" if panel.ndim == 2 else None)
            axes[row, col].axis("off")

        # Row label
        axes[row, 0].set_ylabel(f"Sample {sid}", color="white", fontsize=9, labelpad=6)

    # Separator line between UNet and Hybrid columns
    line_x = 0.505
    fig.add_artist(plt.Line2D([line_x, line_x], [0.02, 0.95],
                              transform=fig.transFigure, color="gray",
                              linewidth=1.5, linestyle="--"))

    axes[0, 0].annotate("← Baseline U-Net →", xy=(0.25, 1.06),
                         xycoords="axes fraction", fontsize=10,
                         color="#7EC8E3", fontweight="bold", ha="center")
    axes[0, 4].annotate("← Hybrid U-Net Transformer →", xy=(0.25, 1.06),
                         xycoords="axes fraction", fontsize=10,
                         color="#FFB347", fontweight="bold", ha="center")

    fig.suptitle("Figure 4: Qualitative Comparison — U-Net vs. Hybrid U-Net Transformer on BraTS 2020 Validation",
                 color="white", fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout(pad=0.5)
    out = FIGURES / "fig4_qualitative.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {out}")


# ─── Figure 5: Ablation bar chart ─────────────────────────────────────────────
def fig_ablation():
    configs = ["Baseline\nU-Net", "Hybrid\n(L=4 Transformer)"]
    dice_vals = [0.8339, 0.8409]
    hd_vals   = [14.984, 14.266]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    fig.suptitle("Figure 5: Ablation Study — Contribution of Transformer Bottleneck",
                 fontsize=11, fontweight="bold")

    bar_colors = [BLUE, ORANGE]
    bars = ax1.barh(configs, dice_vals, color=bar_colors, height=0.4, edgecolor="white")
    for bar, v in zip(bars, dice_vals):
        ax1.text(v + 0.0005, bar.get_y() + bar.get_height()/2,
                 f"{v:.4f}", va="center", fontsize=10, fontweight="bold")
    ax1.set_xlim(0.82, 0.855)
    ax1.set_xlabel("Mean Dice Score")
    ax1.set_title("Dice Score (↑ better)", fontweight="bold")
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.xaxis.grid(True, alpha=0.4)
    ax1.set_axisbelow(True)

    bars = ax2.barh(configs, hd_vals, color=bar_colors, height=0.4, edgecolor="white")
    for bar, v in zip(bars, hd_vals):
        ax2.text(v + 0.05, bar.get_y() + bar.get_height()/2,
                 f"{v:.3f} px", va="center", fontsize=10, fontweight="bold")
    ax2.set_xlim(13.5, 15.5)
    ax2.set_xlabel("Hausdorff Distance (pixels)")
    ax2.set_title("Hausdorff Distance (↓ better)", fontweight="bold")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.xaxis.grid(True, alpha=0.4)
    ax2.set_axisbelow(True)

    fig.tight_layout()
    out = FIGURES / "fig5_ablation.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    print("Generating paper figures...")
    fig_architecture()
    fig_metrics()
    fig_distribution()
    fig_qualitative()
    fig_ablation()
    print(f"\nAll figures saved to: {FIGURES}")
    print("Files:")
    for f in sorted(FIGURES.glob("*.png")):
        print(f"  {f.name}  ({f.stat().st_size // 1024} KB)")
