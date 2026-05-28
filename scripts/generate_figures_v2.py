"""
Generate all high-quality paper figures matching the provided architecture style.
Run: python scripts/generate_figures_v2.py
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np
from PIL import Image
import pandas as pd

ROOT   = Path(__file__).resolve().parents[1]
FIGS   = ROOT / "paper_figures"
RES    = ROOT / "results"
FIGS.mkdir(exist_ok=True)

# ── Shared style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
})

ENC_LIGHT  = "#89BDD3"
ENC_DARK   = "#2E86AB"
DEC_COLOR  = "#1B998B"
BOT_COLOR  = "#E76F51"
OUT_COLOR  = "#2D6A4F"
INPUT_COLOR= "#D8E8F0"
SKIP_COLOR = "#AAAAAA"
ARROW_COLOR= "#E76F51"
TEXT_WHITE = "white"
TEXT_DARK  = "#2C3E50"
BG_COLOR   = "#FAFBFC"


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Architecture Diagram  (matches provided reference)
# ══════════════════════════════════════════════════════════════════════════════
def rounded_box(ax, x, y, w, h, color, text, subtext="", fontsize=13,
                radius=0.12, text_color="white", alpha=1.0):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle=f"round,pad=0,rounding_size={radius}",
                         linewidth=0, facecolor=color, alpha=alpha, zorder=3)
    ax.add_patch(box)
    if subtext:
        ax.text(x, y + h*0.12, text, ha="center", va="center",
                fontsize=fontsize, color=text_color, fontweight="bold", zorder=4)
        ax.text(x, y - h*0.22, subtext, ha="center", va="center",
                fontsize=fontsize - 3.5, color=text_color, alpha=0.85, zorder=4)
    else:
        ax.text(x, y, text, ha="center", va="center",
                fontsize=fontsize, color=text_color, fontweight="bold", zorder=4)


def caption(ax, x, y, txt, fontsize=7.5, color="#666666"):
    ax.text(x, y, txt, ha="center", va="top", fontsize=fontsize,
            color=color, zorder=4)


def fig_architecture():
    fig, ax = plt.subplots(figsize=(16, 8))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 16)
    ax.set_ylim(-0.5, 8.5)
    ax.axis("off")

    BW, BH = 1.05, 0.72   # block width / height
    ENC_X  = 2.5           # encoder centre-x
    DEC_X  = 13.5          # decoder centre-x
    BOT_Y  = 1.3           # bottleneck y

    # ── encoder y positions (top → bottom)
    enc_ys   = [7.2, 6.0, 4.8, 3.6, 2.4]
    enc_chs  = ["32", "64", "128", "256", "256"]
    enc_res  = ["224×224", "112×112", "56×56", "28×28", "14×14"]
    enc_cols = [ENC_LIGHT, ENC_LIGHT, ENC_DARK, ENC_DARK, ENC_DARK]

    # ── decoder y positions (bottom → top)
    dec_ys   = [2.4, 3.6, 4.8, 6.0, 7.2]
    dec_chs  = ["256", "256", "128", "64", "32"]
    dec_res  = ["14×14", "28×28", "56×56", "112×112", "224×224"]

    # ─── Input block ─────────────────────────────────────────────
    rounded_box(ax, ENC_X - 2.0, enc_ys[0], BW, BH,
                INPUT_COLOR, "2", "", fontsize=16, text_color=ENC_DARK)
    ax.text(ENC_X - 2.0, enc_ys[0] - BH/2 - 0.18, "224×224",
            ha="center", va="top", fontsize=7.5, color="#888")
    ax.text(ENC_X - 2.0, enc_ys[0] + BH/2 + 0.12, "Input (FLAIR + T1ce)",
            ha="center", va="bottom", fontsize=8, color=TEXT_DARK, fontweight="bold")
    ax.annotate("", xy=(ENC_X - BW/2 - 0.28, enc_ys[0]),
                xytext=(ENC_X - 2.0 + BW/2 + 0.05, enc_ys[0]),
                arrowprops=dict(arrowstyle="-|>", color=ENC_DARK, lw=1.6), zorder=5)

    # ─── Encoder blocks ───────────────────────────────────────────
    for i, (y, ch, res, col) in enumerate(zip(enc_ys, enc_chs, enc_res, enc_cols)):
        rounded_box(ax, ENC_X, y, BW, BH, col, ch, fontsize=16)
        ax.text(ENC_X, y - BH/2 - 0.18, res,
                ha="center", va="top", fontsize=7.5, color="#888")
        if i < len(enc_ys) - 1:
            ax.annotate("", xy=(ENC_X, enc_ys[i+1] + BH/2 + 0.04),
                        xytext=(ENC_X, y - BH/2 - 0.04),
                        arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.3), zorder=5)
            if i == 0:
                ax.text(ENC_X + 0.62, (y + enc_ys[i+1])/2, "MaxPool 2×2 + Conv",
                        ha="left", va="center", fontsize=7, color="#888", style="italic")

    # Section label
    ax.text(ENC_X, 8.1, "CNN Encoder", ha="center", fontsize=13,
            color=ENC_DARK, fontweight="bold")
    ax.text(ENC_X, 7.8, "local features", ha="center", fontsize=9,
            color="#888", style="italic")

    # ─── Decoder blocks ───────────────────────────────────────────
    for i, (y, ch, res) in enumerate(zip(dec_ys, dec_chs, dec_res)):
        rounded_box(ax, DEC_X, y, BW, BH, DEC_COLOR, ch, fontsize=16)
        ax.text(DEC_X, y - BH/2 - 0.18, res,
                ha="center", va="top", fontsize=7.5, color="#888")
        if i < len(dec_ys) - 1:
            ax.annotate("", xy=(DEC_X, dec_ys[i+1] - BH/2 - 0.04),
                        xytext=(DEC_X, y + BH/2 + 0.04),
                        arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.3), zorder=5)
            if i == 1:
                ax.text(DEC_X + 0.62, (y + dec_ys[i+1])/2, "UpConv 2×2 + Conv",
                        ha="left", va="center", fontsize=7, color="#888", style="italic")

    ax.text(DEC_X, 8.1, "U-Net Decoder", ha="center", fontsize=13,
            color=DEC_COLOR, fontweight="bold")
    ax.text(DEC_X, 7.8, "precise reconstruction", ha="center", fontsize=9,
            color="#888", style="italic")

    # ─── Output mask ──────────────────────────────────────────────
    rounded_box(ax, DEC_X + 2.0, 7.2, BW, BH, OUT_COLOR, "1", fontsize=16)
    ax.text(DEC_X + 2.0, 7.2 - BH/2 - 0.18, "224×224",
            ha="center", va="top", fontsize=7.5, color="#888")
    ax.text(DEC_X + 2.0, 7.2 + BH/2 + 0.12, "Output mask",
            ha="center", va="bottom", fontsize=8, color=TEXT_DARK, fontweight="bold")
    ax.annotate("", xy=(DEC_X + 2.0 - BW/2 - 0.05, 7.2),
                xytext=(DEC_X + BW/2 + 0.28, 7.2),
                arrowprops=dict(arrowstyle="-|>", color=DEC_COLOR, lw=1.6), zorder=5)

    # ─── Skip connections (dashed gray horizontal) ────────────────
    skip_pairs = list(zip(enc_ys[:-1], dec_ys[1:]))  # skip top 4 pairs
    ax.text(8.0, 7.55, "skip connections (concat)",
            ha="center", va="center", fontsize=8, color=SKIP_COLOR, style="italic")
    for ey, dy in skip_pairs:
        ax.plot([ENC_X + BW/2 + 0.05, DEC_X - BW/2 - 0.05], [ey, dy],
                linestyle="--", color=SKIP_COLOR, lw=1.1, zorder=2, alpha=0.7)

    # ─── Transformer Bottleneck ────────────────────────────────────
    bot_w, bot_h = 4.0, 1.4
    bot_x = 8.0
    box = FancyBboxPatch((bot_x - bot_w/2, BOT_Y - bot_h/2), bot_w, bot_h,
                         boxstyle="round,pad=0,rounding_size=0.18",
                         linewidth=0, facecolor=BOT_COLOR, zorder=3)
    ax.add_patch(box)
    ax.text(bot_x, BOT_Y + 0.22, "Transformer Bottleneck",
            ha="center", va="center", fontsize=14, color="white",
            fontweight="bold", zorder=4)
    ax.text(bot_x, BOT_Y - 0.22,
            "4 × Transformer Blocks\n8 attention heads  •  dim = 256  •  196 patch tokens",
            ha="center", va="center", fontsize=9, color="white", alpha=0.9, zorder=4,
            linespacing=1.6)
    ax.text(bot_x, BOT_Y - bot_h/2 - 0.22, "global context modelling",
            ha="center", va="top", fontsize=8.5, color=BOT_COLOR,
            style="italic", fontweight="bold")

    # ─── Curved arrows: Patch Embed & Reshape ────────────────────
    # Patch Embed: enc bottom → bot left
    ax.annotate("", xy=(bot_x - bot_w/2 - 0.05, BOT_Y),
                xytext=(ENC_X + BW/2 + 0.05, enc_ys[-1]),
                arrowprops=dict(arrowstyle="-|>", color=ARROW_COLOR, lw=2.0,
                                connectionstyle="arc3,rad=-0.28"), zorder=5)
    ax.text((ENC_X + bot_x)/2 - 0.4, (enc_ys[-1] + BOT_Y)/2 + 0.3,
            "Patch Embed", ha="center", va="center",
            fontsize=9, color=ARROW_COLOR, style="italic", fontweight="bold")

    # Reshape: bot right → dec bottom
    ax.annotate("", xy=(DEC_X - BW/2 - 0.05, dec_ys[0]),
                xytext=(bot_x + bot_w/2 + 0.05, BOT_Y),
                arrowprops=dict(arrowstyle="-|>", color=ARROW_COLOR, lw=2.0,
                                connectionstyle="arc3,rad=-0.28"), zorder=5)
    ax.text((bot_x + DEC_X)/2 + 0.4, (BOT_Y + dec_ys[0])/2 + 0.3,
            "Reshape", ha="center", va="center",
            fontsize=9, color=ARROW_COLOR, style="italic", fontweight="bold")

    # ─── Legend ───────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(color=ENC_LIGHT, label="Conv block (encoder)"),
        mpatches.Patch(color=DEC_COLOR, label="UpConv block (decoder)"),
        mpatches.Patch(color=BOT_COLOR, label="Transformer bottleneck"),
        mpatches.Patch(color=OUT_COLOR, label="Output mask"),
        mpatches.Patch(color=SKIP_COLOR, label="Skip connection", alpha=0.6),
    ]
    ax.legend(handles=legend_items, loc="lower center", ncol=5,
              fontsize=9, frameon=True, framealpha=0.95,
              bbox_to_anchor=(0.5, -0.06), edgecolor="#ddd")

    # ─── Title ────────────────────────────────────────────────────
    ax.set_title("Hybrid U-Net Transformer  —  Architecture Overview",
                 fontsize=15, fontweight="bold", color=TEXT_DARK, pad=14)

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out = FIGS / "fig1_architecture.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"✓ {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Metrics bar chart  (clean, no grid clutter)
# ══════════════════════════════════════════════════════════════════════════════
def fig_metrics():
    labels  = ["Baseline\nU-Net", "Hybrid U-Net\nTransformer"]
    dice    = [0.8339, 0.8409]
    iou     = [0.7512, 0.7577]
    hd      = [14.984, 14.266]
    colors  = [ENC_DARK, BOT_COLOR]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle("Quantitative Comparison — BraTS 2020 Validation Set  (n = 1,176 slices)",
                 fontsize=12, fontweight="bold", color=TEXT_DARK, y=1.01)

    def draw(ax, vals, ylims, ylabel, title, fmt="{:.4f}", suffix=""):
        bars = ax.bar(labels, vals, color=colors, width=0.42,
                      edgecolor="white", linewidth=1.5, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    v + (ylims[1]-ylims[0])*0.012,
                    fmt.format(v) + suffix,
                    ha="center", va="bottom", fontsize=11, fontweight="bold",
                    color=TEXT_DARK)
        ax.set_ylim(*ylims)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.spines[["top","right","left"]].set_visible(False)
        ax.yaxis.grid(True, alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.set_facecolor(BG_COLOR)
        for bar in bars:
            bar.set_linewidth(0)

    draw(axes[0], dice, (0.820, 0.850), "Dice Score",    "Mean Dice  ↑ better")
    draw(axes[1], iou,  (0.740, 0.770), "IoU Score",     "Mean IoU  ↑ better")
    draw(axes[2], hd,   (13.8,  15.4),  "Distance (px)", "Mean Hausdorff  ↓ better",
         fmt="{:.3f}", suffix=" px")

    fig.tight_layout()
    out = FIGS / "fig2_metrics.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"✓ {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Dice distribution  (clean histogram)
# ══════════════════════════════════════════════════════════════════════════════
def fig_distribution():
    udf = pd.read_csv(RES / "unet_per_slice.csv")
    hdf = pd.read_csv(RES / "hybrid_per_slice.csv")
    u = udf["dice"].values
    h = hdf["dice"].values

    fig, ax = plt.subplots(figsize=(10, 4.8))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    bins = np.linspace(0, 1, 36)
    ax.hist(u, bins=bins, alpha=0.65, color=ENC_DARK,  label=f"Baseline U-Net   (μ={u.mean():.4f}, σ={u.std():.4f})", zorder=3)
    ax.hist(h, bins=bins, alpha=0.65, color=BOT_COLOR, label=f"Hybrid Transformer (μ={h.mean():.4f}, σ={h.std():.4f})", zorder=3)
    ax.axvline(u.mean(), color=ENC_DARK,  ls="--", lw=2.0, alpha=0.9, zorder=4)
    ax.axvline(h.mean(), color=BOT_COLOR, ls="--", lw=2.0, alpha=0.9, zorder=4)

    ax.set_xlabel("Per-Slice Dice Score", fontsize=11)
    ax.set_ylabel("Number of Slices", fontsize=11)
    ax.set_title("Per-Slice Dice Score Distribution  (n = 1,176 slices)",
                 fontsize=12, fontweight="bold", pad=10)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.spines[["top","right"]].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # annotation
    ax.annotate("Most slices\nachieve Dice > 0.85",
                xy=(0.91, 270), xytext=(0.65, 230),
                fontsize=8.5, color="#555",
                arrowprops=dict(arrowstyle="->", color="#888", lw=1.0))

    fig.tight_layout()
    out = FIGS / "fig3_distribution.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"✓ {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Qualitative comparison  (polished, dark bg, 3 cases)
# ══════════════════════════════════════════════════════════════════════════════
def fig_qualitative():
    sample_ids = [4, 0, 7]
    n = len(sample_ids)
    DARK = "#16213E"

    fig = plt.figure(figsize=(18, 3.8 * n + 1.0))
    fig.patch.set_facecolor(DARK)

    col_labels = ["FLAIR Input", "Ground Truth", "U-Net Prediction", "U-Net Overlay",
                  "FLAIR Input", "Ground Truth", "Hybrid Prediction", "Hybrid Overlay"]

    axes = []
    for row in range(n):
        row_axes = []
        for col in range(8):
            ax = fig.add_subplot(n, 8, row * 8 + col + 1)
            row_axes.append(ax)
        axes.append(row_axes)

    # Column headers
    header_colors = [ENC_DARK]*4 + [BOT_COLOR]*4
    for col, (lbl, col_color) in enumerate(zip(col_labels, header_colors)):
        axes[0][col].set_title(lbl, color="white", fontsize=9.5,
                               fontweight="bold", pad=5)

    # Group labels
    fig.text(0.255, 0.97, "← Baseline U-Net →",
             ha="center", fontsize=12, color="#7EC8E3", fontweight="bold",
             transform=fig.transFigure)
    fig.text(0.745, 0.97, "← Hybrid U-Net Transformer →",
             ha="center", fontsize=12, color="#FFB347", fontweight="bold",
             transform=fig.transFigure)

    # Vertical separator
    fig.add_artist(plt.Line2D([0.505, 0.505], [0.02, 0.96],
                              transform=fig.transFigure,
                              color="#888", linewidth=1.5, linestyle="--"))

    for row, sid in enumerate(sample_ids):
        u_img = np.array(Image.open(RES / "unet_predictions"   / f"sample_{sid:03d}.png"))
        h_img = np.array(Image.open(RES / "hybrid_predictions" / f"sample_{sid:03d}.png"))
        W = u_img.shape[1] // 4
        panels = [u_img[:, i*W:(i+1)*W] for i in range(4)] + \
                 [h_img[:, i*W:(i+1)*W] for i in range(4)]

        for col, panel in enumerate(panels):
            ax = axes[row][col]
            ax.imshow(panel)
            ax.axis("off")
            for spine in ax.spines.values():
                spine.set_edgecolor("#333")
            ax.set_facecolor(DARK)

        # Row label
        axes[row][0].set_ylabel(f"Case {row+1}", color="white",
                                fontsize=10, labelpad=6, fontweight="bold")

    fig.suptitle("Qualitative Comparison — U-Net vs. Hybrid U-Net Transformer on BraTS 2020",
                 color="white", fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout(pad=0.6, rect=[0, 0, 1, 0.96])
    out = FIGS / "fig4_qualitative.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    print(f"✓ {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Ablation study  (horizontal bars, clean)
# ══════════════════════════════════════════════════════════════════════════════
def fig_ablation():
    configs    = ["U-Net Baseline", "Hybrid (L=4 Transformer)"]
    dice_vals  = [0.8339, 0.8409]
    hd_vals    = [14.984, 14.266]
    colors     = [ENC_DARK, BOT_COLOR]
    deltas     = ["+0.0070 Dice", "−0.718 px HD"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle("Ablation Study — Effect of Transformer Bottleneck",
                 fontsize=12, fontweight="bold", color=TEXT_DARK)

    def hbar(ax, vals, xlim, xlabel, title):
        bars = ax.barh(configs, vals, color=colors, height=0.38,
                       edgecolor="white", linewidth=1.2, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(v + (xlim[1]-xlim[0])*0.008,
                    bar.get_y() + bar.get_height()/2,
                    f"{v:.4f}", va="center", fontsize=11, fontweight="bold",
                    color=TEXT_DARK)
        ax.set_xlim(*xlim)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.spines[["top","right","bottom"]].set_visible(False)
        ax.xaxis.grid(True, alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0)
        ax.set_facecolor(BG_COLOR)

    hbar(ax1, dice_vals, (0.820, 0.852), "Mean Dice Score", "Mean Dice  ↑ better")
    hbar(ax2, hd_vals,   (13.8, 15.4),  "Hausdorff Distance (px)", "Hausdorff Distance  ↓ better")

    # Improvement annotation
    ax1.annotate("+0.0070", xy=(0.8409, 1), xytext=(0.836, 0.5),
                 fontsize=9, color=BOT_COLOR, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=BOT_COLOR, lw=1.2))

    fig.tight_layout()
    out = FIGS / "fig5_ablation.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"✓ {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating high-quality figures...\n")
    fig_architecture()
    fig_metrics()
    fig_distribution()
    fig_qualitative()
    fig_ablation()
    print("\nDone! All figures saved to:", FIGS)
