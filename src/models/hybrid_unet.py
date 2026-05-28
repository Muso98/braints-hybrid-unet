"""
Hybrid U-Net Transformer — IMPROVED with Deep Supervision.

Changes from v1:
  - Transformer depth doubled (4 → 8) via config
  - Deep supervision: auxiliary outputs at intermediate decoder stages
    (gradients flow better, especially under limited data)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src import config
from src.models.unet import DoubleConv, Down


# ─────────────────────────────────────────────────────────────────────────────
# Transformer components
# ─────────────────────────────────────────────────────────────────────────────
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        return self.dropout(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, num_tokens: int, dim: int) -> None:
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_embed


class _UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid model with Deep Supervision
# ─────────────────────────────────────────────────────────────────────────────
class HybridUNetTransformer(nn.Module):
    """
    Hybrid U-Net + Transformer with Deep Supervision.

    Outputs:
      - During training (and if deep_supervision=True): list of 4 outputs
        [main_output, aux_28x28, aux_56x56, aux_112x112]
      - During inference: single tensor (main output only)
    """

    def __init__(
        self,
        in_channels: int = config.NUM_INPUT_CHANNELS,
        num_classes: int = config.NUM_CLASSES,
        base_ch: int = config.UNET_BASE_CHANNELS,
        trans_dim: int = config.TRANSFORMER_DIM,
        trans_depth: int = config.TRANSFORMER_DEPTH,
        trans_heads: int = config.TRANSFORMER_HEADS,
        trans_mlp_ratio: int = config.TRANSFORMER_MLP_RATIO,
        trans_dropout: float = config.TRANSFORMER_DROPOUT,
        deep_supervision: bool = None,
    ) -> None:
        super().__init__()
        c = base_ch

        # Whether to use deep supervision. Default reads from config.
        if deep_supervision is None:
            deep_supervision = getattr(config, "USE_DEEP_SUPERVISION", False)
        self.deep_supervision = deep_supervision

        # ── Encoder ──
        self.in_conv = DoubleConv(in_channels, c)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)
        self.down4 = Down(c * 8, c * 8)

        # ── Patch embed ──
        self.proj_in = nn.Conv2d(c * 8, trans_dim, kernel_size=1)
        self.feat_size = config.IMG_SIZE // 16
        num_tokens = self.feat_size ** 2
        self.pos_embed = PositionalEncoding(num_tokens, trans_dim)
        self.tok_dropout = nn.Dropout(trans_dropout)

        # ── Transformer bottleneck ──
        self.transformer = nn.ModuleList(
            [
                TransformerBlock(trans_dim, trans_heads, trans_mlp_ratio, trans_dropout)
                for _ in range(trans_depth)
            ]
        )
        self.norm = nn.LayerNorm(trans_dim)
        self.proj_out = nn.Conv2d(trans_dim, c * 8, kernel_size=1)

        # ── Decoder ──
        self.up1 = _UpBlock(in_ch=c * 8, skip_ch=c * 8, out_ch=c * 4)   # 256→128, 28x28
        self.up2 = _UpBlock(in_ch=c * 4, skip_ch=c * 4, out_ch=c * 2)   # 128→64,  56x56
        self.up3 = _UpBlock(in_ch=c * 2, skip_ch=c * 2, out_ch=c)       # 64→32,   112x112
        self.up4 = _UpBlock(in_ch=c, skip_ch=c, out_ch=c)               # 32→32,   224x224

        # Main output head
        self.out_conv = nn.Conv2d(c, num_classes, kernel_size=1)

        # Auxiliary heads for deep supervision
        # Each takes the decoder feature map and produces a low-resolution prediction
        # which gets upsampled to full size in forward().
        if self.deep_supervision:
            self.aux_head_28 = nn.Conv2d(c * 4, num_classes, kernel_size=1)
            self.aux_head_56 = nn.Conv2d(c * 2, num_classes, kernel_size=1)
            self.aux_head_112 = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor):
        # ── Encoder ──
        x1 = self.in_conv(x)        # (B, 32,  224, 224)
        x2 = self.down1(x1)         # (B, 64,  112, 112)
        x3 = self.down2(x2)         # (B, 128, 56,  56)
        x4 = self.down3(x3)         # (B, 256, 28,  28)
        x5 = self.down4(x4)         # (B, 256, 14,  14)

        # ── Transformer bottleneck ──
        z = self.proj_in(x5)
        B, D, H, W = z.shape
        z = z.flatten(2).transpose(1, 2)
        z = self.pos_embed(z)
        z = self.tok_dropout(z)

        for block in self.transformer:
            z = block(z)
        z = self.norm(z)

        z = z.transpose(1, 2).reshape(B, D, H, W)
        z = self.proj_out(z)

        # ── Decoder ──
        d1 = self.up1(z, x4)        # (B, 128, 28, 28)
        d2 = self.up2(d1, x3)       # (B, 64, 56, 56)
        d3 = self.up3(d2, x2)       # (B, 32, 112, 112)
        d4 = self.up4(d3, x1)       # (B, 32, 224, 224)

        main_out = self.out_conv(d4)  # (B, num_classes, 224, 224)

        # During inference or if deep supervision is off, return single tensor
        if not self.deep_supervision or not self.training:
            return main_out

        # During training with deep supervision: return list of outputs
        # All upsampled to full input resolution for unified loss computation
        target_size = main_out.shape[-2:]
        aux1 = F.interpolate(self.aux_head_28(d1),  size=target_size, mode="bilinear", align_corners=False)
        aux2 = F.interpolate(self.aux_head_56(d2),  size=target_size, mode="bilinear", align_corners=False)
        aux3 = F.interpolate(self.aux_head_112(d3), size=target_size, mode="bilinear", align_corners=False)

        return [main_out, aux1, aux2, aux3]


if __name__ == "__main__":
    model = HybridUNetTransformer()
    x = torch.randn(2, config.NUM_INPUT_CHANNELS, config.IMG_SIZE, config.IMG_SIZE)

    model.train()
    y = model(x)
    print(f"Training mode: {len(y) if isinstance(y, list) else 1} outputs")
    if isinstance(y, list):
        for i, o in enumerate(y):
            print(f"  Output {i}: {tuple(o.shape)}")

    model.eval()
    y = model(x)
    print(f"Eval mode: {tuple(y.shape) if not isinstance(y, list) else len(y)}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,}")
