"""
Baseline U-Net for binary brain tumor segmentation.

Standard encoder-decoder with skip connections.
This is the baseline for thesis comparison (Slide 11, step 4).

Reference: Ronneberger et al., "U-Net: Convolutional Networks for
Biomedical Image Segmentation" (2015).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src import config


class DoubleConv(nn.Module):
    """Two consecutive Conv-BN-ReLU blocks (the U-Net building block)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """Downscale + double conv."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Upscale + concat skip + double conv."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        # Transposed conv for upsampling. Halves channels.
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)  # in_ch because we concat skip

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle off-by-one shape mismatches (rare with 224x224 but safe)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    Baseline U-Net.

    Args:
        in_channels:  number of input modalities (default: from config)
        num_classes:  number of output classes (1 for binary)
        base_ch:      base channel count (default: from config)
    """

    def __init__(
        self,
        in_channels: int = config.NUM_INPUT_CHANNELS,
        num_classes: int = config.NUM_CLASSES,
        base_ch: int = config.UNET_BASE_CHANNELS,
    ) -> None:
        super().__init__()
        c = base_ch  # 32, 64, 128, 256, 512

        # Encoder
        self.in_conv = DoubleConv(in_channels, c)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)

        # Bottleneck
        self.bottleneck = Down(c * 8, c * 16)

        # Decoder
        self.up1 = Up(c * 16, c * 8)
        self.up2 = Up(c * 8, c * 4)
        self.up3 = Up(c * 4, c * 2)
        self.up4 = Up(c * 2, c)

        # Output
        self.out_conv = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.in_conv(x)        # (B,  c,   H,    W)
        x2 = self.down1(x1)         # (B, 2c,   H/2,  W/2)
        x3 = self.down2(x2)         # (B, 4c,   H/4,  W/4)
        x4 = self.down3(x3)         # (B, 8c,   H/8,  W/8)
        x5 = self.bottleneck(x4)    # (B, 16c,  H/16, W/16)

        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.out_conv(x)     # (B, num_classes, H, W)


if __name__ == "__main__":
    # Quick smoke test
    model = UNet()
    x = torch.randn(2, config.NUM_INPUT_CHANNELS, config.IMG_SIZE, config.IMG_SIZE)
    y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,}")
