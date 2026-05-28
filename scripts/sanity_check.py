"""
Sanity check the full setup before launching long training.

Verifies:
  1. CUDA + PyTorch
  2. All imports work
  3. Both models forward pass (handles deep supervision)
  4. Loss + metrics compute
  5. Dataset can be loaded (if preprocessing was done)
  6. One full train step works on GPU

Run:
    python -m scripts.sanity_check
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src import config


def check_torch() -> None:
    print("─" * 60)
    print(" 1. PyTorch + CUDA")
    print("─" * 60)
    print(f"   torch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   Device: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"   CUDA: {torch.version.cuda}")


def check_models() -> None:
    print()
    print("─" * 60)
    print(" 2. Models")
    print("─" * 60)
    from src.models.hybrid_unet import HybridUNetTransformer
    from src.models.unet import UNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, config.NUM_INPUT_CHANNELS, config.IMG_SIZE, config.IMG_SIZE, device=device)

    for name, cls in [("UNet", UNet), ("HybridUNetTransformer", HybridUNetTransformer)]:
        m = cls().to(device)

        # Test in eval mode (single output) — for inference behavior
        m.eval()
        with torch.no_grad():
            y_eval = m(x)

        n_params = sum(p.numel() for p in m.parameters())

        # Handle both single tensor and list outputs (deep supervision)
        if isinstance(y_eval, list):
            shapes_str = f"list of {len(y_eval)} tensors, main = {tuple(y_eval[0].shape)}"
        else:
            shapes_str = f"{tuple(y_eval.shape)}"

        print(f"   {name:25s}: input {tuple(x.shape)} → output {shapes_str} | {n_params/1e6:.2f}M params")

        # Also check training mode for hybrid (deep supervision)
        if name == "HybridUNetTransformer" and getattr(config, "USE_DEEP_SUPERVISION", False):
            m.train()
            y_train = m(x)
            if isinstance(y_train, list):
                print(f"   {'  → train mode':25s}: deep supervision active, {len(y_train)} outputs")
            else:
                print(f"   {'  → train mode':25s}: single output (deep supervision OFF)")

        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()


def check_loss_metrics() -> None:
    print()
    print("─" * 60)
    print(" 3. Loss + metrics")
    print("─" * 60)
    from src.losses import CombinedLoss
    from src.metrics import dice_score, iou_score

    logits = torch.randn(4, 1, 64, 64)
    target = (torch.rand(4, 1, 64, 64) > 0.5).float()
    loss = CombinedLoss()(logits, target)
    d = dice_score(logits, target)
    i = iou_score(logits, target)
    print(f"   CombinedLoss : {loss.item():.4f}")
    print(f"   Dice         : {d:.4f}")
    print(f"   IoU          : {i:.4f}")

    if getattr(config, "BOUNDARY_WEIGHT", 0.0) > 0:
        from src.losses import BoundaryLoss
        bl = BoundaryLoss()(logits, target)
        print(f"   BoundaryLoss : {bl.item():.4f}  (active, weight={config.BOUNDARY_WEIGHT})")


def check_dataset() -> None:
    print()
    print("─" * 60)
    print(" 4. Dataset")
    print("─" * 60)
    if not config.DATA_PROCESSED.exists() or not (config.DATA_PROCESSED / "train" / "images").exists():
        print("   ⚠  Processed data not found.")
        print("   Run `python -m src.preprocess` after downloading BraTS 2020.")
        print("   (Skipping dataset check.)")
        return

    from src.dataset import BraTSDataset
    from src.transforms import get_train_transforms

    ds = BraTSDataset(split="train", transform=get_train_transforms())
    img, mask = ds[0]
    print(f"   Train samples : {len(ds)}")
    print(f"   First sample  : img {tuple(img.shape)}, mask {tuple(mask.shape)}")
    print(f"   img dtype     : {img.dtype}, mask dtype: {mask.dtype}")
    print(f"   img range     : [{img.min().item():.3f}, {img.max().item():.3f}]")
    print(f"   mask values   : {torch.unique(mask).tolist()}")


def check_train_step() -> None:
    print()
    print("─" * 60)
    print(" 5. One training step (GPU)")
    print("─" * 60)
    from src.losses import CombinedLoss
    from src.models.hybrid_unet import HybridUNetTransformer

    if not torch.cuda.is_available():
        print("   ⚠  CUDA not available, skipping GPU test.")
        return

    device = torch.device("cuda")
    model = HybridUNetTransformer().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    crit = CombinedLoss()

    x = torch.randn(4, config.NUM_INPUT_CHANNELS, config.IMG_SIZE, config.IMG_SIZE, device=device)
    y = (torch.rand(4, 1, config.IMG_SIZE, config.IMG_SIZE, device=device) > 0.5).float()

    model.train()
    optim.zero_grad()
    with torch.cuda.amp.autocast(enabled=config.USE_AMP):
        out = model(x)
        # Handle deep supervision: if list, sum weighted losses
        if isinstance(out, list):
            weights = config.DEEP_SUPERVISION_WEIGHTS[:len(out)]
            total_w = sum(weights)
            weights = [w / total_w for w in weights]
            loss = sum(w * crit(o, y) for w, o in zip(weights, out))
            print(f"   Deep supervision: {len(out)} outputs combined")
        else:
            loss = crit(out, y)

    loss.backward()
    optim.step()

    mem_alloc = torch.cuda.memory_allocated() / 1e9
    mem_reserved = torch.cuda.memory_reserved() / 1e9
    print(f"   Forward+backward+step OK")
    print(f"   Loss          : {loss.item():.4f}")
    print(f"   VRAM allocated: {mem_alloc:.2f} GB")
    print(f"   VRAM reserved : {mem_reserved:.2f} GB")


def main() -> None:
    print("\n" + "=" * 60)
    print("  BRAIN TUMOR SEGMENTATION — SANITY CHECK (v2)")
    print("=" * 60)

    check_torch()
    check_models()
    check_loss_metrics()
    check_dataset()
    check_train_step()

    print()
    print("=" * 60)
    print("  Sanity check complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()