"""
Training loop for both baseline U-Net and Hybrid U-Net Transformer.
IMPROVED: supports deep supervision (multi-output loss).
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src import config
from src.dataset import BraTSDataset
from src.losses import CombinedLoss
from src.metrics import dice_score, iou_score
from src.models.hybrid_unet import HybridUNetTransformer
from src.models.unet import UNet
from src.transforms import get_train_transforms, get_val_transforms

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


def build_model(name: str) -> nn.Module:
    if name == "unet":
        return UNet()
    if name == "hybrid":
        return HybridUNetTransformer()
    raise ValueError(f"Unknown model: {name}")


def get_batch_size(model_name: str) -> int:
    return config.BATCH_SIZE_HYBRID if model_name == "hybrid" else config.BATCH_SIZE


def compute_loss(criterion: nn.Module, output, target: torch.Tensor) -> torch.Tensor:
    """
    Compute loss, handling both single-output and deep-supervision cases.

    If output is a list (deep supervision), apply weighted sum of losses.
    Otherwise treat as single output.
    """
    if isinstance(output, list):
        weights = config.DEEP_SUPERVISION_WEIGHTS
        # Pad weights if too few; truncate if too many
        if len(weights) < len(output):
            weights = weights + [weights[-1]] * (len(output) - len(weights))
        weights = weights[:len(output)]
        # Normalize weights so total ≈ 1
        total_w = sum(weights)
        weights = [w / total_w for w in weights]

        total_loss = 0
        for w, out in zip(weights, output):
            total_loss = total_loss + w * criterion(out, target)
        return total_loss
    else:
        return criterion(output, target)


def get_main_output(output):
    """Extract main output for metric computation."""
    return output[0] if isinstance(output, list) else output


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
) -> dict:
    model.train()
    losses, dices, ious = [], [], []
    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)

    for step, (img, mask) in enumerate(pbar):
        img = img.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=config.USE_AMP):
            output = model(img)
            loss = compute_loss(criterion, output, mask)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            main = get_main_output(output)
            d = dice_score(main.float(), mask)
            i = iou_score(main.float(), mask)

        losses.append(loss.item())
        dices.append(d)
        ious.append(i)

        pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{d:.4f}")

        if step % config.LOG_EVERY_N_STEPS == 0:
            global_step = epoch * len(loader) + step
            writer.add_scalar("train/loss_step", loss.item(), global_step)
            writer.add_scalar("train/dice_step", d, global_step)

    return {
        "loss": sum(losses) / len(losses),
        "dice": sum(dices) / len(dices),
        "iou": sum(ious) / len(ious),
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> dict:
    model.eval()
    losses, dices, ious = [], [], []
    pbar = tqdm(loader, desc=f"Epoch {epoch} [val]", leave=False)

    for img, mask in pbar:
        img = img.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        with autocast(enabled=config.USE_AMP):
            output = model(img)  # in eval mode, model returns single tensor
            loss = criterion(output, mask)

        d = dice_score(output.float(), mask)
        i = iou_score(output.float(), mask)
        losses.append(loss.item())
        dices.append(d)
        ious.append(i)

    return {
        "loss": sum(losses) / len(losses),
        "dice": sum(dices) / len(dices),
        "iou": sum(ious) / len(ious),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["unet", "hybrid"], required=True)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    if device.type == "cuda":
        log.info("GPU: %s", torch.cuda.get_device_name(0))
        log.info("VRAM: %.1f GB", torch.cuda.get_device_properties(0).total_memory / 1e9)

    bs = args.batch_size or get_batch_size(args.model)
    log.info("Batch size: %d", bs)

    train_ds = BraTSDataset(split="train", transform=get_train_transforms())
    val_ds = BraTSDataset(split="val", transform=get_val_transforms())
    log.info("Train: %d slices, Val: %d slices", len(train_ds), len(val_ds))

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY,
    )

    model = build_model(args.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model: %s (%.2fM params)", args.model, n_params / 1e6)
    if args.model == "hybrid" and getattr(config, "USE_DEEP_SUPERVISION", False):
        log.info("Deep supervision: ENABLED (weights=%s)", config.DEEP_SUPERVISION_WEIGHTS)
    if getattr(config, "BOUNDARY_WEIGHT", 0.0) > 0:
        log.info("Boundary loss: ENABLED (weight=%.2f)", config.BOUNDARY_WEIGHT)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    criterion = CombinedLoss()
    scaler = GradScaler(enabled=config.USE_AMP)

    start_epoch = 0
    best_dice = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_dice = ckpt.get("best_dice", 0.0)
        log.info("Resumed from %s (epoch %d, best Dice %.4f)", args.resume, start_epoch, best_dice)

    run_name = f"{args.model}_v2_{int(time.time())}"
    writer = SummaryWriter(config.LOG_DIR / run_name)

    patience_left = config.PATIENCE
    for epoch in range(start_epoch, args.epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, epoch, writer)
        val_metrics = validate(model, val_loader, criterion, device, epoch)
        scheduler.step()

        log.info(
            "Epoch %d | train loss %.4f dice %.4f | val loss %.4f dice %.4f iou %.4f | lr %.2e",
            epoch,
            train_metrics["loss"], train_metrics["dice"],
            val_metrics["loss"], val_metrics["dice"], val_metrics["iou"],
            optimizer.param_groups[0]["lr"],
        )

        writer.add_scalar("train/loss_epoch", train_metrics["loss"], epoch)
        writer.add_scalar("train/dice_epoch", train_metrics["dice"], epoch)
        writer.add_scalar("val/loss", val_metrics["loss"], epoch)
        writer.add_scalar("val/dice", val_metrics["dice"], epoch)
        writer.add_scalar("val/iou", val_metrics["iou"], epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            patience_left = config.PATIENCE
            ckpt_path = config.CHECKPOINT_DIR / f"{args.model}_best.pth"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "best_dice": best_dice,
                    "args": vars(args),
                },
                ckpt_path,
            )
            log.info("Saved best checkpoint (Dice %.4f) → %s", best_dice, ckpt_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                log.info("Early stopping at epoch %d (no improvement for %d epochs)", epoch, config.PATIENCE)
                break

        torch.save(
            {"model": model.state_dict(), "epoch": epoch, "best_dice": best_dice},
            config.CHECKPOINT_DIR / f"{args.model}_last.pth",
        )

    writer.close()
    log.info("Training done. Best val Dice: %.4f", best_dice)


if __name__ == "__main__":
    main()