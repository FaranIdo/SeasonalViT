#!/usr/bin/env python3
"""Fine-tuning script for Prithvi-EO-2.0 baseline.

Fine-tunes the Prithvi foundation model for NDVI forecasting.
Uses differential learning rates (lower for backbone, higher for head)
and optional backbone freezing for efficient adaptation.

Usage:
    # Fine-tune with frozen backbone (recommended for small datasets)
    python baselines/prithvi/train.py \
        --train-dir /path/to/train \
        --val-dir /path/to/val \
        --output-dir outputs/prithvi \
        --freeze-backbone --unfreeze-last-n 4

    # Full fine-tuning
    python baselines/prithvi/train.py \
        --train-dir /path/to/train \
        --val-dir /path/to/val \
        --output-dir outputs/prithvi

    # Multi-GPU
    python baselines/prithvi/train.py \
        --train-dir /path/to/train \
        --val-dir /path/to/val \
        --output-dir outputs/prithvi \
        --multi-gpu --spawn --num-gpus 2
"""

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from baselines.prithvi.model import PrithviForecaster
from datasets.prefetch_dataset import ShardPrefetchLoader
from datasets.sharded_dataset import InMemoryNDVIDataset, create_dataloader
from train.distributed import (
    cleanup_distributed,
    is_main_process,
    setup_distributed,
    setup_distributed_spawn,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune Prithvi-EO-2.0 baseline")
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/prithvi"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--predict-years", type=int, nargs="+", default=[2022])
    parser.add_argument("--model-name", type=str, default="prithvi_eo_v2_300")
    parser.add_argument("--num-input-frames", type=int, default=4)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--unfreeze-last-n", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--train-shard-range", type=str, default=None)
    parser.add_argument("--val-shard-range", type=str, default=None)
    parser.add_argument("--multi-gpu", action="store_true")
    parser.add_argument("--spawn", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=2)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--val-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    return parser.parse_args()


def get_target_indices(predict_years: List[int]) -> List[int]:
    """Convert years to timestep indices."""
    indices = []
    for year in predict_years:
        start = (year - 1990) * 4
        indices.extend(range(start, start + 4))
    return indices


def parse_shard_range(s: Optional[str]) -> Optional[Tuple[int, int]]:
    """Parse shard range string."""
    if s is None:
        return None
    start, end = map(int, s.split(":"))
    return (start, end)


def create_optimizer(
    model: PrithviForecaster,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """Create optimizer with differential learning rates.

    Backbone parameters get a lower learning rate to preserve pre-trained
    features, while the input adapter and prediction head get higher LR
    for faster convergence.
    """
    backbone_params = list(model.backbone.parameters())
    head_params = (
        list(model.input_adapter.parameters())
        + list(model.prediction_head.parameters())
    )

    # Filter to trainable only
    backbone_params = [p for p in backbone_params if p.requires_grad]
    head_params = [p for p in head_params if p.requires_grad]

    param_groups = [
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params, "lr": head_lr},
    ]

    logger.info(
        f"Optimizer: backbone={len(backbone_params)} params (lr={backbone_lr:.1e}), "
        f"head={len(head_params)} params (lr={head_lr:.1e})"
    )

    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def sample_target_indices(
    batch_size: int,
    target_indices: List[int],
    device: torch.device,
) -> torch.Tensor:
    """Sample random target indices for the batch."""
    idx_tensor = torch.tensor(target_indices, device=device)
    samples = torch.randint(0, len(target_indices), (batch_size,), device=device)
    return idx_tensor[samples]


def prepare_input_and_target(
    batch: Dict[str, torch.Tensor],
    target_idx: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Extract input sequence and target from batch."""
    images = batch["images"].to(device, non_blocking=True)
    nodata_mask = batch.get("nodata_mask")
    if nodata_mask is not None:
        nodata_mask = nodata_mask.to(device, non_blocking=True)

    B = images.shape[0]
    batch_idx = torch.arange(B, device=device)

    min_target = target_idx.min().item()
    input_seq = images[:, :min_target].unsqueeze(2)

    target = images[batch_idx, target_idx].unsqueeze(1)
    target_nodata = None
    if nodata_mask is not None:
        target_nodata = nodata_mask[batch_idx, target_idx].unsqueeze(1)

    return input_seq, target, target_nodata


def compute_masked_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    nodata_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Compute L1 loss excluding nodata pixels."""
    if nodata_mask is not None:
        valid = ~nodata_mask
        if valid.any():
            return (pred[valid] - target[valid]).abs().mean()
    return (pred - target).abs().mean()


def train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    target_indices: List[int],
    device: torch.device,
    epoch: int,
    grad_accum_steps: int = 1,
) -> Dict[str, float]:
    """Train for one epoch with gradient accumulation."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    epoch_start = time.time()
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}") if is_main_process() else loader

    for step, batch in enumerate(pbar):
        B = batch["images"].shape[0]
        target_idx = sample_target_indices(B, target_indices, device)
        input_seq, target, target_nodata = prepare_input_and_target(
            batch, target_idx, device,
        )

        with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
            pred = model(input_seq)
            loss = compute_masked_l1(pred, target, target_nodata) / grad_accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        total_loss += loss.item() * grad_accum_steps
        num_batches += 1

        if is_main_process() and hasattr(pbar, "set_postfix"):
            pbar.set_postfix(
                loss=f"{loss.item() * grad_accum_steps:.4f}",
                lr=f"{optimizer.param_groups[1]['lr']:.2e}",
            )

    return {
        "loss": total_loss / max(num_batches, 1),
        "epoch_time": time.time() - epoch_start,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    target_indices: List[int],
    device: torch.device,
) -> Dict[str, float]:
    """Run validation."""
    model.eval()
    total_mae = 0.0
    total_pixels = 0

    for batch in loader:
        B = batch["images"].shape[0]
        target_idx = sample_target_indices(B, target_indices, device)
        input_seq, target, target_nodata = prepare_input_and_target(
            batch, target_idx, device,
        )

        with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
            pred = model(input_seq)

        valid = ~target_nodata if target_nodata is not None else torch.ones_like(target, dtype=torch.bool)
        if valid.any():
            total_mae += (pred[valid] - target[valid]).abs().sum().item()
            total_pixels += valid.sum().item()

    mae = total_mae / max(total_pixels, 1)
    return {"val_mae": mae}


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_mae: float,
    path: Path,
    distributed: bool = False,
) -> None:
    """Save training checkpoint."""
    state = model.module.state_dict() if distributed else model.state_dict()
    torch.save({
        "model_state_dict": state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "best_mae": best_mae,
    }, path)


def train_worker(
    rank: int,
    world_size: int,
    args: argparse.Namespace,
    distributed_mode: str,
    store_path: Optional[str] = None,
) -> None:
    """Training worker."""
    distributed = distributed_mode != "none"
    is_main = rank == 0

    if distributed_mode == "spawn":
        setup_distributed_spawn(rank, world_size, store_path)
    elif distributed_mode == "torchrun":
        setup_distributed()

    device = _setup_device(rank, distributed_mode, distributed)
    torch.backends.cudnn.benchmark = True

    if is_main:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    target_indices = get_target_indices(args.predict_years)
    if is_main:
        logger.info(f"Target indices: {target_indices}")

    # Data loaders
    train_loader = _create_train_loader(args, rank, world_size)
    val_loader = _create_val_loader(args)

    # Create model
    model = PrithviForecaster(
        model_name=args.model_name,
        pretrained=True,
        num_input_frames=args.num_input_frames,
        freeze_backbone=args.freeze_backbone,
        unfreeze_last_n=args.unfreeze_last_n,
    )
    if is_main:
        total = model.get_num_parameters()
        trainable = model.get_num_parameters(trainable_only=True)
        logger.info(f"Prithvi params: {total:,} total, {trainable:,} trainable")

    model = model.to(device)
    if distributed:
        model = DDP(
            model,
            device_ids=[device.index] if device.index is not None else None,
            find_unused_parameters=True,
        )

    # Optimizer with differential LR
    base_model = model.module if distributed else model
    optimizer = create_optimizer(base_model, args.backbone_lr, args.head_lr, args.weight_decay)

    steps_per_epoch = len(train_loader) // args.gradient_accumulation
    total_steps = args.epochs * steps_per_epoch

    def lr_lambda(step):
        warmup_steps = steps_per_epoch * 2
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.5 * (1 + math.cos(math.pi * progress)), 1e-7 / args.head_lr)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    start_epoch = 0
    best_mae = float("inf")
    if args.resume and args.resume.exists():
        start_epoch, best_mae = _load_checkpoint(
            args.resume, model, optimizer, scheduler, device, distributed,
        )

    for epoch in range(start_epoch, args.epochs):
        if distributed and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            target_indices, device, epoch, args.gradient_accumulation,
        )

        if is_main:
            logger.info(
                f"Epoch {epoch+1}: loss={train_metrics['loss']:.4f}, "
                f"time={train_metrics['epoch_time']:.1f}s"
            )

            val_metrics = validate(model, val_loader, target_indices, device)
            logger.info(f"  Val MAE: {val_metrics['val_mae']:.4f}")

            if val_metrics["val_mae"] < best_mae:
                best_mae = val_metrics["val_mae"]
                save_checkpoint(
                    model, optimizer, scheduler, epoch, best_mae,
                    args.output_dir / "best.pt", distributed,
                )
                logger.info(f"  New best MAE: {best_mae:.4f}")

            if (epoch + 1) % 5 == 0:
                save_checkpoint(
                    model, optimizer, scheduler, epoch, best_mae,
                    args.output_dir / f"epoch_{epoch+1}.pt", distributed,
                )

    if is_main:
        save_checkpoint(
            model, optimizer, scheduler, args.epochs - 1, best_mae,
            args.output_dir / "final.pt", distributed,
        )
        logger.info(f"Training complete. Best val MAE: {best_mae:.4f}")

    if distributed:
        cleanup_distributed()


def _setup_device(rank: int, distributed_mode: str, distributed: bool) -> torch.device:
    """Set up compute device."""
    if distributed_mode == "spawn":
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    elif torch.cuda.is_available():
        if distributed:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def _create_train_loader(
    args: argparse.Namespace,
    rank: int,
    world_size: int,
) -> ShardPrefetchLoader:
    """Create training data loader."""
    shard_range = parse_shard_range(args.train_shard_range)
    return ShardPrefetchLoader(
        shard_dir=args.train_dir,
        batch_size=args.batch_size,
        shuffle_shards=True,
        shuffle_within_shard=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        rank=rank,
        world_size=world_size,
        shard_range=shard_range,
    )


def _create_val_loader(args: argparse.Namespace):
    """Create validation data loader."""
    val_shard_range = parse_shard_range(args.val_shard_range)
    val_dataset = InMemoryNDVIDataset(args.val_dir, shard_range=val_shard_range)
    return create_dataloader(
        val_dataset, batch_size=args.val_batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )


def _load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    distributed: bool,
) -> Tuple[int, float]:
    """Load checkpoint and return (start_epoch, best_mae)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    base_model = model.module if distributed else model
    base_model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    start_epoch = ckpt["epoch"] + 1
    best_mae = ckpt.get("best_mae", float("inf"))
    logger.info(f"Resumed from {path} at epoch {start_epoch}")
    return start_epoch, best_mae


def main():
    """Main entry point."""
    args = parse_args()

    if args.multi_gpu and args.spawn:
        world_size = min(args.num_gpus, torch.cuda.device_count())
        store_dir = args.output_dir
        store_dir.mkdir(parents=True, exist_ok=True)
        store_path = str(store_dir / "ddp_store")
        if Path(store_path).exists():
            Path(store_path).unlink()

        mp.spawn(
            _spawn_wrapper,
            args=(world_size, args, store_path),
            nprocs=world_size,
            join=True,
        )
    else:
        distributed_mode = "torchrun" if args.multi_gpu else "none"
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        train_worker(rank, world_size, args, distributed_mode)


def _spawn_wrapper(rank, world_size, args, store_path):
    """Wrapper for mp.spawn."""
    train_worker(rank, world_size, args, "spawn", store_path)


if __name__ == "__main__":
    main()
