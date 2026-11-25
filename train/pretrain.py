"""Pre-training loop for SeasonalViT.

MAE-style pre-training with configurable masking strategy:
- "efficient": Levels 2+3 only, ~2x faster (temporal masking before encoding)
- "full": All 3 levels (patch + temporal masking after encoding)

Uses mixed precision, gradient clipping, and checkpointing.
Supports multi-GPU training with DistributedDataParallel.
"""

import logging
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model.config import PretrainingConfig, SeasonalViTConfig
from model.pretrain_model import PretrainingModel
from train.distributed import is_main_process
from train.losses import pretrain_loss, pretrain_loss_efficient
from train.masking import generate_mask, generate_efficient_temporal_mask
from train.metrics import compute_metrics

logger = logging.getLogger(__name__)


def _format_time(seconds: float) -> str:
    """Format seconds as human-readable string (e.g., '12m 34s' or '1h 23m')."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def _sample_year_subset(
    images: torch.Tensor,
    season_ids: torch.Tensor,
    year_indices: torch.Tensor,
    nodata_mask: Optional[torch.Tensor],
    num_years: int,
    num_seasons: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """Randomly sample K complete years from the full temporal sequence.

    Selects the same year subset for all samples in the batch. Each sampled
    year keeps all 4 seasons intact, preserving the dual-scale transformer structure.

    Args:
        images: Input images (B, T, H, W) where T = total_years * num_seasons.
        season_ids: Season indices (B, T).
        year_indices: Year indices (B, T).
        nodata_mask: Optional nodata mask (B, T, H, W) or None.
        num_years: Number of years to sample (K).
        num_seasons: Seasons per year (4).

    Returns:
        Tuple of (images, season_ids, year_indices, nodata_mask, seq_positions)
        where all temporal tensors are subsetted to K*4 timesteps and
        seq_positions contains the original timestep indices for positional encoding.
    """
    # Infer total years from input tensor
    T = images.shape[1]
    total_years = T // num_seasons

    # Sample K years from 0..total_years-1 (same for all samples in batch)
    selected_years = sorted(random.sample(range(total_years), num_years))

    # Convert year indices to timestep indices (each year has 4 seasons)
    timestep_indices = []
    for y in selected_years:
        base = y * num_seasons
        timestep_indices.extend(range(base, base + num_seasons))
    idx = torch.tensor(timestep_indices, device=images.device)

    # seq_positions: original timestep indices for sinusoidal PE
    B = images.shape[0]
    seq_positions = idx.unsqueeze(0).expand(B, -1)  # (B, K*4)

    # Subset all temporal tensors along dim=1
    images = images[:, idx]
    season_ids = season_ids[:, idx]
    year_indices = year_indices[:, idx]
    if nodata_mask is not None:
        nodata_mask = nodata_mask[:, idx]

    return images, season_ids, year_indices, nodata_mask, seq_positions


class TrainingDivergedError(RuntimeError):
    """Raised when training produces NaN/Inf loss."""

    pass


class PretrainingTrainer:
    """Trainer for MAE-style pre-training.

    Handles:
    - Mixed precision training with GradScaler
    - Gradient clipping
    - Learning rate scheduling with warmup
    - Checkpointing
    - TensorBoard logging

    Attributes:
        model: PretrainingModel instance.
        config: PretrainingConfig with training hyperparameters.
        device: Training device (cuda/cpu).
        optimizer: AdamW optimizer.
        scheduler: Learning rate scheduler.
        scaler: GradScaler for mixed precision.
        writer: TensorBoard SummaryWriter.
    """

    def __init__(
        self,
        model: PretrainingModel,
        config: PretrainingConfig,
        device: torch.device,
        output_dir: Path,
        resume_from: Optional[Path] = None,
        steps_per_epoch: Optional[int] = None,
        distributed: bool = False,
        rank: int = 0,
    ) -> None:
        """Initialize trainer.

        Args:
            model: PretrainingModel to train.
            config: Training configuration.
            device: Device to train on.
            output_dir: Directory for checkpoints and logs.
            resume_from: Optional checkpoint to resume from.
            steps_per_epoch: Number of training steps per epoch. If None,
                will be computed from dataloader length in fit().
            distributed: Whether to use DistributedDataParallel.
            rank: Process rank for distributed training (0 = main process).
        """
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.steps_per_epoch = steps_per_epoch
        self.distributed = distributed
        self.rank = rank

        # Move model to device and wrap with DDP if distributed
        model = model.to(device)
        if distributed:
            # broadcast_buffers=False: model has no BN and all buffers are
            # constants (positional encodings, delta_scale), so the per-step
            # buffer broadcast is pure overhead. Same fix as finetune; mirrors
            # the gloo timeout extension in train.distributed.
            self.model = DDP(
                model,
                device_ids=[device.index] if device.index is not None else None,
                gradient_as_bucket_view=False,
                broadcast_buffers=False,
            )
        else:
            self.model = model

        # Get underlying model for parameter access
        self._base_model = model if not distributed else self.model.module

        # Training state (must be set before scheduler creation)
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float("inf")

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self._base_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Learning rate scheduler with warmup
        self.scheduler = self._create_scheduler()

        # Mixed precision: use bf16 (same exponent range as fp32, no overflow risk)
        self.amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        if hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=False)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=False)

        # Logging - only on main process
        if is_main_process():
            self.writer = SummaryWriter(log_dir=self.output_dir / "logs")
        else:
            self.writer = None

        if resume_from is not None:
            self._load_checkpoint(resume_from)

    def _create_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        """Create learning rate scheduler with linear warmup and cosine decay to min_lr."""
        steps_per_epoch = self.steps_per_epoch if self.steps_per_epoch else 1000
        remaining_epochs = self.config.max_epochs - self.epoch
        total_steps = remaining_epochs * steps_per_epoch
        warmup_steps = min(self.config.warmup_epochs, remaining_epochs) * steps_per_epoch
        # min_lr as a fraction of learning_rate
        min_lr_ratio = self.config.min_lr / self.config.learning_rate

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                # Linear warmup from 0 to learning_rate
                return step / warmup_steps
            # Cosine decay from learning_rate to min_lr
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            # Scale from [0, 1] to [min_lr_ratio, 1]
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        use_progress_bar: bool = True,
    ) -> Dict[str, float]:
        """Train for one epoch with gradient accumulation support.

        Args:
            dataloader: Training data loader.
            epoch: Current epoch number.
            use_progress_bar: Whether to show tqdm progress bar.

        Returns:
            Dictionary with epoch metrics.
        """
        epoch_start_time = time.time()
        self.model.train()

        # Accumulate loss on GPU to avoid sync every step
        total_loss_tensor = torch.tensor(0.0, device=self.device)
        num_batches = 0
        total_samples = 0
        accum_steps = self.config.gradient_accumulation_steps

        self.optimizer.zero_grad()

        # Set up progress bar - only show on main process
        show_pbar = use_progress_bar and is_main_process()
        if show_pbar:
            bar_format = "Epoch {desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed} elapsed, {remaining} left, {rate_fmt}]{postfix}\n"
            pbar = tqdm(dataloader, desc=f"{epoch + 1}", bar_format=bar_format, leave=True)
        else:
            pbar = dataloader

        for step, batch in enumerate(pbar):
            # Start timing at beginning of accumulation cycle
            if step % accum_steps == 0:
                accum_start_time = time.time()
                accum_samples = 0

            loss = self._train_step_accumulate(batch, accum_steps)
            actual_loss = loss * accum_steps  # Undo the division for logging
            # Accumulate on GPU tensor - no sync
            total_loss_tensor += actual_loss
            num_batches += 1
            batch_samples = batch["images"].shape[0]
            total_samples += batch_samples
            accum_samples += batch_samples

            # Optimizer step at accumulation boundary
            if (step + 1) % accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
                self.global_step += 1

                # Calculate throughput for full accumulation cycle
                accum_time = time.time() - accum_start_time
                samples_per_sec = accum_samples / accum_time if accum_time > 0 else 0

                # Update progress bar (only on main process)
                if show_pbar and hasattr(pbar, "set_postfix"):
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    pbar.set_postfix(
                        loss=f"{actual_loss.item():.4f}",
                        lr=f"{current_lr:.2e}",
                        throughput=f"{samples_per_sec:.1f}",
                    )

                # Log step metrics to TensorBoard (only on main process)
                if self.global_step % 100 == 0 and is_main_process():
                    self._log_step_metrics(
                        actual_loss.item(), grad_norm, samples_per_sec, accum_time
                    )

        # Handle remaining gradients at epoch end
        if num_batches % accum_steps != 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.scheduler.step()
            self.global_step += 1

        # Only sync to CPU at epoch end for final average
        total_loss = total_loss_tensor.item()
        avg_loss = total_loss / max(num_batches, 1)

        # Calculate epoch timing
        epoch_time = time.time() - epoch_start_time
        avg_throughput = total_samples / epoch_time if epoch_time > 0 else 0

        # Log epoch metrics (only on main process)
        if self.writer is not None:
            self.writer.add_scalar("train/loss_epoch", avg_loss, epoch)
            self.writer.add_scalar("train/epoch_time_s", epoch_time, epoch)
            self.writer.add_scalar("train/throughput_epoch", avg_throughput, epoch)

        return {
            "loss": avg_loss,
            "num_batches": num_batches,
            "num_samples": total_samples,
            "epoch_time": epoch_time,
            "throughput": avg_throughput,
        }

    def _log_step_metrics(
        self,
        loss: float,
        grad_norm: torch.Tensor,
        samples_per_sec: float,
        step_time: float,
    ) -> None:
        """Log training metrics to TensorBoard.

        Args:
            loss: Current loss value.
            grad_norm: Gradient norm after clipping.
            samples_per_sec: Training throughput.
            step_time: Time for this step in seconds.
        """
        if self.writer is None:
            return

        # Core training metrics
        self.writer.add_scalar("train/loss_step", loss, self.global_step)
        self.writer.add_scalar(
            "train/lr", self.optimizer.param_groups[0]["lr"], self.global_step
        )
        self.writer.add_scalar("train/grad_norm", grad_norm.item(), self.global_step)

        # Performance metrics
        self.writer.add_scalar("perf/samples_per_sec", samples_per_sec, self.global_step)
        self.writer.add_scalar("perf/step_time_ms", step_time * 1000, self.global_step)

        # GPU memory metrics (if on CUDA)
        if self.device.type == "cuda":
            mem_allocated = torch.cuda.memory_allocated(self.device) / 1e9
            mem_reserved = torch.cuda.memory_reserved(self.device) / 1e9
            self.writer.add_scalar(
                "system/gpu_memory_allocated_gb", mem_allocated, self.global_step
            )
            self.writer.add_scalar(
                "system/gpu_memory_reserved_gb", mem_reserved, self.global_step
            )

    def _move_batch_to_device(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Move batch tensors to device with non_blocking transfer.

        Args:
            batch: Batch dictionary from dataloader.

        Returns:
            Tuple of (images, season_ids, year_indices, nodata_mask).
        """
        images = batch["images"].to(self.device, non_blocking=True)
        season_ids = batch["season_ids"].to(self.device, non_blocking=True)
        year_indices = batch["year_indices"].to(self.device, non_blocking=True)
        nodata_mask = batch.get("nodata_mask")
        if nodata_mask is not None:
            nodata_mask = nodata_mask.to(self.device, non_blocking=True)
        return images, season_ids, year_indices, nodata_mask

    def _forward_and_loss(
        self,
        images: torch.Tensor,
        season_ids: torch.Tensor,
        year_indices: torch.Tensor,
        nodata_mask: Optional[torch.Tensor],
        apply_temporal_subset: bool = True,
    ) -> torch.Tensor:
        """Run forward pass and compute loss with autocast.

        Uses either efficient or full masking strategy based on config.
        Optionally applies temporal year subsampling before the forward pass.

        Args:
            images: Input images tensor (B, T, H, W).
            season_ids: Season identifiers (B, T).
            year_indices: Year indices (B, T).
            nodata_mask: Optional nodata mask.
            apply_temporal_subset: Whether to apply year subsampling. Set to
                False during validation to use the full temporal sequence.

        Returns:
            Raw loss tensor (not scaled for accumulation).
        """
        # Temporal year subsampling (training only)
        seq_positions = None
        subset_years = self.config.temporal_subset_years
        if apply_temporal_subset and subset_years > 0:
            images, season_ids, year_indices, nodata_mask, seq_positions = (
                _sample_year_subset(
                    images, season_ids, year_indices, nodata_mask,
                    num_years=subset_years,
                )
            )

        B, T, H, W = images.shape
        N = self._base_model.config.num_spatial_patches

        with torch.amp.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.device.type == "cuda"):
            if self.config.masking_strategy == "efficient":
                efficient_mask = generate_efficient_temporal_mask(
                    B, T, config=self.config, device=self.device
                )
                output = self._base_model.forward_efficient(
                    images, season_ids, year_indices, efficient_mask,
                    seq_positions=seq_positions,
                )
                loss = pretrain_loss_efficient(
                    output["reconstruction"],
                    output["target"],
                    output["temporal_mask"],
                    nodata_mask=nodata_mask,
                    normalize_target=self.config.normalize_target,
                )
            else:
                mask_output = generate_mask(B, T, N, config=self.config, device=self.device)
                mask = mask_output.combined_mask
                output = self.model(
                    images, season_ids, year_indices, mask=mask,
                    seq_positions=seq_positions,
                )
                loss = pretrain_loss(
                    output["reconstruction"],
                    output["target"],
                    mask,
                    nodata_mask=nodata_mask,
                    normalize_target=self.config.normalize_target,
                )
        return loss

    def _train_step_accumulate(
        self, batch: Dict[str, torch.Tensor], accum_steps: int
    ) -> torch.Tensor:
        """Execute forward/backward pass for gradient accumulation.

        Args:
            batch: Batch dictionary from dataloader.
            accum_steps: Number of accumulation steps (for loss scaling).

        Returns:
            Detached loss tensor (call .item() only when needed for logging).
        """
        images, season_ids, year_indices, nodata_mask = self._move_batch_to_device(batch)
        loss = self._forward_and_loss(images, season_ids, year_indices, nodata_mask)
        loss = loss / accum_steps
        self.scaler.scale(loss).backward()
        return loss.detach()

    @torch.no_grad()
    def validate(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        """Run validation.

        Uses same masking strategy as training for fair comparison.

        Args:
            dataloader: Validation data loader.

        Returns:
            Dictionary with validation metrics.
        """
        val_start_time = time.time()
        self.model.eval()
        total_loss = torch.tensor(0.0, device=self.device)
        num_batches = 0
        total_samples = 0

        if is_main_process():
            bar_format = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]\n"
            iterator = tqdm(dataloader, desc="Validation", bar_format=bar_format, leave=True)
        else:
            iterator = dataloader

        for batch in iterator:
            images, season_ids, year_indices, nodata_mask = self._move_batch_to_device(batch)
            loss = self._forward_and_loss(
                images, season_ids, year_indices, nodata_mask,
                apply_temporal_subset=False,
            )
            total_loss += loss.detach()
            num_batches += 1
            total_samples += images.shape[0]

        avg_loss = total_loss.item() / max(num_batches, 1)
        val_time = time.time() - val_start_time

        return {
            "val_loss": avg_loss,
            "num_samples": total_samples,
            "val_time": val_time,
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:
        """Run full training loop.

        Args:
            train_loader: Training data loader.
            val_loader: Optional validation data loader.

        Returns:
            Training history dictionary.
        """
        if self.steps_per_epoch is None:
            self.steps_per_epoch = len(train_loader)
            self.scheduler = self._create_scheduler()
            if is_main_process():
                logger.info(f"Using {self.steps_per_epoch} steps per epoch for LR schedule")

        history = {"train_loss": [], "val_loss": []}

        for epoch in range(self.epoch, self.config.max_epochs):
            self.epoch = epoch
            epoch_start_time = time.time()

            if is_main_process():
                logger.info(f"Epoch {epoch + 1}/{self.config.max_epochs}")

            if self.distributed and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)
            train_metrics = self.train_epoch(train_loader, epoch)

            history["train_loss"].append(train_metrics["loss"])
            if is_main_process():
                current_lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"  Train: loss={train_metrics['loss']:.4f} | "
                    f"lr={current_lr:.2e} | "
                    f"throughput={train_metrics['throughput']:.1f} samples/s | "
                    f"time={_format_time(train_metrics['epoch_time'])}"
                )

            # Validation and checkpoint tracking
            val_loss = None
            saved_best = False
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                val_loss = val_metrics["val_loss"]
                history["val_loss"].append(val_loss)
                if is_main_process():
                    logger.info(
                        f"  Val: loss={val_loss:.4f} | "
                        f"time={_format_time(val_metrics['val_time'])}"
                    )
                if self.writer is not None:
                    self.writer.add_scalar("val/loss", val_loss, epoch)

                if val_loss < self.best_loss:
                    prev_best = self.best_loss
                    self.best_loss = val_loss
                    self._save_checkpoint(self.output_dir / "best.pt")
                    saved_best = True

            if (epoch + 1) % self.config.checkpoint_every_n_epochs == 0:
                self._save_checkpoint(self.output_dir / f"epoch_{epoch + 1}.pt")

            # Epoch summary
            if is_main_process():
                total_epoch_time = time.time() - epoch_start_time
                summary_parts = [
                    f"Epoch {epoch + 1} complete",
                    f"train_loss={train_metrics['loss']:.4f}",
                ]
                if val_loss is not None:
                    summary_parts.append(f"val_loss={val_loss:.4f}")
                    if saved_best:
                        improvement = prev_best - val_loss
                        summary_parts.append(f"NEW BEST (improved {improvement:.4f})")
                    else:
                        gap = val_loss - self.best_loss
                        summary_parts.append(f"best={self.best_loss:.4f} (+{gap:.4f})")
                summary_parts.append(f"total_time={_format_time(total_epoch_time)}")
                logger.info("  " + " | ".join(summary_parts))

        self._save_checkpoint(self.output_dir / "final.pt")

        if self.writer is not None:
            self.writer.close()
        return history

    def _save_checkpoint(self, path: Path) -> None:
        """Save training checkpoint including RNG states for reproducibility.

        Only saves on main process in distributed mode.

        Args:
            path: Path to save checkpoint.
        """
        # Only save on main process
        if not is_main_process():
            return

        # Use base model state dict (unwrap DDP if necessary)
        model_state = self._base_model.state_dict()

        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "config": self.config,
            # RNG states for reproducibility
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def _load_checkpoint(self, path: Path) -> None:
        """Load training checkpoint including RNG states for reproducibility.

        Args:
            path: Path to checkpoint.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Load into base model (unwrapped from DDP)
        self._base_model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.epoch = checkpoint["epoch"] + 1
        self.global_step = checkpoint["global_step"]
        self.best_loss = checkpoint["best_loss"]

        # Restore RNG states for reproducibility
        if "torch_rng_state" in checkpoint:
            # RNG state must be CPU ByteTensor (map_location may have moved it to GPU)
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if checkpoint.get("cuda_rng_state") and torch.cuda.is_available():
            # CUDA RNG states must also be CPU ByteTensors
            cuda_states = [s.cpu() for s in checkpoint["cuda_rng_state"]]
            torch.cuda.set_rng_state_all(cuda_states)
        if "numpy_rng_state" in checkpoint:
            np.random.set_state(checkpoint["numpy_rng_state"])
        if "python_rng_state" in checkpoint:
            random.setstate(checkpoint["python_rng_state"])

        if is_main_process():
            logger.info(f"Resumed from {path} at epoch {self.epoch}")
