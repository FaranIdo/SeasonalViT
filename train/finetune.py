"""Fine-tuning loop for SeasonalViT forecasting.

Trains the full forecasting model on NDVI prediction task.
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
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model.config import FinetuningConfig, SeasonalViTConfig
from model.seasonal_vit import SeasonalViT
from train.distributed import is_main_process
from train.losses import ForecastingLoss
from train.metrics import compute_metrics, MetricsTracker, PerSampleMetricsTracker

logger = logging.getLogger(__name__)


def _format_time(seconds: float) -> str:
    """Format seconds as human-readable string."""
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


class FinetuningTrainer:
    """Trainer for SeasonalViT fine-tuning on forecasting task.

    Handles:
    - Loading pre-trained encoder weights
    - Optional encoder freezing
    - Mixed precision training
    - Gradient clipping
    - Learning rate scheduling
    - Checkpointing
    - TensorBoard logging

    Attributes:
        model: SeasonalViT instance.
        config: FinetuningConfig with training hyperparameters.
        device: Training device.
        optimizer: AdamW optimizer.
        scheduler: Learning rate scheduler.
        scaler: GradScaler for mixed precision.
        loss_fn: ForecastingLoss instance.
        writer: TensorBoard SummaryWriter.
    """

    def __init__(
        self,
        model: SeasonalViT,
        config: FinetuningConfig,
        device: torch.device,
        output_dir: Path,
        resume_from: Optional[Path] = None,
        steps_per_epoch: Optional[int] = None,
        distributed: bool = False,
        rank: int = 0,
    ) -> None:
        """Initialize trainer.

        Args:
            model: SeasonalViT to fine-tune.
            config: Fine-tuning configuration.
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

        # Optionally freeze encoder (before moving to device or wrapping with DDP)
        if config.freeze_encoder:
            model.freeze_encoder()
            if is_main_process():
                logger.info("Encoder frozen for fine-tuning")
        elif config.freeze_spatial:
            model.freeze_spatial_encoder()
            if is_main_process():
                logger.info("Spatial encoder frozen for fine-tuning")

        # Move model to device and wrap with DDP if distributed
        model = model.to(device)
        if distributed:
            # broadcast_buffers=False: model has no BN (see _assert_no_batchnorm)
            # and all buffers are constants (positional encodings, delta_scale),
            # so the per-step buffer broadcast is pure overhead and was the
            # collective that timed out on epoch 15 of a multi-hour fine-tuning run.
            self.model = DDP(
                model,
                device_ids=[device.index] if device.index is not None else None,
                gradient_as_bucket_view=True,
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
        self.best_mae = float("inf")

        # Optimizer with differential learning rates
        # Pre-trained encoder components use base LR; randomly-initialized
        # decoder/location_encoder use higher LR for faster convergence.
        encoder_params, decoder_params = self._split_param_groups()
        decoder_lr = config.learning_rate * config.decoder_lr_multiplier
        self.optimizer = torch.optim.AdamW(
            [
                {"params": encoder_params, "lr": config.learning_rate},
                {"params": decoder_params, "lr": decoder_lr},
            ],
            weight_decay=config.weight_decay,
        )
        if is_main_process():
            logger.info(
                f"Optimizer: encoder params={len(encoder_params)} (lr={config.learning_rate:.1e}), "
                f"decoder params={len(decoder_params)} (lr={decoder_lr:.1e})"
            )

        # Learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Mixed precision: use bf16 (same exponent range as fp32, no overflow risk).
        # fp16 caused NaN collapse via GradScaler cascade failure at epoch 11.
        self.amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        # GradScaler is disabled for bf16 (unnecessary — bf16 never overflows)
        if hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=False)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=False)

        # Anchor mode drives delta supervision semantics. When anchor_mode='none'
        # there is no true delta, so delta_supervision and delta_reg are disabled.
        self._anchor_mode = getattr(
            self._base_model.config, "anchor_mode", "same_season"
        )
        anchor_free = self._anchor_mode == "none"
        delta_sup_weight = 0.0 if anchor_free else config.delta_supervision_weight
        delta_reg_weight = 0.0 if anchor_free else config.delta_reg_weight

        # Loss function
        self.loss_fn = ForecastingLoss(
            delta_reg_weight=delta_reg_weight,
            correlation_loss_weight=config.correlation_loss_weight,
            delta_supervision_weight=delta_sup_weight,
            delta_std=config.delta_std if config.delta_std > 0 else 1.0,
        )
        # If delta_std not provided, will be auto-estimated before training
        self._need_delta_std_estimation = (
            config.delta_std <= 0 and delta_sup_weight > 0
        )

        # Logging - only on main process
        if is_main_process():
            self.writer = SummaryWriter(log_dir=self.output_dir / "logs")
        else:
            self.writer = None

        if resume_from is not None:
            self._load_checkpoint(resume_from)

    def _assert_no_batchnorm(self) -> None:
        """Guard validation-bypasses-DDP assumption.

        Validation runs on `self._base_model` (bypassing the DDP wrapper) to
        avoid the ~30 min gloo timeout from `_sync_buffers` in
        `model.eval()` → first forward, which hangs because only rank 0
        validates. That's safe today because the model has no BatchNorm:
        eval mode doesn't update BN running stats, and all other buffers
        (positional encodings, delta_scale) are constants. If someone adds
        BN, rank 0's running stats would drift from other ranks. Fail loud
        here rather than silently regressing.
        """
        bn_types = (
            nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm,
        )
        for name, module in self._base_model.named_modules():
            if isinstance(module, bn_types):
                raise RuntimeError(
                    f"Model contains BatchNorm at '{name}' ({type(module).__name__}). "
                    "Validation runs on _base_model (not DDP) — BN running stats "
                    "would drift between ranks. Convert to SyncBatchNorm or move "
                    "back to a DDP-aware validation path."
                )

    def _split_param_groups(self) -> Tuple[list, list]:
        """Split trainable parameters into encoder and decoder groups.

        Encoder group: spatial_encoder, temporal_encoding, temporal_transformer.
        Decoder group: everything else (decoder, location_encoder, etc.).

        Returns:
            Tuple of (encoder_params, decoder_params).
        """
        encoder_prefixes = (
            "spatial_encoder", "temporal_encoding", "temporal_transformer",
        )
        encoder_params = []
        decoder_params = []
        for name, param in self._base_model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith(encoder_prefixes):
                encoder_params.append(param)
            else:
                decoder_params.append(param)
        return encoder_params, decoder_params

    def _create_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        """Create learning rate scheduler with warmup and cosine decay to min_lr."""
        steps_per_epoch = self.steps_per_epoch if self.steps_per_epoch else 1000

        # Capture fixed values at creation time to avoid closure over self.epoch
        total_steps = self.config.max_epochs * steps_per_epoch
        warmup_steps = self.config.warmup_epochs * steps_per_epoch
        min_scale = self.config.min_lr / self.config.learning_rate

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            # Cosine decay after warmup, floored at min_lr
            decay_steps = max(total_steps - warmup_steps, 1)
            progress = min((step - warmup_steps) / decay_steps, 1.0)
            return max(0.5 * (1 + math.cos(math.pi * progress)), min_scale)

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _move_batch_to_device(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
    ]:
        """Move batch tensors to device with non_blocking transfer.

        Args:
            batch: Batch dictionary from dataloader.

        Returns:
            Tuple of (images, season_ids, year_indices, lat, lon, nodata_mask).
        """
        images = batch["images"].to(self.device, non_blocking=True)
        if self.device.type != "cuda":
            # Shards store float16; autocast only runs on CUDA, so cast here
            # for CPU/MPS. CUDA keeps the original float16 inputs unchanged.
            images = images.float()
        season_ids = batch["season_ids"].to(self.device, non_blocking=True)
        year_indices = batch["year_indices"].to(self.device, non_blocking=True)
        lat = batch["lat"].to(self.device, non_blocking=True)
        lon = batch["lon"].to(self.device, non_blocking=True)
        nodata_mask = batch.get("nodata_mask")
        if nodata_mask is not None:
            nodata_mask = nodata_mask.to(self.device, non_blocking=True)
        return images, season_ids, year_indices, lat, lon, nodata_mask

    def _log_step_metrics(
        self,
        loss: float,
        recon_loss: float,
        grad_norm: torch.Tensor,
        samples_per_sec: float,
        step_time: float,
        delta_stats: Optional[Dict[str, float]] = None,
        delta_reg: float = 0.0,
        corr_loss: float = 0.0,
        delta_sup_loss: float = 0.0,
    ) -> None:
        """Log training metrics to TensorBoard.

        Args:
            loss: Current total loss value.
            recon_loss: Reconstruction loss component.
            grad_norm: Gradient norm after clipping.
            samples_per_sec: Training throughput.
            step_time: Time for this step in seconds.
            delta_stats: Optional delta statistics from decoder.
            delta_reg: Delta regularization loss for this step.
            corr_loss: Correlation loss for this step.
            delta_sup_loss: Delta supervision loss for this step.
        """
        if self.writer is None:
            return

        # Core training metrics
        self.writer.add_scalar("train/loss_step", loss, self.global_step)
        self.writer.add_scalar("train/recon_loss_step", recon_loss, self.global_step)
        self.writer.add_scalar("train/delta_reg_step", delta_reg, self.global_step)
        self.writer.add_scalar("train/corr_loss_step", corr_loss, self.global_step)
        self.writer.add_scalar("train/delta_sup_loss_step", delta_sup_loss, self.global_step)
        self.writer.add_scalar(
            "train/lr_encoder", self.optimizer.param_groups[0]["lr"], self.global_step
        )
        self.writer.add_scalar(
            "train/lr_decoder", self.optimizer.param_groups[1]["lr"], self.global_step
        )
        self.writer.add_scalar("train/grad_norm", grad_norm.item(), self.global_step)

        # Delta statistics (critical for detecting anchor collapse)
        if delta_stats is not None:
            self.writer.add_scalar(
                "train/delta_abs_mean", delta_stats["delta_abs_mean"], self.global_step
            )
            self.writer.add_scalar(
                "train/delta_std", delta_stats["delta_std"], self.global_step
            )
            self.writer.add_scalar(
                "train/delta_min", delta_stats["delta_min"], self.global_step
            )
            self.writer.add_scalar(
                "train/delta_max", delta_stats["delta_max"], self.global_step
            )
        delta_scale = self._base_model.decoder.delta_scale.item()
        self.writer.add_scalar("train/delta_scale", delta_scale, self.global_step)

        # Anchor vs prediction quality
        if delta_stats is not None and "anchor_mae" in delta_stats:
            self.writer.add_scalar(
                "train/anchor_mae", delta_stats["anchor_mae"], self.global_step
            )
            self.writer.add_scalar(
                "train/prediction_mae", delta_stats["prediction_mae"], self.global_step
            )
            anchor_mae = delta_stats["anchor_mae"]
            if anchor_mae > 0:
                delta_contribution = 1.0 - delta_stats["prediction_mae"] / anchor_mae
            else:
                delta_contribution = 0.0
            self.writer.add_scalar(
                "train/delta_contribution", delta_contribution, self.global_step
            )

        # Performance metrics
        self.writer.add_scalar("perf/samples_per_sec", samples_per_sec, self.global_step)
        self.writer.add_scalar("perf/step_time_ms", step_time * 1000, self.global_step)

        # AMP scale (track GradScaler health)
        self.writer.add_scalar(
            "train/amp_scale", self.scaler.get_scale(), self.global_step
        )

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
        total_recon_tensor = torch.tensor(0.0, device=self.device)
        total_delta_tensor = torch.tensor(0.0, device=self.device)
        total_corr_tensor = torch.tensor(0.0, device=self.device)
        total_delta_sup_tensor = torch.tensor(0.0, device=self.device)
        num_batches = 0
        total_samples = 0
        scaler_skips = 0
        optimizer_steps = 0
        accum_steps = self.config.gradient_accumulation_steps

        self.optimizer.zero_grad()

        # Set up progress bar - only show on main process
        show_pbar = use_progress_bar and is_main_process()
        if show_pbar:
            bar_format = (
                "Epoch {desc}: {percentage:3.0f}%|{bar}| "
                "{n_fmt}/{total_fmt} [{elapsed} elapsed, {remaining} left, "
                "{rate_fmt}]{postfix}\n"
            )
            pbar = tqdm(dataloader, desc=f"{epoch + 1}", bar_format=bar_format, leave=True)
        else:
            pbar = dataloader

        for step, batch in enumerate(pbar):
            # Start timing at beginning of accumulation cycle
            if step % accum_steps == 0:
                accum_start_time = time.time()
                accum_samples = 0

            loss, components = self._train_step_accumulate(batch, accum_steps)
            actual_loss = loss * accum_steps  # Undo the division for logging

            # Skip NaN batches for accumulation (backward was already skipped)
            if math.isnan(actual_loss):
                continue

            # Accumulate on GPU tensor - no sync
            total_loss_tensor += actual_loss
            total_recon_tensor += components["recon_loss"] * accum_steps
            total_delta_tensor += components["delta_reg"] * accum_steps
            total_corr_tensor += components.get("corr_loss", 0.0) * accum_steps
            total_delta_sup_tensor += components.get("delta_sup_loss", 0.0) * accum_steps
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
                old_scale = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                # Only advance LR when optimizer actually stepped
                optimizer_steps += 1
                if self.scaler.get_scale() >= old_scale:
                    self.scheduler.step()
                else:
                    scaler_skips += 1
                self.global_step += 1

                # Calculate throughput for full accumulation cycle
                accum_time = time.time() - accum_start_time
                samples_per_sec = accum_samples / accum_time if accum_time > 0 else 0

                # Update progress bar (only on main process)
                if show_pbar and hasattr(pbar, "set_postfix"):
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    pbar.set_postfix(
                        loss=f"{actual_loss:.4f}",
                        lr=f"{current_lr:.2e}",
                        throughput=f"{samples_per_sec:.1f}",
                    )

                # Log step metrics to TensorBoard (only on main process)
                if self.global_step % 100 == 0 and is_main_process():
                    self._log_step_metrics(
                        actual_loss,
                        components["recon_loss"] * accum_steps,
                        grad_norm,
                        samples_per_sec,
                        accum_time,
                        delta_stats=components.get("delta_stats"),
                        delta_reg=components["delta_reg"] * accum_steps,
                        corr_loss=components.get("corr_loss", 0.0) * accum_steps,
                        delta_sup_loss=components.get("delta_sup_loss", 0.0) * accum_steps,
                    )

        # Handle remaining gradients at epoch end
        if num_batches % accum_steps != 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )
            old_scale = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            optimizer_steps += 1
            if self.scaler.get_scale() >= old_scale:
                self.scheduler.step()
            else:
                scaler_skips += 1
            self.global_step += 1

        # Only sync to CPU at epoch end for final average
        n = max(num_batches, 1)
        total_loss = total_loss_tensor.item()
        total_recon = total_recon_tensor.item()
        total_delta = total_delta_tensor.item()
        total_corr = total_corr_tensor.item()
        total_delta_sup = total_delta_sup_tensor.item()

        # Calculate epoch timing
        epoch_time = time.time() - epoch_start_time
        avg_throughput = total_samples / epoch_time if epoch_time > 0 else 0

        metrics = {
            "loss": total_loss / n,
            "recon_loss": total_recon / n,
            "delta_reg": total_delta / n,
            "corr_loss": total_corr / n,
            "delta_sup_loss": total_delta_sup / n,
            "num_batches": num_batches,
            "num_samples": total_samples,
            "epoch_time": epoch_time,
            "throughput": avg_throughput,
        }

        # Log epoch metrics (only on main process)
        if self.writer is not None:
            self.writer.add_scalar("train/loss_epoch", metrics["loss"], epoch)
            self.writer.add_scalar("train/recon_loss_epoch", metrics["recon_loss"], epoch)
            self.writer.add_scalar("train/delta_reg_epoch", metrics["delta_reg"], epoch)
            self.writer.add_scalar("train/corr_loss_epoch", metrics["corr_loss"], epoch)
            self.writer.add_scalar("train/delta_sup_loss_epoch", metrics["delta_sup_loss"], epoch)
            self.writer.add_scalar("train/epoch_time_s", epoch_time, epoch)
            self.writer.add_scalar("train/throughput_epoch", avg_throughput, epoch)
            if optimizer_steps > 0:
                skip_rate = scaler_skips / optimizer_steps
                self.writer.add_scalar("train/scaler_skip_rate", skip_rate, epoch)

        return metrics

    def _sample_target_indices(
        self,
        batch_size: int,
        num_timesteps: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Sample random target indices from explicit indices, predict_years, or forecast_horizon.

        Priority (matches FinetuningConfig.get_target_indices()):
          1. predict_target_indices set -> sample uniformly from that list.
          2. predict_years set -> sample uniformly from those years' timesteps.
          3. otherwise -> sample uniformly from the last `forecast_horizon` timesteps.

        Args:
            batch_size: Number of samples in batch.
            num_timesteps: Total number of timesteps in sequence.
            generator: Optional RNG generator for deterministic sampling.

        Returns:
            Target indices of shape (B,), values depend on configuration.
        """
        if self.config.predict_target_indices or self.config.predict_years:
            indices = self.config.get_target_indices()
            idx_tensor = torch.tensor(indices, device=self.device)
            samples = torch.randint(
                0, len(indices), (batch_size,),
                device=self.device, generator=generator,
            )
            return idx_tensor[samples]

        # Default: sample from forecast horizon
        horizon = self.config.forecast_horizon
        min_idx = num_timesteps - horizon
        target_idx = torch.randint(
            min_idx, num_timesteps, (batch_size,),
            device=self.device, generator=generator,
        )
        return target_idx

    def _apply_temporal_subsampling(
        self,
        input_images: torch.Tensor,
        input_season_ids: torch.Tensor,
        input_year_indices: torch.Tensor,
        target_season: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply temporal year subsampling, ensuring the anchor year is included.

        Reuses _sample_year_subset from pre-training but guarantees the anchor
        observation year (most recent same-season) is in the sampled set.

        Note: seq_positions are NOT passed to the model. The model identifies
        each timestep via season_ids (season embedding + cyclical encoding) and
        year_indices (sensor era embedding + multi-frequency year sinusoid),
        which are already subsampled alongside images.

        Args:
            input_images: Input images (B, T_in, H, W).
            input_season_ids: Season indices (B, T_in).
            input_year_indices: Year indices (B, T_in).
            target_season: Target season per sample (B,).

        Returns:
            Tuple of (images, season_ids, year_indices).
        """
        K = self.config.temporal_subset_years
        T_in = input_images.shape[1]
        total_years = T_in // 4

        # Skip if K >= available years
        if K <= 0 or K >= total_years:
            return input_images, input_season_ids, input_year_indices

        # Find the anchor year for each sample (most recent same-season obs)
        anchor_year = self._find_anchor_year(
            input_season_ids, target_season, total_years
        )

        # Sample K years, ensuring anchor_year is included
        selected = self._sample_years_with_anchor(total_years, K, anchor_year)

        # Build timestep indices from selected years
        timestep_indices = []
        for y in selected:
            base = y * 4
            timestep_indices.extend(range(base, base + 4))
        idx = torch.tensor(timestep_indices, device=input_images.device)

        return (
            input_images[:, idx],
            input_season_ids[:, idx],
            input_year_indices[:, idx],
        )

    def _find_anchor_year(
        self,
        season_ids: torch.Tensor,
        target_season: torch.Tensor,
        total_years: int,
    ) -> int:
        """Find the year containing the anchor observation.

        The anchor is the last same-season observation in the input sequence.
        Returns a single year index (same for the whole batch since we need
        a consistent subset).

        Args:
            season_ids: Season indices (B, T).
            target_season: Target season (B,).
            total_years: Number of years in the input sequence.

        Returns:
            Year index (0-based) of the anchor observation.
        """
        # Use first sample to determine anchor year (same structure across batch)
        target_s = target_season[0].item()
        seasons = season_ids[0]  # (T,)
        match_mask = (seasons == target_s)

        if match_mask.any():
            last_match_idx = match_mask.nonzero(as_tuple=False)[-1].item()
            return last_match_idx // 4
        # Fallback: last year
        return total_years - 1

    def _sample_years_with_anchor(
        self, total_years: int, num_years: int, anchor_year: int,
    ) -> list:
        """Sample K years ensuring anchor_year is included.

        Args:
            total_years: Total number of years available.
            num_years: Number of years to sample (K).
            anchor_year: Year index that must be included.

        Returns:
            Sorted list of selected year indices.
        """
        other_years = [y for y in range(total_years) if y != anchor_year]
        sampled = random.sample(other_years, num_years - 1)
        sampled.append(anchor_year)
        return sorted(sampled)

    def _extract_target_from_batch(
        self,
        images: torch.Tensor,
        season_ids: torch.Tensor,
        year_indices: torch.Tensor,
        nodata_mask: Optional[torch.Tensor],
        target_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Extract target data using vectorized advanced indexing.

        Args:
            images: Full image sequence (B, T, H, W).
            season_ids: Season indices (B, T).
            year_indices: Year indices (B, T).
            nodata_mask: Optional nodata mask (B, T, H, W).
            target_idx: Per-sample target timestep indices (B,).

        Returns:
            Tuple of (target, target_season, target_year, target_nodata).
        """
        batch_idx = torch.arange(images.shape[0], device=images.device)
        target = images[batch_idx, target_idx].unsqueeze(1)  # (B, 1, H, W)
        target_season = season_ids[batch_idx, target_idx]  # (B,)
        target_year = year_indices[batch_idx, target_idx]  # (B,)
        target_nodata = None
        if nodata_mask is not None:
            target_nodata = nodata_mask[batch_idx, target_idx].unsqueeze(1)
        return target, target_season, target_year, target_nodata

    def _select_input_timesteps(
        self,
        images: torch.Tensor,
        season_ids: torch.Tensor,
        year_indices: torch.Tensor,
        target_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select input timesteps based on strategy configuration.

        With use_all_history_before_target: uses all timesteps before each
        sample's target (variable length, per-sample loop required).
        Otherwise: uses fixed window of first T-horizon timesteps.

        Args:
            images: Full image sequence (B, T, H, W).
            season_ids: Season indices (B, T).
            year_indices: Year indices (B, T).
            target_idx: Per-sample target timestep indices (B,).

        Returns:
            Tuple of (input_images, input_season_ids, input_year_indices).
        """
        B, T = images.shape[0], images.shape[1]

        if not self.config.use_all_history_before_target:
            fixed_len = ((T - self.config.forecast_horizon) // 4) * 4
            return images[:, :fixed_len], season_ids[:, :fixed_len], year_indices[:, :fixed_len]

        # Variable length: use last min_input_len timesteps before each target
        idx_range = torch.arange(T, device=self.device).unsqueeze(0)
        min_len = (idx_range < target_idx.unsqueeze(1)).sum(dim=1).min().item()
        min_len = (min_len // 4) * 4  # Round to multiple of 4 for temporal transformer

        img_list, season_list, year_list = [], [], []
        for b in range(B):
            end = target_idx[b].item()
            start = end - min_len
            img_list.append(images[b, start:end])
            season_list.append(season_ids[b, start:end])
            year_list.append(year_indices[b, start:end])
        return torch.stack(img_list), torch.stack(season_list), torch.stack(year_list)

    def _maybe_compute_anchor(
        self,
        input_images: torch.Tensor,
        input_season_ids: torch.Tensor,
        target_season: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Compute climatology anchor unless anchor_mode='none'."""
        if self._anchor_mode == "none":
            return None
        return self._base_model._get_climatology_anchor(
            input_images, input_season_ids, target_season,
        )

    def _compute_anchor_mae(
        self,
        precomputed_anchor: Optional[torch.Tensor],
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[float, float]:
        """Return (anchor_mae, prediction_mae). anchor_mae=0 when anchor-free."""
        pred_mae = (prediction.detach() - target).abs().mean().item()
        if precomputed_anchor is None:
            return 0.0, pred_mae
        anchor_mae = (precomputed_anchor.unsqueeze(1) - target).abs().mean().item()
        return anchor_mae, pred_mae

    def _forward_and_backward(
        self,
        input_images: torch.Tensor,
        input_season_ids: torch.Tensor,
        input_year_indices: torch.Tensor,
        lat: torch.Tensor,
        lon: torch.Tensor,
        target_season: torch.Tensor,
        target_year: torch.Tensor,
        precomputed_anchor: Optional[torch.Tensor],
        target: torch.Tensor,
        target_nodata: Optional[torch.Tensor],
        accum_steps: int,
    ) -> Tuple[float, Dict[str, float]]:
        """Run forward pass, compute loss, and backward pass.

        Returns:
            Tuple of (scaled_loss_value, scaled_loss_components_dict).
        """
        with torch.amp.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.device.type == "cuda"):
            output = self.model(
                input_images, input_season_ids, input_year_indices,
                lat, lon, target_season, target_year,
                precomputed_anchor=precomputed_anchor,
            )
            loss, components = self.loss_fn(output, target, target_nodata)
            loss = loss / accum_steps

        # NaN safety: sync the NaN flag across ranks so all ranks skip
        # together. Skipping locally on rank 0 while rank 1 enters backward
        # causes rank 1 to hang at the gradient all-reduce waiting for a
        # peer that's never coming — this is what silently killed the
        # last_step ablation at step 73 / 133 of epoch 1.
        nan_flag = torch.tensor(
            [float(not torch.isfinite(loss))], device=self.device,
        )
        if self.distributed and dist.is_available() and dist.is_initialized():
            dist.all_reduce(nan_flag, op=dist.ReduceOp.MAX)
        if nan_flag.item() > 0:
            self.optimizer.zero_grad()
            logger.warning(
                f"NaN/Inf loss detected at step {self.global_step} "
                "(some rank), skipping batch on all ranks"
            )
            components_dict = {k: 0.0 for k in components}
            components_dict["delta_stats"] = None
            return float("nan"), components_dict

        delta_stats = self._base_model.decoder.get_delta_statistics(
            output["delta"].detach()
        )
        # Anchor vs prediction quality (detached, no grad)
        anchor_mae, pred_mae = self._compute_anchor_mae(
            precomputed_anchor, output["prediction"], target,
        )
        delta_stats["anchor_mae"] = anchor_mae
        delta_stats["prediction_mae"] = pred_mae

        self.scaler.scale(loss).backward()

        components_dict = {k: v.item() / accum_steps for k, v in components.items()}
        components_dict["delta_stats"] = delta_stats
        return loss.item(), components_dict

    def _train_step_accumulate(
        self, batch: Dict[str, torch.Tensor], accum_steps: int
    ) -> Tuple[float, Dict[str, float]]:
        """Execute forward/backward pass for gradient accumulation.

        Uses variable target training: randomly samples target from the last
        `forecast_horizon` timesteps rather than always using the last timestep.

        Args:
            batch: Batch dictionary from dataloader.
            accum_steps: Number of accumulation steps (for loss scaling).

        Returns:
            Tuple of (scaled_loss_value, scaled_loss_components_dict).
        """
        images, season_ids, year_indices, lat, lon, nodata_mask = (
            self._move_batch_to_device(batch)
        )
        B, T, H, W = images.shape
        target_idx = self._sample_target_indices(B, T)

        target, target_season, target_year, target_nodata = (
            self._extract_target_from_batch(
                images, season_ids, year_indices, nodata_mask, target_idx,
            )
        )
        input_images, input_season_ids, input_year_indices = (
            self._select_input_timesteps(
                images, season_ids, year_indices, target_idx,
            )
        )

        # Compute anchor on full input BEFORE subsampling for higher quality.
        # Skip when anchor_mode='none' (direct absolute prediction).
        precomputed_anchor = self._maybe_compute_anchor(
            input_images, input_season_ids, target_season,
        )
        input_images, input_season_ids, input_year_indices = (
            self._apply_temporal_subsampling(
                input_images, input_season_ids, input_year_indices, target_season,
            )
        )

        return self._forward_and_backward(
            input_images, input_season_ids, input_year_indices,
            lat, lon, target_season, target_year,
            precomputed_anchor, target, target_nodata, accum_steps,
        )

    def _validate_batch(
        self,
        batch: Dict[str, torch.Tensor],
        val_generator: torch.Generator,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, Optional[torch.Tensor], int]:
        """Process a single validation batch.

        Returns:
            Tuple of (loss, output_dict, target, target_nodata, batch_size).
        """
        images, season_ids, year_indices, lat, lon, nodata_mask = (
            self._move_batch_to_device(batch)
        )
        B, T = images.shape[0], images.shape[1]
        target_idx = self._sample_target_indices(B, T, generator=val_generator)

        target, target_season, target_year, target_nodata = (
            self._extract_target_from_batch(
                images, season_ids, year_indices, nodata_mask, target_idx,
            )
        )
        input_images, input_season_ids, input_year_indices = (
            self._select_input_timesteps(images, season_ids, year_indices, target_idx)
        )

        # No subsampling in validation — model computes anchor internally.
        # Call _base_model (not self.model) so we bypass DDP entirely during
        # validation. Validation runs only on rank 0; going through the DDP
        # wrapper would trigger a _sync_buffers collective that blocks waiting
        # for rank 1 (which has already moved on to the next training epoch),
        # causing a ~30 min gloo timeout per first validation after training.
        with torch.amp.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.device.type == "cuda"):
            output = self._base_model(
                input_images, input_season_ids, input_year_indices,
                lat, lon, target_season, target_year,
            )
            loss, _ = self.loss_fn(output, target, target_nodata)
        return loss.detach(), output, target, target_nodata, B

    def _validate_with_retry(
        self, val_loader: DataLoader, max_attempts: int = 3,
    ) -> Optional[Dict[str, float]]:
        """Run validation with retry on DDP timeout.

        DDP gloo on Windows can timeout during _sync_buffers at the start
        of validation. This is transient — retrying usually succeeds.

        Args:
            val_loader: Validation data loader.
            max_attempts: Number of attempts before giving up.

        Returns:
            Validation metrics dict, or None if all attempts failed.
        """
        for attempt in range(max_attempts):
            try:
                return self.validate(val_loader)
            except Exception as e:
                logger.warning(
                    f"Validation attempt {attempt + 1}/{max_attempts} failed: {e}"
                )
        logger.error("All validation attempts failed, skipping this epoch's validation")
        return None

    def _process_val_metrics(
        self, val_metrics: Dict[str, float], history: Dict, epoch: int,
    ) -> None:
        """Log and record validation metrics."""
        history["val_loss"].append(val_metrics["val_loss"])
        history["val_mae"].append(val_metrics["val_mae"])
        history["val_r2"].append(val_metrics["val_r2"])
        history["val_bias"].append(val_metrics["val_bias"])

        logger.info(
            f"  Val: loss={val_metrics['val_loss']:.4f} | "
            f"MAE={val_metrics['val_mae']:.4f} | "
            f"RMSE={val_metrics['val_rmse']:.4f} | "
            f"r²={val_metrics['val_r2']:.4f} | "
            f"R²={val_metrics['val_R2_det']:.4f} | "
            f"bias={val_metrics['val_bias']:.4f} | "
            f"time={_format_time(val_metrics['val_time'])}"
        )

        if self.writer is not None:
            for key, value in val_metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"val/{key}", value, epoch)

    @torch.no_grad()
    def validate(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        """Run validation.

        Uses same random target sampling as training for consistency.
        RNG states are saved and restored to ensure deterministic results.

        Args:
            dataloader: Validation data loader.

        Returns:
            Dictionary with validation metrics.
        """
        val_start_time = time.time()
        self._assert_no_batchnorm()
        self._base_model.eval()

        # Fixed-seed generator for deterministic target sampling
        val_generator = torch.Generator(device=self.device)
        val_generator.manual_seed(42)

        metrics_tracker = PerSampleMetricsTracker()
        anchor_metrics_tracker = PerSampleMetricsTracker()
        total_loss = torch.tensor(0.0, device=self.device)
        total_delta_abs = 0.0
        num_batches = 0
        total_samples = 0

        if is_main_process():
            bar_format = (
                "{desc}: {percentage:3.0f}%|{bar}| "
                "{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]\n"
            )
            iterator = tqdm(dataloader, desc="Validation", bar_format=bar_format, leave=True)
        else:
            iterator = dataloader

        for batch in iterator:
            loss, output, target, target_nodata, B = self._validate_batch(
                batch, val_generator,
            )
            total_loss += loss
            num_batches += 1
            total_samples += B

            valid_mask = ~target_nodata if target_nodata is not None else None
            metrics_tracker.update(output["prediction"], target, valid_mask)
            anchor_metrics_tracker.update(output["anchor"], target, valid_mask)
            total_delta_abs += output["delta"].abs().mean().item()

        avg_loss = total_loss.item() / max(num_batches, 1)
        val_time = time.time() - val_start_time
        prediction_metrics = metrics_tracker.compute()
        anchor_metrics = anchor_metrics_tracker.compute()

        return {
            "val_loss": avg_loss,
            "val_mae": prediction_metrics["mae"],
            "val_rmse": prediction_metrics["rmse"],
            "val_r2": prediction_metrics["r2"],
            "val_R2_det": prediction_metrics["R2_det"],
            "val_bias": prediction_metrics["bias"],
            "val_anchor_mae": anchor_metrics["mae"],
            "val_delta_abs_mean": total_delta_abs / max(num_batches, 1),
            "num_samples": prediction_metrics["n_samples"],
            "val_time": val_time,
        }

    @torch.no_grad()
    def _estimate_delta_std(
        self, val_loader: DataLoader, max_batches: int = 20,
    ) -> float:
        """Estimate std of (target - anchor) from validation data.

        Used to normalize the delta supervision loss so that the gradient
        signal is independent of the absolute magnitude of vegetation changes.

        Args:
            val_loader: Validation data loader.
            max_batches: Maximum number of batches to use for estimation.

        Returns:
            Estimated standard deviation of true deltas.
        """
        # Use _base_model for consistency with validate() — see _validate_batch
        # for why we bypass DDP for inference-only paths.
        self._base_model.eval()
        gen = torch.Generator(device=self.device)
        gen.manual_seed(42)

        all_deltas = []
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            true_deltas = self._compute_batch_true_deltas(batch, gen)
            if true_deltas.numel() > 0:
                all_deltas.append(true_deltas.cpu())

        self._base_model.train()
        if not all_deltas:
            return 1.0
        all_deltas = torch.cat(all_deltas)
        median = all_deltas.median()
        delta_std = (all_deltas - median).abs().median().item() * 1.4826
        return max(delta_std, 1e-6)  # Guard against zero std

    @torch.no_grad()
    def _compute_batch_true_deltas(
        self, batch: Dict[str, torch.Tensor], gen: torch.Generator,
    ) -> torch.Tensor:
        """Compute (target - anchor) for valid pixels in a single batch."""
        images, season_ids, year_indices, lat, lon, nodata_mask = (
            self._move_batch_to_device(batch)
        )
        B, T = images.shape[0], images.shape[1]
        target_idx = self._sample_target_indices(B, T, generator=gen)

        target, target_season, _, target_nodata = (
            self._extract_target_from_batch(
                images, season_ids, year_indices, nodata_mask, target_idx,
            )
        )
        input_images, input_season_ids, _ = (
            self._select_input_timesteps(images, season_ids, year_indices, target_idx)
        )
        anchor = self._base_model._get_climatology_anchor(
            input_images, input_season_ids, target_season,
        )  # (B, 1, H, W)
        true_delta = target - anchor
        if target_nodata is not None:
            return true_delta[~target_nodata]
        return true_delta.flatten()

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
        # Auto-estimate delta_std for delta supervision normalization.
        # Skipped automatically when anchor_mode='none' (flag defaults to False).
        if self._need_delta_std_estimation and val_loader is not None:
            estimated_std = self._estimate_delta_std(val_loader)
            self.loss_fn.delta_std = estimated_std
            if is_main_process():
                logger.info(f"Auto-estimated delta_std={estimated_std:.6f}")

        # Set steps_per_epoch from dataloader if not provided
        if self.steps_per_epoch is None:
            self.steps_per_epoch = len(train_loader) // self.config.gradient_accumulation_steps
            # Recreate scheduler with correct warmup steps
            self.scheduler = self._create_scheduler()
            if self.global_step > 0:
                self.scheduler.last_epoch = self.global_step
                self.scheduler._step_count = self.global_step + 1
            if is_main_process():
                logger.info(f"Using {self.steps_per_epoch} steps per epoch for LR schedule")

        history = {
            "train_loss": [],
            "val_loss": [],
            "val_mae": [],
            "val_r2": [],
            "val_bias": [],
        }

        for epoch in range(self.epoch, self.config.max_epochs):
            self.epoch = epoch
            epoch_start_time = time.time()

            # Set epoch on DistributedSampler for proper shuffling
            if self.distributed and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            if is_main_process():
                logger.info(f"Epoch {epoch + 1}/{self.config.max_epochs}")

            # Train
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

            # Save checkpoint BEFORE validation so weights are preserved
            # even if validation crashes (DDP/gloo timeout on Windows).
            if (epoch + 1) % self.config.checkpoint_every_n_epochs == 0:
                self._save_checkpoint(self.output_dir / f"epoch_{epoch + 1}.pt")

            # Validation with retry (main process only).
            # DDP gloo on Windows can timeout during _sync_buffers at the
            # start of validation. This is a transient issue — retrying
            # usually succeeds because the DDP state resets between attempts.
            val_loss = None
            saved_best = False
            if val_loader is not None and is_main_process():
                val_metrics = self._validate_with_retry(val_loader, max_attempts=3)
                if val_metrics is not None:
                    self._process_val_metrics(
                        val_metrics, history, epoch,
                    )
                    val_loss = val_metrics["val_loss"]
                    prev_best = self.best_mae
                    saved_best = val_metrics["val_mae"] < self.best_mae
                    if saved_best:
                        self.best_mae = val_metrics["val_mae"]
                        self.best_loss = val_loss
                        self._save_checkpoint(self.output_dir / "best.pt")

            # Epoch summary
            if is_main_process():
                total_epoch_time = time.time() - epoch_start_time
                summary_parts = [
                    f"Epoch {epoch + 1} complete",
                    f"train_loss={train_metrics['loss']:.4f}",
                ]
                if val_loss is not None:
                    summary_parts.append(f"val_mae={val_metrics['val_mae']:.4f}")
                    if saved_best:
                        improvement = prev_best - val_metrics["val_mae"]
                        summary_parts.append(f"NEW BEST MAE (improved {improvement:.4f})")
                    else:
                        gap = val_metrics["val_mae"] - self.best_mae
                        summary_parts.append(f"best_mae={self.best_mae:.4f} (+{gap:.4f})")
                summary_parts.append(f"total_time={_format_time(total_epoch_time)}")
                logger.info("  " + " | ".join(summary_parts))

        # Final checkpoint
        self._save_checkpoint(self.output_dir / "final.pt")

        if self.writer is not None:
            self.writer.close()
        return history

    def _save_checkpoint(self, path: Path) -> None:
        """Save training checkpoint including RNG states for reproducibility.

        Only saves on main process in distributed mode.
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
            "best_mae": self.best_mae,
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "config": self.config,
            # Persist model architecture so eval can rebuild the same arch
            # without depending on directory naming conventions or external
            # ARCH_OVERRIDES maps. Older checkpoints lack this field.
            "model_config": self._base_model.config,
            # RNG states for reproducibility
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def _load_checkpoint(self, path: Path) -> None:
        """Load training checkpoint including RNG states for reproducibility."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Load into base model (unwrapped from DDP)
        self._base_model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.epoch = checkpoint["epoch"] + 1
        self.global_step = checkpoint["global_step"]
        self.best_loss = checkpoint["best_loss"]
        self.best_mae = checkpoint.get("best_mae", float("inf"))

        # epoch_N.pt is saved BEFORE validation, so best_mae/best_loss in the
        # resumed checkpoint may be stale (inf on first epoch). best.pt is
        # always written right after validation improves, so it holds the
        # true best. Prefer its stats when available.
        self._recover_best_from_best_pt()

        # Restore RNG states for reproducibility
        # RNG states must be CPU ByteTensors regardless of training device
        if "torch_rng_state" in checkpoint:
            rng_state = checkpoint["torch_rng_state"]
            if not isinstance(rng_state, torch.ByteTensor):
                rng_state = rng_state.cpu().byte()
            torch.set_rng_state(rng_state)
        if checkpoint.get("cuda_rng_state") and torch.cuda.is_available():
            cuda_states = checkpoint["cuda_rng_state"]
            cuda_states = [s.cpu() if s.device.type != "cpu" else s for s in cuda_states]
            torch.cuda.set_rng_state_all(cuda_states)
        if "numpy_rng_state" in checkpoint:
            np.random.set_state(checkpoint["numpy_rng_state"])
        if "python_rng_state" in checkpoint:
            random.setstate(checkpoint["python_rng_state"])

        if is_main_process():
            logger.info(f"Resumed from {path} at epoch {self.epoch}")

    def _recover_best_from_best_pt(self) -> None:
        """Restore best_mae/best_loss from best.pt if it holds better values.

        epoch_N.pt is snapshotted before validation runs, so its best_mae is
        stale (inf on the first epoch). best.pt is only written when a new
        best is achieved, so it is the authoritative source.
        """
        best_pt = self.output_dir / "best.pt"
        if not best_pt.exists():
            return
        try:
            best_ckpt = torch.load(best_pt, map_location="cpu", weights_only=False)
        except Exception as exc:
            if is_main_process():
                logger.warning(f"Could not read best.pt for best_mae recovery: {exc}")
            return
        best_mae = best_ckpt.get("best_mae", float("inf"))
        best_loss = best_ckpt.get("best_loss", float("inf"))
        if best_mae < self.best_mae:
            if is_main_process():
                logger.info(
                    f"Recovered best_mae={best_mae:.4f} from best.pt "
                    f"(checkpoint had {self.best_mae})"
                )
            self.best_mae = best_mae
            self.best_loss = best_loss
