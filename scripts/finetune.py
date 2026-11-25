#!/usr/bin/env python3
"""Fine-tuning entry point for SeasonalViT forecasting.

Usage:
    # Single GPU
    python scripts/finetune.py --train-dir /path/to/train --val-dir /path/to/val --pretrained /path/to/pretrain.pt

    # Multi-GPU with torchrun (Linux/macOS)
    torchrun --nproc_per_node=2 scripts/finetune.py --train-dir DATA/train --val-dir DATA/val --pretrained MODEL.pt --multi-gpu

    # Multi-GPU with mp.spawn (Windows - avoids libuv issue)
    python scripts/finetune.py --train-dir DATA/train --val-dir DATA/val --pretrained MODEL.pt --multi-gpu --spawn --num-gpus 2
"""

import argparse
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from datasets.sharded_dataset import InMemoryNDVIDataset, create_dataloader
from datasets.prefetch_dataset import ShardPrefetchLoader
from model.config import SeasonalViTConfig, FinetuningConfig
from model.seasonal_vit import SeasonalViT
from train.distributed import (
    setup_distributed,
    setup_distributed_spawn,
    cleanup_distributed,
    is_main_process,
    get_local_rank,
    get_rank,
)
from train.finetune import FinetuningTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def setup_file_logging(output_dir: Path) -> None:
    """Add file handler to write logs to output directory."""
    log_file = output_dir / "training.log"
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logging.getLogger().addHandler(file_handler)


@dataclass
class FinetuneRunConfig:
    """Picklable configuration for fine-tuning run.

    Encapsulates all arguments needed to configure a fine-tuning run,
    avoiding the need to reconstruct argparse.Namespace in subprocesses.
    """

    train_dir: str
    val_dir: str
    output_dir: str
    pretrained: Optional[str]
    resume: Optional[str]
    epochs: Optional[int]
    batch_size: Optional[int]
    lr: Optional[float]
    freeze_encoder: bool
    freeze_spatial: bool
    limit: Optional[int]
    train_shard_range: Optional[str]
    val_shard_range: Optional[str]
    num_workers: Optional[int]
    predict_years: Optional[List[int]]
    temporal_subset_years: Optional[int] = None
    val_batch_size: Optional[int] = None
    gradient_accumulation_steps: Optional[int] = None
    warmup_epochs: Optional[int] = None
    weight_decay: Optional[float] = None
    decoder_lr_multiplier: Optional[float] = None
    delta_reg_weight: Optional[float] = None
    correlation_loss_weight: Optional[float] = None
    delta_supervision_weight: Optional[float] = None
    delta_std: Optional[float] = None
    max_grad_norm: Optional[float] = None
    # Ablation flags (default None = preserve current model defaults)
    anchor_mode: Optional[str] = None
    temporal_pathway: Optional[str] = None

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "FinetuneRunConfig":
        """Create config from parsed arguments."""
        return cls(
            train_dir=str(args.train_dir),
            val_dir=str(args.val_dir),
            output_dir=str(args.output_dir) if args.output_dir else "outputs/finetune",
            pretrained=str(args.pretrained) if args.pretrained else None,
            resume=str(args.resume) if args.resume else None,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            freeze_encoder=args.freeze_encoder,
            freeze_spatial=args.freeze_spatial,
            limit=args.limit,
            train_shard_range=args.train_shard_range,
            val_shard_range=args.val_shard_range,
            num_workers=args.num_workers,
            predict_years=args.predict_years,
            temporal_subset_years=args.temporal_subset_years,
            val_batch_size=args.val_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_epochs=args.warmup_epochs,
            weight_decay=args.weight_decay,
            decoder_lr_multiplier=args.decoder_lr_multiplier,
            delta_reg_weight=args.delta_reg_weight,
            correlation_loss_weight=args.corr_loss_weight,
            delta_supervision_weight=args.delta_supervision_weight,
            delta_std=args.delta_std,
            max_grad_norm=args.max_grad_norm,
            anchor_mode=args.anchor_mode,
            temporal_pathway=args.temporal_pathway,
        )

    @property
    def output_path(self) -> Path:
        """Return output directory as Path."""
        return Path(self.output_dir)

    @property
    def resume_path(self) -> Optional[Path]:
        """Return resume checkpoint as Path, or None."""
        return Path(self.resume) if self.resume else None

    @property
    def pretrained_path(self) -> Optional[Path]:
        """Return pretrained checkpoint as Path, or None."""
        return Path(self.pretrained) if self.pretrained else None

    @property
    def train_path(self) -> Path:
        """Return training directory as Path."""
        return Path(self.train_dir)

    @property
    def val_path(self) -> Path:
        """Return validation directory as Path."""
        return Path(self.val_dir)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune SeasonalViT for NDVI forecasting"
    )

    # Data arguments
    parser.add_argument(
        "--train-dir",
        type=Path,
        required=True,
        help="Directory containing training shards (shard_*.safetensors files)",
    )
    parser.add_argument(
        "--val-dir",
        type=Path,
        required=True,
        help="Directory containing validation shards (shard_*.safetensors files)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/finetune"),
        help="Directory for checkpoints and logs",
    )

    # Pre-trained model
    parser.add_argument(
        "--pretrained",
        type=Path,
        help="Path to pre-trained checkpoint",
    )

    # Training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Training batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate",
    )

    # Encoder freezing
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze pre-trained encoder weights",
    )
    parser.add_argument(
        "--freeze-spatial",
        action="store_true",
        help="Freeze only spatial encoder (Swin-T) to speed up training",
    )

    # Prediction target
    parser.add_argument(
        "--predict-years",
        type=int,
        nargs="+",
        default=None,
        help="Years to predict (e.g., --predict-years 2022). Overrides forecast_horizon.",
    )

    # Temporal subsampling
    parser.add_argument(
        "--temporal-subset-years",
        type=int,
        default=None,
        help="Sample K years per forward pass to reduce memory (default: 12). Set 0 to disable.",
    )
    parser.add_argument(
        "--val-batch-size",
        type=int,
        default=None,
        help="Validation batch size (default: auto-derived from batch_size and subsampling)",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="Number of gradient accumulation steps (default: 1)",
    )
    parser.add_argument(
        "--delta-reg-weight",
        type=float,
        default=None,
        help="Weight for L2 delta regularization loss (default: 0.0, disabled).",
    )
    parser.add_argument(
        "--corr-loss-weight",
        type=float,
        default=None,
        help="Weight for per-sample correlation loss, which optimizes r² "
        "(default: 0.5, the value used in the paper). Set 0 to disable.",
    )
    parser.add_argument(
        "--delta-supervision-weight",
        type=float,
        default=None,
        help="Weight for direct delta supervision loss (default: 0.1; the paper used 0.2). "
        "Directly supervises delta to match (target - anchor), normalized by delta_std.",
    )
    parser.add_argument(
        "--delta-std",
        type=float,
        default=None,
        help="Std of historical (target - anchor) for delta normalization. "
        "0 = auto-estimate from validation data (default). Set manually to skip estimation.",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=None,
        help="Number of warmup epochs (default: 1)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Weight decay for AdamW optimizer (default: 0.01)",
    )
    parser.add_argument(
        "--decoder-lr-multiplier",
        type=float,
        default=None,
        help="Learning rate multiplier for decoder (default: 10.0)",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=None,
        help="Maximum gradient norm for clipping (default: 1.0)",
    )

    # Ablation flags
    parser.add_argument(
        "--anchor-mode",
        type=str,
        choices=["same_season", "last_step", "none"],
        default=None,
        help="Anchor mode (ablations). Default preserves same-season anchor.",
    )
    parser.add_argument(
        "--temporal-pathway",
        type=str,
        choices=["both", "cross_season", "cross_year"],
        default=None,
        help="Temporal pathway (ablations). Default preserves dual-scale fusion.",
    )

    # Resume training
    parser.add_argument(
        "--resume",
        type=Path,
        help="Path to checkpoint to resume from",
    )

    # Development / debugging
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of samples (for debugging)",
    )
    parser.add_argument(
        "--train-shard-range",
        type=str,
        help="Range of training shards to load, format: START:END (e.g., 0:100)",
    )
    parser.add_argument(
        "--val-shard-range",
        type=str,
        help="Range of validation shards to load, format: START:END (e.g., 0:10)",
    )

    # Data loading
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Number of DataLoader workers (overrides config)",
    )

    # Multi-GPU training
    parser.add_argument(
        "--multi-gpu",
        action="store_true",
        help="Enable multi-GPU training with DistributedDataParallel",
    )
    parser.add_argument(
        "--spawn",
        action="store_true",
        help="Use mp.spawn instead of torchrun (required for Windows)",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=2,
        help="Number of GPUs when using --spawn (default: 2)",
    )

    return parser.parse_args()


def parse_shard_range(shard_range_str: Optional[str]) -> Optional[Tuple[int, int]]:
    """Parse shard range string into tuple."""
    if shard_range_str is None:
        return None
    start, end = map(int, shard_range_str.split(":"))
    return (start, end)


def create_configs_from_run_config(
    run_config: FinetuneRunConfig,
) -> Tuple[SeasonalViTConfig, FinetuningConfig]:
    """Create model and training configs from FinetuneRunConfig."""
    model_kwargs = {}
    if run_config.anchor_mode is not None:
        model_kwargs["anchor_mode"] = run_config.anchor_mode
    if run_config.temporal_pathway is not None:
        model_kwargs["temporal_pathway"] = run_config.temporal_pathway
    model_config = SeasonalViTConfig(**model_kwargs)

    train_kwargs = {}
    if run_config.epochs is not None:
        train_kwargs["max_epochs"] = run_config.epochs
    if run_config.batch_size is not None:
        train_kwargs["batch_size"] = run_config.batch_size
    if run_config.lr is not None:
        train_kwargs["learning_rate"] = run_config.lr
    if run_config.freeze_encoder:
        train_kwargs["freeze_encoder"] = True
    if run_config.freeze_spatial:
        train_kwargs["freeze_spatial"] = True
    if run_config.num_workers is not None:
        train_kwargs["train_num_workers"] = run_config.num_workers
        train_kwargs["val_num_workers"] = run_config.num_workers
    if run_config.predict_years is not None:
        train_kwargs["predict_years"] = run_config.predict_years
    if run_config.temporal_subset_years is not None:
        train_kwargs["temporal_subset_years"] = run_config.temporal_subset_years
    if run_config.val_batch_size is not None:
        train_kwargs["val_batch_size"] = run_config.val_batch_size
    if run_config.gradient_accumulation_steps is not None:
        train_kwargs["gradient_accumulation_steps"] = run_config.gradient_accumulation_steps
    if run_config.warmup_epochs is not None:
        train_kwargs["warmup_epochs"] = run_config.warmup_epochs
    if run_config.weight_decay is not None:
        train_kwargs["weight_decay"] = run_config.weight_decay
    if run_config.decoder_lr_multiplier is not None:
        train_kwargs["decoder_lr_multiplier"] = run_config.decoder_lr_multiplier
    if run_config.delta_reg_weight is not None:
        train_kwargs["delta_reg_weight"] = run_config.delta_reg_weight
    if run_config.correlation_loss_weight is not None:
        train_kwargs["correlation_loss_weight"] = run_config.correlation_loss_weight
    if run_config.delta_supervision_weight is not None:
        train_kwargs["delta_supervision_weight"] = run_config.delta_supervision_weight
    if run_config.delta_std is not None:
        train_kwargs["delta_std"] = run_config.delta_std
    if run_config.max_grad_norm is not None:
        train_kwargs["max_grad_norm"] = run_config.max_grad_norm

    train_config = FinetuningConfig(**train_kwargs)

    return model_config, train_config


def create_prefetch_loaders(
    run_config: FinetuneRunConfig,
    train_config: FinetuningConfig,
    rank: int,
    world_size: int,
    distributed: bool,
    is_main: bool,
    val_dataset: Optional[InMemoryNDVIDataset] = None,
) -> Tuple[ShardPrefetchLoader, DataLoader]:
    """Create data loaders using ShardPrefetchLoader for training.

    Args:
        run_config: Fine-tuning run configuration.
        train_config: Fine-tuning training configuration.
        rank: Process rank.
        world_size: Total number of processes.
        distributed: Whether distributed training is enabled.
        is_main: Whether this is the main process.
        val_dataset: Optional pre-loaded validation dataset.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    if is_main:
        logger.info(f"Creating ShardPrefetchLoader for {run_config.train_path}")

    train_shard_range = parse_shard_range(run_config.train_shard_range)
    train_loader = ShardPrefetchLoader(
        shard_dir=run_config.train_path,
        batch_size=train_config.batch_size,
        shuffle_shards=True,
        shuffle_within_shard=True,
        num_workers=train_config.train_num_workers,
        prefetch_factor=2,
        pin_memory=True,
        drop_last=True,
        rank=rank,
        world_size=world_size,
        shard_range=train_shard_range,
    )

    if is_main:
        logger.info(
            f"Train shards: {len(train_loader.shard_paths)}, "
            f"~{train_loader.total_samples} samples (estimated)"
        )

    # Use pre-loaded dataset if provided, otherwise load here
    if val_dataset is None:
        if is_main:
            logger.info("Loading validation data into memory...")
        val_shard_range = parse_shard_range(run_config.val_shard_range)
        val_dataset = InMemoryNDVIDataset(
            run_config.val_path,
            shard_range=val_shard_range,
        )
        if distributed:
            val_dataset.share_memory_()

    # All ranks load val data, but only rank 0 validates (no distributed split)
    val_sampler = None
    val_batch_size = train_config.effective_val_batch_size
    val_loader = create_dataloader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=train_config.val_num_workers,
        pin_memory=True,
        prefetch_factor=2,
        sampler=val_sampler,
    )

    if is_main:
        logger.info(f"Val samples: {len(val_dataset)}, val batch_size: {val_batch_size}")
        if train_config.temporal_subset_years > 0:
            logger.info(
                f"Temporal subsampling: K={train_config.temporal_subset_years} years "
                f"({train_config.temporal_subset_years * 4} timesteps per sample)"
            )

    return train_loader, val_loader


def load_pretrained_model(
    checkpoint_path: Optional[Path],
    config: SeasonalViTConfig,
    is_main: bool = True,
) -> SeasonalViT:
    """Load model with optional pre-trained weights.

    Args:
        checkpoint_path: Optional path to pre-trained checkpoint.
        config: Model configuration.
        is_main: Whether this is the main process.

    Returns:
        SeasonalViT model.
    """
    if checkpoint_path is not None:
        if is_main:
            logger.info(f"Loading pre-trained weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model = SeasonalViT.from_pretrained(
            checkpoint["model_state_dict"],
            config,
        )
    else:
        if is_main:
            logger.info("Training from scratch (no pre-trained weights)")
        model = SeasonalViT(config)

    return model


def train_worker(
    rank: int,
    world_size: int,
    run_config: FinetuneRunConfig,
    val_dataset: Optional[InMemoryNDVIDataset],
    distributed_mode: Literal["spawn", "torchrun", "none"],
    store_path: Optional[str] = None,
) -> None:
    """Unified training worker for all execution modes.

    Args:
        rank: Process rank (0 for single-GPU).
        world_size: Total number of processes (1 for single-GPU).
        run_config: Fine-tuning configuration.
        val_dataset: Validation dataset (None if using prefetch mode, will load).
        distributed_mode: "spawn" for mp.spawn, "torchrun" for torchrun, "none" for single-GPU.
        store_path: FileStore path (required for spawn mode).
    """
    distributed = distributed_mode != "none"
    is_main = rank == 0

    # Setup distributed process group
    if distributed_mode == "spawn":
        setup_distributed_spawn(rank, world_size, store_path)
    elif distributed_mode == "torchrun":
        setup_distributed()

    # Setup device
    if distributed_mode == "spawn":
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    elif torch.cuda.is_available():
        if distributed:
            local_rank = get_local_rank()
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Auto-tune cuDNN convolution algorithms for fixed input size (256x256)
    torch.backends.cudnn.benchmark = True

    # Create output directory and log device info (main process only)
    if is_main:
        run_config.output_path.mkdir(parents=True, exist_ok=True)
        setup_file_logging(run_config.output_path)
        if distributed:
            logger.info(f"Training with {world_size} GPUs ({distributed_mode} mode)")
        elif device.type == "cuda":
            logger.info(f"Using CUDA: {torch.cuda.get_device_name()}")
        elif device.type == "mps":
            logger.info("Using MPS (Apple Silicon)")
        else:
            logger.info("Using CPU")

    # Create configs
    model_config, train_config = create_configs_from_run_config(run_config)
    if is_main:
        logger.info(f"Model config: {model_config}")
        logger.info(f"Training config: {train_config}")
        # Log target year/index info
        target_indices = train_config.get_target_indices()
        if train_config.predict_years:
            logger.info(
                f"Targeting years: {train_config.predict_years} -> "
                f"indices: {target_indices}"
            )
        else:
            logger.info(
                f"Targeting last {train_config.forecast_horizon} timesteps -> "
                f"indices: {target_indices}"
            )

    # Create data loaders using ShardPrefetchLoader
    train_loader, val_loader = create_prefetch_loaders(
        run_config, train_config, rank, world_size, distributed, is_main,
        val_dataset=val_dataset,
    )

    # Load model
    model = load_pretrained_model(run_config.pretrained_path, model_config, is_main)

    # Create trainer (freezes parameters if configured)
    trainer = FinetuningTrainer(
        model=model,
        config=train_config,
        device=device,
        output_dir=run_config.output_path,
        resume_from=run_config.resume_path,
        distributed=distributed,
        rank=rank,
    )

    if is_main:
        num_params = model.get_num_parameters()
        trainable_params = model.get_num_parameters(trainable_only=True)
        logger.info(f"Total parameters: {num_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

    # Train
    if is_main:
        logger.info("Starting fine-tuning...")
    history = trainer.fit(train_loader, val_loader)

    if is_main:
        logger.info("Fine-tuning complete!")
        logger.info(f"Final train loss: {history['train_loss'][-1]:.4f}")
        if history["val_loss"]:
            logger.info(f"Final val loss: {history['val_loss'][-1]:.4f}")
            logger.info(f"Final val MAE: {history['val_mae'][-1]:.4f}")

    # Cleanup
    if distributed:
        cleanup_distributed()


def _spawn_worker_wrapper(
    rank: int,
    world_size: int,
    run_config: FinetuneRunConfig,
    store_path: str,
    val_dataset: InMemoryNDVIDataset,
) -> None:
    """Thin wrapper for mp.spawn that calls train_worker."""
    train_worker(
        rank=rank,
        world_size=world_size,
        run_config=run_config,
        val_dataset=val_dataset,
        distributed_mode="spawn",
        store_path=store_path,
    )


def main_spawn(args: argparse.Namespace) -> None:
    """Main entry point for mp.spawn multi-GPU training."""
    world_size = args.num_gpus
    if world_size > torch.cuda.device_count():
        logger.warning(
            f"Requested {world_size} GPUs but only {torch.cuda.device_count()} available"
        )
        world_size = torch.cuda.device_count()

    # Create FileStore path
    store_dir = args.output_dir if args.output_dir else Path(tempfile.gettempdir())
    store_dir.mkdir(parents=True, exist_ok=True)
    store_path = str(store_dir / "ddp_store")

    # Clean up any existing store file
    store_file = Path(store_path)
    if store_file.exists():
        store_file.unlink()

    logger.info(f"=== Multi-GPU Fine-tuning with {world_size} GPUs (spawn mode) ===")

    # Create run config
    run_config = FinetuneRunConfig.from_args(args)

    # Load validation once, share across all workers
    logger.info("Loading validation data into shared memory...")
    val_shard_range = parse_shard_range(args.val_shard_range)
    t0 = time.perf_counter()
    val_dataset = InMemoryNDVIDataset(
        args.val_dir,
        shard_range=val_shard_range,
    )
    logger.info(f"[TIMING] InMemoryNDVIDataset(val): {time.perf_counter() - t0:.1f}s")
    t0 = time.perf_counter()
    val_dataset.share_memory_()
    logger.info(f"[TIMING] share_memory_(val): {time.perf_counter() - t0:.1f}s")
    logger.info(f"Validation data loaded: {len(val_dataset)} samples")

    logger.info(f"Spawning {world_size} GPU worker processes...")
    mp.spawn(
        _spawn_worker_wrapper,
        args=(world_size, run_config, store_path, val_dataset),
        nprocs=world_size,
        join=True,
    )


def main_standard(args: argparse.Namespace) -> None:
    """Main entry point for torchrun multi-GPU or single-GPU training."""
    distributed = args.multi_gpu
    distributed_mode: Literal["spawn", "torchrun", "none"] = "torchrun" if distributed else "none"

    if distributed_mode == "none":
        logger.info("=== Single-GPU Fine-tuning ===")
    else:
        logger.info("=== Multi-GPU Fine-tuning (torchrun mode) ===")

    # Create run config
    run_config = FinetuneRunConfig.from_args(args)

    # Run training (validation dataset will be loaded in worker)
    rank = int(os.environ.get("RANK", 0)) if distributed else 0
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    train_worker(
        rank=rank,
        world_size=world_size,
        run_config=run_config,
        val_dataset=None,
        distributed_mode=distributed_mode,
    )


def main():
    """Main fine-tuning entry point."""
    args = parse_args()

    if args.multi_gpu and args.spawn:
        main_spawn(args)
    else:
        main_standard(args)


if __name__ == "__main__":
    main()
