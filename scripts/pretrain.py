#!/usr/bin/env python3
"""Pre-training entry point for SeasonalViT.

Usage:
    # Single GPU
    python scripts/pretrain.py --train-dir DATA/train --val-dir DATA/val --output-dir OUTPUT

    # Multi-GPU with torchrun (Linux/macOS)
    torchrun --nproc_per_node=2 scripts/pretrain.py --train-dir DATA/train --val-dir DATA/val --multi-gpu

    # Multi-GPU with mp.spawn (Windows - avoids libuv issue)
    python scripts/pretrain.py --train-dir DATA/train --val-dir DATA/val --multi-gpu --spawn --num-gpus 2
"""

import argparse
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from datasets.sharded_dataset import InMemoryNDVIDataset, create_dataloader
from datasets.prefetch_dataset import ShardPrefetchLoader
from model.config import SeasonalViTConfig, PretrainingConfig
from model.pretrain_model import PretrainingModel
from train.distributed import (
    setup_distributed,
    setup_distributed_spawn,
    cleanup_distributed,
    is_main_process,
    get_local_rank,
    get_rank,
)
from train.pretrain import PretrainingTrainer

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
class TrainRunConfig:
    """Picklable configuration for training run.

    Encapsulates all arguments needed to configure a training run,
    avoiding the need to reconstruct argparse.Namespace in subprocesses.
    """

    train_dir: str
    val_dir: str
    output_dir: str
    resume: Optional[str]
    epochs: Optional[int]
    batch_size: Optional[int]
    lr: Optional[float]
    no_pretrained: bool
    limit: Optional[int]
    train_shard_range: Optional[str]
    val_shard_range: Optional[str]
    prefetch: bool
    normalize_target: bool = True
    temporal_mask_ratio: Optional[float] = None
    temporal_subset_years: Optional[int] = None
    val_batch_size: Optional[int] = None
    temporal_pathway: Optional[str] = None

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "TrainRunConfig":
        """Create config from parsed arguments."""
        return cls(
            train_dir=str(args.train_dir),
            val_dir=str(args.val_dir),
            output_dir=str(args.output_dir) if args.output_dir else "outputs/pretrain",
            resume=str(args.resume) if args.resume else None,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            no_pretrained=args.no_pretrained,
            limit=args.limit,
            train_shard_range=args.train_shard_range,
            val_shard_range=args.val_shard_range,
            prefetch=args.prefetch,
            normalize_target=not args.no_normalize_target,
            temporal_mask_ratio=args.temporal_mask_ratio,
            temporal_subset_years=args.temporal_subset_years,
            val_batch_size=args.val_batch_size,
            temporal_pathway=getattr(args, "temporal_pathway", None),
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
        description="Pre-train SeasonalViT with MAE-style reconstruction"
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
        default=Path("outputs/pretrain"),
        help="Directory for checkpoints and logs",
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

    # Model parameters
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Don't use pretrained Swin-T weights",
    )

    # Loss parameters
    parser.add_argument(
        "--no-normalize-target",
        action="store_true",
        help="Disable MAE-style per-patch target normalization (enabled by default)",
    )
    parser.add_argument(
        "--temporal-mask-ratio",
        type=float,
        help="Override temporal mask ratio (default: 0.3)",
    )
    parser.add_argument(
        "--temporal-subset-years",
        type=int,
        help="Sample K complete years per forward pass instead of all 35. "
             "Reduces memory ~3x with K=12, enabling larger batches. "
             "Default: 12 (from config). Set to 0 to disable.",
    )
    parser.add_argument(
        "--temporal-pathway",
        type=str,
        choices=["both", "cross_season", "cross_year"],
        default=None,
        help="Temporal pathway (ablations). Default preserves dual-scale fusion.",
    )
    parser.add_argument(
        "--val-batch-size",
        type=int,
        help="Validation batch size. Default: auto-derived (batch_size//4 when "
             "temporal subsampling is active, else batch_size).",
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

    # Data loading strategy
    parser.add_argument(
        "--prefetch",
        action="store_true",
        help="Use shard prefetching (memory-efficient, loads 2 shards at a time)",
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


def load_datasets_into_memory(
    args: argparse.Namespace,
) -> Tuple[InMemoryNDVIDataset, InMemoryNDVIDataset]:
    """Load train and validation datasets into memory.

    Args:
        args: Parsed command line arguments.

    Returns:
        Tuple of (train_dataset, val_dataset) with shared memory enabled.
    """
    train_shard_range = parse_shard_range(args.train_shard_range)
    val_shard_range = parse_shard_range(args.val_shard_range)

    logger.info("Loading training data...")
    train_dataset = InMemoryNDVIDataset(
        args.train_dir,
        max_samples=args.limit,
        shard_range=train_shard_range,
    )
    logger.info(f"Training data loaded: {len(train_dataset)} samples")

    logger.info("Loading validation data...")
    val_dataset = InMemoryNDVIDataset(
        args.val_dir,
        max_samples=args.limit // 10 if args.limit else None,
        shard_range=val_shard_range,
    )
    logger.info(f"Validation data loaded: {len(val_dataset)} samples")

    logger.info("Moving data to shared memory...")
    train_dataset.share_memory_()
    val_dataset.share_memory_()

    return train_dataset, val_dataset


def create_prefetch_loaders(
    run_config: TrainRunConfig,
    train_config: PretrainingConfig,
    rank: int,
    world_size: int,
    distributed: bool,
    is_main: bool,
    val_dataset: Optional[InMemoryNDVIDataset] = None,
) -> Tuple[ShardPrefetchLoader, DataLoader]:
    """Create data loaders for prefetch mode.

    Args:
        run_config: Training run configuration.
        train_config: Pre-training configuration.
        rank: Process rank.
        world_size: Total number of processes.
        distributed: Whether distributed training is enabled.
        is_main: Whether this is the main process.

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
    # If val_dataset was passed in, it's already in shared memory

    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None
    val_loader = create_dataloader(
        val_dataset,
        batch_size=train_config.effective_val_batch_size,
        shuffle=False,
        num_workers=train_config.val_num_workers,
        pin_memory=True,
        prefetch_factor=2,
        sampler=val_sampler,
    )

    if is_main:
        logger.info(f"Val samples: {len(val_dataset)}")

    return train_loader, val_loader


def create_standard_loaders(
    train_dataset: InMemoryNDVIDataset,
    val_dataset: InMemoryNDVIDataset,
    train_config: PretrainingConfig,
    distributed: bool,
    is_main: bool,
) -> Tuple[DataLoader, DataLoader]:
    """Create data loaders for standard (in-memory) mode.

    Args:
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        train_config: Pre-training configuration.
        distributed: Whether distributed training is enabled.
        is_main: Whether this is the main process.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    if is_main:
        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Val samples: {len(val_dataset)}")

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None

    train_loader = create_dataloader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=(not distributed),
        num_workers=train_config.train_num_workers,
        pin_memory=True,
        prefetch_factor=2,
        sampler=train_sampler,
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=train_config.effective_val_batch_size,
        shuffle=False,
        num_workers=train_config.val_num_workers,
        pin_memory=True,
        prefetch_factor=2,
        sampler=val_sampler,
    )

    return train_loader, val_loader


def create_configs_from_run_config(
    run_config: TrainRunConfig,
) -> Tuple[SeasonalViTConfig, PretrainingConfig]:
    """Create model and training configs from TrainRunConfig."""
    model_kwargs: dict = {"swin_pretrained": not run_config.no_pretrained}
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
    if run_config.temporal_mask_ratio is not None:
        train_kwargs["temporal_mask_ratio"] = run_config.temporal_mask_ratio
    if run_config.temporal_subset_years is not None:
        train_kwargs["temporal_subset_years"] = run_config.temporal_subset_years
    if run_config.val_batch_size is not None:
        train_kwargs["val_batch_size"] = run_config.val_batch_size

    train_kwargs["normalize_target"] = run_config.normalize_target

    train_config = PretrainingConfig(**train_kwargs)

    return model_config, train_config


def train_worker(
    rank: int,
    world_size: int,
    run_config: TrainRunConfig,
    train_dataset: Optional[InMemoryNDVIDataset],
    val_dataset: Optional[InMemoryNDVIDataset],
    distributed_mode: Literal["spawn", "torchrun", "none"],
    store_path: Optional[str] = None,
) -> None:
    """Unified training worker for all execution modes.

    Args:
        rank: Process rank (0 for single-GPU).
        world_size: Total number of processes (1 for single-GPU).
        run_config: Training configuration.
        train_dataset: Training dataset (None if using prefetch mode).
        val_dataset: Validation dataset (None if using prefetch mode).
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
        if run_config.prefetch:
            logger.info("Using shard prefetch mode (memory-efficient)")

    # Create configs
    model_config, train_config = create_configs_from_run_config(run_config)
    if is_main:
        logger.info(f"Model config: {model_config}")
        logger.info(f"Training config: {train_config}")
        if train_config.effective_val_batch_size != train_config.batch_size:
            logger.info(
                f"Val batch size: {train_config.effective_val_batch_size} "
                f"(train: {train_config.batch_size})"
            )

    # Create data loaders based on mode
    if run_config.prefetch:
        train_loader, val_loader = create_prefetch_loaders(
            run_config, train_config, rank, world_size, distributed, is_main,
            val_dataset=val_dataset,
        )
    else:
        train_loader, val_loader = create_standard_loaders(
            train_dataset, val_dataset, train_config, distributed, is_main
        )

    # Create model
    model = PretrainingModel(model_config)
    if is_main:
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {num_params:,}")

    # Create trainer
    trainer = PretrainingTrainer(
        model=model,
        config=train_config,
        device=device,
        output_dir=run_config.output_path,
        resume_from=run_config.resume_path,
        distributed=distributed,
        rank=rank,
    )

    # Train
    if is_main:
        logger.info("Starting pre-training...")
    history = trainer.fit(train_loader, val_loader)

    if is_main:
        logger.info("Pre-training complete!")
        logger.info(f"Final train loss: {history['train_loss'][-1]:.4f}")
        if history["val_loss"]:
            logger.info(f"Final val loss: {history['val_loss'][-1]:.4f}")

    # Cleanup
    if distributed:
        cleanup_distributed()


def _spawn_worker_wrapper(
    rank: int,
    world_size: int,
    run_config: TrainRunConfig,
    store_path: str,
    train_dataset: InMemoryNDVIDataset,
    val_dataset: InMemoryNDVIDataset,
) -> None:
    """Thin wrapper for mp.spawn that calls train_worker."""
    train_worker(
        rank=rank,
        world_size=world_size,
        run_config=run_config,
        train_dataset=train_dataset,
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

    logger.info(f"=== Multi-GPU Training with {world_size} GPUs (spawn mode) ===")

    # Create run config
    run_config = TrainRunConfig.from_args(args)

    if args.prefetch:
        logger.info("Using prefetch mode - workers will load shards on demand")
        train_dataset = None
        # Load validation once, share across all workers
        logger.info("Loading validation data into shared memory...")
        val_shard_range = parse_shard_range(args.val_shard_range)
        val_dataset = InMemoryNDVIDataset(
            args.val_dir,
            shard_range=val_shard_range,
        )
        val_dataset.share_memory_()
        logger.info(f"Validation data loaded: {len(val_dataset)} samples")
    else:
        train_dataset, val_dataset = load_datasets_into_memory(args)

    logger.info(f"Spawning {world_size} GPU worker processes...")
    mp.spawn(
        _spawn_worker_wrapper,
        args=(world_size, run_config, store_path, train_dataset, val_dataset),
        nprocs=world_size,
        join=True,
    )


def main_standard(args: argparse.Namespace) -> None:
    """Main entry point for torchrun multi-GPU or single-GPU training."""
    distributed = args.multi_gpu
    distributed_mode: Literal["spawn", "torchrun", "none"] = "torchrun" if distributed else "none"

    if distributed_mode == "none":
        logger.info("=== Single-GPU Training ===")
    else:
        logger.info("=== Multi-GPU Training (torchrun mode) ===")

    # Create run config
    run_config = TrainRunConfig.from_args(args)

    if args.prefetch:
        logger.info("Using prefetch mode - shards will be loaded on demand")
        train_dataset, val_dataset = None, None
    else:
        train_dataset, val_dataset = load_datasets_into_memory(args)

    # Run training
    rank = get_rank() if distributed else 0
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    train_worker(
        rank=rank,
        world_size=world_size,
        run_config=run_config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        distributed_mode=distributed_mode,
    )


def main():
    """Main pre-training entry point."""
    args = parse_args()

    if args.multi_gpu and args.spawn:
        main_spawn(args)
    else:
        main_standard(args)


if __name__ == "__main__":
    main()
