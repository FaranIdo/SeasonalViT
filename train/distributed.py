"""Distributed training utilities for multi-GPU support.

Provides helper functions for PyTorch DistributedDataParallel (DDP) training.
Uses NCCL backend for GPU-to-GPU communication.

Usage:
    # With torchrun launcher
    torchrun --nproc_per_node=2 scripts/pretrain.py --multi-gpu

    # In script
    from train.distributed import setup_distributed, cleanup_distributed, is_main_process

    setup_distributed()
    # ... training code ...
    cleanup_distributed()
"""

import os
from datetime import timedelta
from typing import Optional

import torch
import torch.distributed as dist


def setup_distributed(backend: str = "nccl") -> None:
    """Initialize the distributed process group.

    Should be called at the start of training when using torchrun.
    torchrun sets RANK, LOCAL_RANK, and WORLD_SIZE environment variables.

    Args:
        backend: Communication backend. Use "nccl" for GPU training,
                 "gloo" for CPU training.
    """
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)


def setup_distributed_spawn(rank: int, world_size: int, store_path: str) -> None:
    """Initialize distributed training for mp.spawn using FileStore.

    Uses FileStore instead of TCPStore to avoid the libuv issue on Windows.
    Sets environment variables for compatibility with torchrun-style code.

    Args:
        rank: Process rank (0 to world_size - 1).
        world_size: Total number of processes.
        store_path: Path for FileStore synchronization file.
    """
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    store = dist.FileStore(store_path, world_size)

    # Use gloo backend (NCCL not supported on Windows). 2h timeout (vs gloo's
    # 30min default) absorbs slow shard loads on one rank without aborting a
    # multi-hour run; saw a real 30min trip at epoch 15 of a multi-hour fine-tuning run
    # when one rank's prefetch stalled mid-step.
    dist.init_process_group(
        backend="gloo",
        store=store,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(hours=2),
    )


def cleanup_distributed() -> None:
    """Clean up the distributed process group.

    Should be called at the end of training.
    """
    if dist.is_initialized():
        dist.destroy_process_group()


def is_distributed() -> bool:
    """Check if running in distributed mode.

    Returns:
        True if distributed training is initialized.
    """
    return dist.is_initialized()


def is_main_process() -> bool:
    """Check if this is the main process (rank 0).

    Used for logging, checkpointing, and other operations that should
    only happen once across all processes.

    Returns:
        True if this is rank 0 or if not in distributed mode.
    """
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_rank() -> int:
    """Get the global rank of the current process.

    Returns:
        Process rank (0 to world_size - 1), or 0 if not distributed.
    """
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_local_rank() -> int:
    """Get the local rank of the current process on this node.

    Local rank is used for device assignment (cuda:local_rank).

    Returns:
        Local rank from LOCAL_RANK env var, or 0 if not set.
    """
    return int(os.environ.get("LOCAL_RANK", 0))


def get_world_size() -> int:
    """Get the total number of processes in the distributed group.

    Returns:
        Number of processes, or 1 if not distributed.
    """
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def barrier() -> None:
    """Synchronize all processes.

    Blocks until all processes reach this point. Useful for ensuring
    all processes have completed an operation before proceeding.

    No-op if not in distributed mode.
    """
    if dist.is_initialized():
        dist.barrier()


def reduce_tensor(tensor: torch.Tensor, op: dist.ReduceOp = dist.ReduceOp.SUM) -> torch.Tensor:
    """Reduce a tensor across all processes.

    Args:
        tensor: Tensor to reduce. Must be on GPU for NCCL backend.
        op: Reduction operation (SUM, PRODUCT, MIN, MAX, etc.).

    Returns:
        Reduced tensor (in-place modification also occurs).
    """
    if not dist.is_initialized():
        return tensor
    dist.all_reduce(tensor, op=op)
    return tensor


def all_gather_tensors(tensor: torch.Tensor) -> list:
    """Gather tensors from all processes.

    Args:
        tensor: Local tensor to gather.

    Returns:
        List of tensors from all processes.
    """
    if not dist.is_initialized():
        return [tensor]

    world_size = dist.get_world_size()
    tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensors, tensor)
    return tensors
