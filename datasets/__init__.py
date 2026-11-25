"""Dataset utilities for NDVI time series data.

This module provides:
- InMemoryNDVIDataset: Fast in-memory dataset for pre-processed shards
- create_dataloader: DataLoader factory for in-memory datasets
- Transforms: Temporal and spatial augmentations
- Splits: Geographic block splitting for train/val/test
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datasets.shard_io import load_shard
    from datasets.sharded_dataset import (
        InMemoryNDVIDataset,
        create_dataloader,
    )
    from datasets.prefetch_dataset import (
        ShardPrefetchLoader,
        SingleShardDataset,
    )
    from datasets.transforms import (
        Compose,
        get_train_transforms,
        get_val_transforms,
        Normalize,
        RandomHorizontalFlip,
        RandomRotation90,
        RandomVerticalFlip,
        SeasonalJitter,
        TemporalDropout,
        TemporalNoise,
    )
    from datasets.splits import (
        create_geographic_splits,
        get_split_statistics,
        load_locations_json,
    )

__all__ = [
    # Shard I/O
    "load_shard",
    # Dataset
    "InMemoryNDVIDataset",
    "create_dataloader",
    # Prefetch loader
    "ShardPrefetchLoader",
    "SingleShardDataset",
    # Transforms
    "Compose",
    "get_train_transforms",
    "get_val_transforms",
    "Normalize",
    "RandomHorizontalFlip",
    "RandomRotation90",
    "RandomVerticalFlip",
    "SeasonalJitter",
    "TemporalDropout",
    "TemporalNoise",
    # Splits
    "create_geographic_splits",
    "get_split_statistics",
    "load_locations_json",
]


def __getattr__(name: str):
    """Lazy import for module components."""
    if name == "load_shard":
        from datasets import shard_io
        return shard_io.load_shard

    if name in ("InMemoryNDVIDataset", "create_dataloader"):
        from datasets import sharded_dataset
        return getattr(sharded_dataset, name)

    if name in ("ShardPrefetchLoader", "SingleShardDataset"):
        from datasets import prefetch_dataset
        return getattr(prefetch_dataset, name)

    if name in (
        "Compose", "get_train_transforms", "get_val_transforms",
        "Normalize", "RandomHorizontalFlip", "RandomRotation90",
        "RandomVerticalFlip", "SeasonalJitter", "TemporalDropout",
        "TemporalNoise",
    ):
        from datasets import transforms
        return getattr(transforms, name)

    if name in ("create_geographic_splits", "get_split_statistics",
                "load_locations_json"):
        from datasets import splits
        return getattr(splits, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
