"""Shard prefetching dataset for memory-efficient training.

Uses 2 persistent DataLoaders with shared memory buffers. While yielding from
buffer A, prefetch next shard into buffer B (and vice versa). Shards are loaded
from SafeTensors format.
"""

import gc
import json
import logging
import time
import random
import sys
import threading
from collections.abc import Iterator
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from datasets.shard_io import discover_shards, load_shard

logger = logging.getLogger(__name__)


def _shutdown_loader(loader: DataLoader) -> None:
    """Explicitly shutdown DataLoader workers."""
    if hasattr(loader, "_iterator") and loader._iterator is not None:
        if hasattr(loader._iterator, "_shutdown_workers"):
            loader._iterator._shutdown_workers()


def _log_memory(context: str) -> None:
    """Log memory stats for monitoring. Shows detailed metrics on Windows."""
    if sys.platform == "win32":
        try:
            import psutil
            proc = psutil.Process()
            mem = proc.memory_info()
            sys_mem = psutil.virtual_memory()
            logger.info(
                f"[Memory:{context}] "
                f"RSS={mem.rss / 1e9:.2f}GB, "
                f"VMS={mem.vms / 1e9:.2f}GB, "
                f"private={mem.private / 1e9:.2f}GB, "
                f"handles={proc.num_handles()} | "
                f"sys_avail={sys_mem.available / 1e9:.0f}GB, "
                f"sys_commit={getattr(sys_mem, 'commit', 0) / 1e9:.0f}GB"
            )
            return
        except ImportError:
            pass
    logger.info(f"[Memory:{context}]")


class SingleShardDataset(Dataset):
    """Dataset with shared memory buffers for persistent DataLoader workers.

    Pre-allocates tensors in shared memory, then updates them via copy for
    each new shard. Workers see the updated data since storage is shared.
    """

    def __init__(self, max_samples: int) -> None:
        """Initialize with pre-allocated shared memory buffers.

        Args:
            max_samples: Maximum samples to allocate space for.
        """
        self._max_samples = max_samples
        self._num_samples = 0

        # Pre-allocate shared memory buffers (use empty, not zeros - we overwrite anyway)
        self.images = torch.empty(
            max_samples, 140, 256, 256, dtype=torch.float16
        ).share_memory_()
        self.nodata_masks = torch.empty(
            max_samples, 140, 256, 256, dtype=torch.bool
        ).share_memory_()
        self.lats = torch.empty(max_samples, dtype=torch.float32).share_memory_()
        self.lons = torch.empty(max_samples, dtype=torch.float32).share_memory_()

        # Pre-compute season and year indices (shared across all samples)
        num_timesteps = 140
        num_seasons = 4
        self._season_ids = torch.tensor(
            [t % num_seasons for t in range(num_timesteps)], dtype=torch.long
        ).share_memory_()
        self._year_indices = torch.tensor(
            [t // num_seasons for t in range(num_timesteps)], dtype=torch.long
        ).share_memory_()

        logger.info(
            f"Allocated shared memory buffer for {max_samples} samples "
            f"(~{max_samples * 140 * 256 * 256 * 2 / 1e9:.1f}GB)"
        )

    def set_length(self, n: int) -> None:
        """Set dataset length without copying data (for RandomSampler compatibility)."""
        self._num_samples = n

    def update_data(self, shard_data: dict[str, torch.Tensor]) -> None:
        """Copy new shard data into shared memory buffers.

        Args:
            shard_data: Dict with images, nodata_masks, lats, lons, files.
        """
        n = len(shard_data["files"])
        if n > self._max_samples:
            raise ValueError(
                f"Shard has {n} samples but buffer only holds {self._max_samples}"
            )
        self.images[:n].copy_(shard_data["images"])
        self.nodata_masks[:n].copy_(shard_data["nodata_masks"])
        self.lats[:n].copy_(shard_data["lats"])
        self.lons[:n].copy_(shard_data["lons"])
        self._num_samples = n

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "images": self.images[idx],
            "season_ids": self._season_ids,
            "year_indices": self._year_indices,
            "lat": self.lats[idx],
            "lon": self.lons[idx],
            "nodata_mask": self.nodata_masks[idx],
        }


class ShardPrefetchLoader:
    """Iterates through shards with background prefetching using persistent workers.

    Uses 2 DataLoaders (A and B) with persistent workers and shared memory.
    Alternates between them: while yielding from A, prefetch into B.

    Memory Profile:
        - 2 shared memory buffers (~26GB each for 1000 samples)
        - ~52GB peak for double buffering
        - Memory stays constant (no leak from worker spawn/terminate)

    Timeline:
        [Init loaders A & B with shared memory]
                    │
                    ▼
        [Load shard 0 → buffer A]  ← blocking
                    │
                    ▼
        [Yield from loader A] ──────────────────►
        [Prefetch shard 1 → buffer B] ────►      │
                                   │             │
                                   ▼             ▼
                           buffer B ready   loader A done
                                                 │
                                                 ▼
                                [Yield from loader B] ──────►
                                [Prefetch shard 2 → buffer A] ──►
                                                              ...
    """

    def __init__(
        self,
        shard_dir: str | Path,
        batch_size: int,
        shuffle_shards: bool = True,
        shuffle_within_shard: bool = True,
        num_workers: int = 8,
        prefetch_factor: int = 2,
        pin_memory: bool = True,
        drop_last: bool = True,
        rank: int = 0,
        world_size: int = 1,
        shard_range: tuple[int, int] | None = None,
        samples_per_shard: int | None = None,
    ) -> None:
        """Initialize the shard prefetch loader.

        Args:
            shard_dir: Directory containing shard_*.safetensors files.
            batch_size: Batch size for DataLoader.
            shuffle_shards: Whether to shuffle shard order each epoch.
            shuffle_within_shard: Whether to shuffle samples within each shard.
            num_workers: Number of DataLoader workers.
            prefetch_factor: Batches to prefetch per worker.
            pin_memory: Whether to pin memory for faster GPU transfer.
            drop_last: Whether to drop the last incomplete batch.
            rank: Process rank for distributed training.
            world_size: Total number of processes for distributed training.
            shard_range: Optional (start, end) to limit which shards to load.
            samples_per_shard: Expected samples per shard (for __len__ estimate).
                If None, reads from metadata.json or defaults to 1000.
        """
        self.shard_dir = Path(shard_dir)
        self.batch_size = batch_size
        self.shuffle_shards = shuffle_shards
        self.shuffle_within_shard = shuffle_within_shard
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.rank = rank
        self.world_size = world_size
        self.samples_per_shard = self._detect_samples_per_shard(samples_per_shard)

        # Discover shard files (.safetensors preferred, falls back to .pt)
        all_shard_paths = discover_shards(self.shard_dir)
        if not all_shard_paths:
            raise ValueError(
                f"No shard_*.safetensors or shard_*.pt files found in {self.shard_dir}"
            )

        # Apply shard range filter
        if shard_range is not None:
            start, end = shard_range
            self.shard_paths = all_shard_paths[start:end]
            logger.info(
                f"Using shards {start}:{end} of {len(all_shard_paths)} "
                f"({len(self.shard_paths)} shards)"
            )
        else:
            self.shard_paths = all_shard_paths
            logger.info(f"Found {len(self.shard_paths)} shards in {self.shard_dir}")

        # Distribute shards for DDP
        if world_size > 1:
            self.shard_paths = [
                self.shard_paths[i]
                for i in range(self.rank, len(self.shard_paths), self.world_size)
            ]
            logger.info(f"Rank {rank}: assigned {len(self.shard_paths)} shards")

        # State
        self._shard_order: list[int] = []
        self._prefetch_thread: threading.Thread | None = None
        self._prefetch_error: Exception | None = None

        # Persistent loaders (lazily initialized per-buffer)
        self._max_samples = int(self.samples_per_shard * 1.1)
        self._datasets: list[SingleShardDataset | None] = [None, None]
        self._loaders: list[DataLoader | None] = [None, None]

    def _detect_samples_per_shard(self, explicit_value: int | None) -> int:
        """Determine samples_per_shard from explicit value, metadata, or default.

        Priority: explicit value > metadata.json > default (1000)
        """
        if explicit_value is not None:
            return explicit_value

        metadata_path = self.shard_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            if "samples_per_shard" in metadata:
                value = metadata["samples_per_shard"]
                logger.info(f"Read samples_per_shard={value} from metadata.json")
                return value

        return 1000

    def _ensure_buffer(self, idx: int) -> None:
        """Lazily initialize a single buffer + DataLoader.

        Called on-demand so buffer B is only allocated when the first prefetch
        starts, not upfront. This cuts startup time in half (~20GB less to allocate
        before training begins).
        """
        if self._loaders[idx] is not None:
            return

        logger.info(f"Initializing buffer {idx} (max_samples={self._max_samples})")
        t0 = time.perf_counter()
        dataset = SingleShardDataset(self._max_samples)
        logger.info(f"[TIMING] SingleShardDataset({idx}): {time.perf_counter() - t0:.1f}s")
        dataset.set_length(self.samples_per_shard)
        self._datasets[idx] = dataset

        use_persistent = self.num_workers > 0
        t0 = time.perf_counter()
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_within_shard,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=use_persistent,
            drop_last=self.drop_last,
        )
        logger.info(f"[TIMING] DataLoader({idx}): {time.perf_counter() - t0:.1f}s")
        self._loaders[idx] = loader

        logger.info(f"Buffer {idx} initialized")
        _log_memory(f"buffer_{idx}_init")

    def reset(self, epoch: int = 0) -> None:
        """Reset for a new epoch. Shuffles shard order if enabled."""
        self._shard_order = list(range(len(self.shard_paths)))
        if self.shuffle_shards:
            rng = random.Random(epoch + self.rank * 1000)
            rng.shuffle(self._shard_order)

        # Wait for any pending prefetch
        if self._prefetch_thread is not None:
            self._prefetch_thread.join()
            self._prefetch_thread = None

        self._prefetch_error = None
        _log_memory("epoch_reset")

    def _load_shard_to_buffer(self, path: Path, target_idx: int) -> None:
        """Load shard directly into target buffer.

        SafeTensors has no memory leak, so no subprocess isolation needed.
        """
        dataset = self._datasets[target_idx]
        t0 = time.perf_counter()
        shard_data = load_shard(path)
        n = len(shard_data["files"])
        if n > dataset._max_samples:
            raise ValueError(
                f"Shard has {n} samples but buffer only holds {dataset._max_samples}"
            )
        dataset.images[:n].copy_(shard_data["images"])
        dataset.nodata_masks[:n].copy_(shard_data["nodata_masks"])
        dataset.lats[:n].copy_(shard_data["lats"])
        dataset.lons[:n].copy_(shard_data["lons"])
        dataset.set_length(n)
        del shard_data
        logger.info(f"[TIMING] _load_shard_to_buffer({path.name}): {time.perf_counter() - t0:.1f}s")

    def _start_prefetch(self, path: Path, target_idx: int) -> None:
        """Start background thread to prefetch shard into target buffer."""
        # Lazily allocate buffer B on first prefetch (while GPU trains on shard 0)
        self._ensure_buffer(target_idx)

        def worker():
            try:
                self._load_shard_to_buffer(path, target_idx)
                gc.collect()
                logger.info(f"Prefetch done: {path.name} -> buffer {target_idx}")
            except Exception as e:
                self._prefetch_error = e

        self._prefetch_thread = threading.Thread(target=worker)
        self._prefetch_thread.start()

    def _wait_prefetch(self) -> None:
        """Wait for prefetch to complete."""
        if self._prefetch_thread is None:
            return

        if self._prefetch_thread.is_alive():
            import time
            start = time.perf_counter()
            logger.warning("GPU waiting for prefetch...")
            self._prefetch_thread.join()
            elapsed = time.perf_counter() - start
            logger.warning(f"Prefetch wait done ({elapsed:.1f}s stall)")
        else:
            self._prefetch_thread.join()

        self._prefetch_thread = None

        if self._prefetch_error is not None:
            error = self._prefetch_error
            self._prefetch_error = None
            raise error

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """Iterate through all shards, yielding batches."""
        if not self._shard_order:
            self.reset()

        num_shards = len(self._shard_order)

        for shard_idx, order_idx in enumerate(self._shard_order):
            shard_path = self.shard_paths[order_idx]
            current_idx = shard_idx % 2  # Alternate: 0, 1, 0, 1, ...

            if shard_idx == 0:
                # First shard: load via subprocess (same as prefetch path)
                # to avoid 28GB allocation in main process that fragments heap
                logger.info(f"Loading first shard: {shard_path.name}")
                self._ensure_buffer(current_idx)
                self._load_shard_to_buffer(shard_path, current_idx)
            else:
                # Wait for prefetch (data already in buffer)
                logger.info(f"Swapping to buffer {current_idx}: {shard_path.name}")
                self._wait_prefetch()

            # Start prefetching next shard into OTHER buffer
            if shard_idx + 1 < num_shards:
                next_idx = (shard_idx + 1) % 2
                next_path = self.shard_paths[self._shard_order[shard_idx + 1]]
                logger.info(f"Prefetching: {next_path.name} -> buffer {next_idx}")
                self._start_prefetch(next_path, next_idx)

            # Yield from current loader (workers see shared memory data)
            yield from self._loaders[current_idx]

            # Force garbage collection to reclaim memory from:
            # - Pinned memory copies from DataLoader
            # - Any lingering tensor references
            gc.collect()

            _log_memory(f"shard_{shard_idx}_done (gc)")

    def __len__(self) -> int:
        """Return estimated total batches across all shards."""
        total_samples = len(self.shard_paths) * self.samples_per_shard
        if self.drop_last:
            return total_samples // self.batch_size
        return (total_samples + self.batch_size - 1) // self.batch_size

    @property
    def total_samples(self) -> int:
        """Estimated total samples across all shards."""
        return len(self.shard_paths) * self.samples_per_shard

    @property
    def sampler(self) -> "_MockSampler":
        """Mock sampler for DDP compatibility (trainer calls sampler.set_epoch())."""
        return _MockSampler(self)

    def shutdown(self) -> None:
        """Shutdown persistent loaders and free shared memory."""
        for i, loader in enumerate(self._loaders):
            if loader is not None:
                _shutdown_loader(loader)
                self._loaders[i] = None
                self._datasets[i] = None
        gc.collect()
        _log_memory("shutdown")


class _MockSampler:
    """Mock sampler for DDP compatibility."""

    def __init__(self, loader: ShardPrefetchLoader) -> None:
        self._loader = loader

    def set_epoch(self, epoch: int) -> None:
        self._loader.reset(epoch=epoch)
