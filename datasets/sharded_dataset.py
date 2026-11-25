"""Fast PyTorch Dataset for pre-processed NDVI shards.

Loads pre-processed float16 shards created by convert_to_shards.py.
All data is loaded into RAM for maximum training speed.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from datasets.shard_io import discover_shards, load_shard

logger = logging.getLogger(__name__)


class InMemoryNDVIDataset(Dataset):
    """Ultra-fast dataset that holds all data as contiguous tensors in RAM.

    This is the fastest possible option when data fits in memory.
    All data is stored as contiguous tensors for cache-friendly access.

    Attributes:
        shard_dir: Directory containing shard files.
        num_samples: Total number of samples loaded.
        images: (N, 140, 256, 256) float16 tensor of NDVI images.
        nodata_masks: (N, 140, 256, 256) bool tensor of nodata masks.
        lats: (N,) float32 tensor of latitudes.
        lons: (N,) float32 tensor of longitudes.
        files: List of original filenames.
    """

    def __init__(
        self,
        shard_dir: Union[str, Path],
        max_samples: Optional[int] = None,
        shard_range: Optional[Tuple[int, int]] = None,
        num_workers: int = 8,
    ) -> None:
        """Initialize and load all data into contiguous tensors.

        Args:
            shard_dir: Directory containing shard_*.safetensors files.
            max_samples: Maximum number of samples to load (for memory limits).
            shard_range: Tuple of (start_shard, end_shard) to load only specific shards.
                         Useful for debugging or chunked training.
            num_workers: Number of parallel workers for loading shards (default: 8).

        Example:
            # Load all shards from directory
            dataset = InMemoryNDVIDataset("/path/to/train_shards")

            # Load only first 50,000 samples
            dataset = InMemoryNDVIDataset("/path/to/train_shards", max_samples=50000)

            # Load shards 0-100 (useful for debugging)
            dataset = InMemoryNDVIDataset("/path/to/train_shards", shard_range=(0, 100))
        """
        self.max_samples = max_samples
        self.shard_range = shard_range
        self.num_workers = num_workers
        self.shard_dir = Path(shard_dir)

        # Auto-discover shards (.safetensors preferred, falls back to .pt)
        shard_files = discover_shards(self.shard_dir)
        if not shard_files:
            raise ValueError(
                f"No shard_*.safetensors or shard_*.pt files found in {self.shard_dir}"
            )

        logger.info(f"Found {len(shard_files)} shards in {self.shard_dir}")

        # Determine which shards to load
        shards_to_load = []
        for shard_idx, shard_path in enumerate(shard_files):
            # Skip shards outside of range
            if shard_range is not None:
                if shard_idx < shard_range[0]:
                    continue
                if shard_idx >= shard_range[1]:
                    break
            shards_to_load.append(shard_path)

        logger.info(f"Loading {len(shards_to_load)} shards (max_samples={max_samples})...")

        # Load shards in parallel
        t0 = time.perf_counter()
        loaded_shards = self._load_shards_parallel(shards_to_load, num_workers)
        logger.info(f"[TIMING] _load_shards_parallel: {time.perf_counter() - t0:.1f}s")

        # Count total samples (respecting max_samples limit)
        total_samples = 0
        for shard in loaded_shards:
            n = len(shard["files"])
            if max_samples and total_samples + n > max_samples:
                n = max_samples - total_samples
            total_samples += n
            if max_samples and total_samples >= max_samples:
                break

        # Pre-allocate contiguous tensors (empty avoids zeroing pages, but
        # the OS still page-faults on first write — for OOD-s at 756 GB this
        # alone takes ~30s-2min). Print before each step so the user sees
        # progress instead of an opaque hang.
        images_gb = total_samples * 140 * 256 * 256 * 2 / 1e9
        masks_gb = total_samples * 140 * 256 * 256 / 1e9
        logger.info(
            f"Allocating memory for {total_samples} samples "
            f"(images ~{images_gb:.1f} GB + nodata_masks ~{masks_gb:.1f} GB)..."
        )
        t0 = time.perf_counter()
        self.images = torch.empty(total_samples, 140, 256, 256, dtype=torch.float16)
        logger.info(f"[TIMING] torch.empty(images, ~{images_gb:.1f} GB): {time.perf_counter() - t0:.1f}s")
        t0 = time.perf_counter()
        self.nodata_masks = torch.empty(total_samples, 140, 256, 256, dtype=torch.bool)
        logger.info(f"[TIMING] torch.empty(nodata_masks, ~{masks_gb:.1f} GB): {time.perf_counter() - t0:.1f}s")
        self.lats = torch.empty(total_samples, dtype=torch.float32)
        self.lons = torch.empty(total_samples, dtype=torch.float32)

        # Copy shards directly into pre-allocated tensors (bulk memcpy).
        # tqdm shows per-shard progress + throughput in GB/s so the user
        # can tell whether DRAM bandwidth or page-faulting is the limit.
        t0 = time.perf_counter()
        offset = 0
        bytes_per_sample = 140 * 256 * 256 * 3  # fp16 (2B) + bool (1B)
        self.files: List[str] = []
        with tqdm(
            loaded_shards, desc="Copying shards to RAM", unit="shard",
        ) as bar:
            for i, shard in enumerate(bar):
                n = shard["images"].shape[0]
                if max_samples and offset + n > max_samples:
                    n = max_samples - offset
                self.images[offset:offset + n] = shard["images"][:n]
                self.nodata_masks[offset:offset + n] = shard["nodata_masks"][:n]
                self.lats[offset:offset + n] = shard["lats"][:n]
                self.lons[offset:offset + n] = shard["lons"][:n]
                self.files.extend(shard["files"][:n])
                offset += n
                gbps = (offset * bytes_per_sample / 1e9) / max(time.perf_counter() - t0, 1e-9)
                bar.set_postfix(samples=offset, GBps=f"{gbps:.1f}")
                loaded_shards[i] = None  # free shard memory immediately
                if max_samples and offset >= max_samples:
                    break
        del loaded_shards
        logger.info(f"[TIMING] copy loop: {time.perf_counter() - t0:.1f}s")

        self.num_samples = total_samples
        mem_gb = (self.images.numel() * 2 + self.nodata_masks.numel()) / 1e9
        logger.info(f"Loaded {self.num_samples} samples into CPU RAM (~{mem_gb:.1f} GB)")

        # Pre-compute season and year indices
        num_timesteps = 140
        num_seasons = 4
        self._season_ids = torch.tensor(
            [t % num_seasons for t in range(num_timesteps)], dtype=torch.long
        )
        self._year_indices = torch.tensor(
            [t // num_seasons for t in range(num_timesteps)], dtype=torch.long
        )

    def _load_shards_parallel(
        self,
        shard_paths: List[Path],
        num_workers: int = 8
    ) -> List[dict]:
        """Load multiple shards in parallel using ThreadPoolExecutor.

        Args:
            shard_paths: List of shard file paths to load.
            num_workers: Number of parallel workers.

        Returns:
            List of loaded shard data dictionaries in same order as input.
        """
        # Use dict to preserve order
        results = {}
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks with index to preserve order
            futures = {
                executor.submit(load_shard, path): (idx, path)
                for idx, path in enumerate(shard_paths)
            }

            # Collect results with progress bar
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Loading shards",
                unit="shard"
            ):
                idx, path = futures[future]
                results[idx] = future.result()

        # Return in original order
        return [results[i] for i in range(len(shard_paths))]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample - just tensor indexing, no I/O."""
        return {
            "images": self.images[idx],
            "season_ids": self._season_ids,  # No clone needed, read-only
            "year_indices": self._year_indices,
            "lat": self.lats[idx],
            "lon": self.lons[idx],
            "nodata_mask": self.nodata_masks[idx],
        }

    def share_memory_(self) -> "InMemoryNDVIDataset":
        """Move all tensors to shared memory for multi-process access.

        Call this before spawning worker processes to avoid duplicating
        the dataset in each process's memory. All processes will then
        share the same underlying memory.

        Returns:
            self for method chaining
        """
        t0 = time.perf_counter()
        self.images.share_memory_()
        logger.info(f"[TIMING] share_memory_(images): {time.perf_counter() - t0:.1f}s")
        t0 = time.perf_counter()
        self.nodata_masks.share_memory_()
        logger.info(f"[TIMING] share_memory_(nodata_masks): {time.perf_counter() - t0:.1f}s")
        self.lats.share_memory_()
        self.lons.share_memory_()
        self._season_ids.share_memory_()
        self._year_indices.share_memory_()
        logger.info(f"Moved {self.num_samples} samples to shared CPU RAM")
        return self


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    sampler: Optional[torch.utils.data.Sampler] = None,
) -> torch.utils.data.DataLoader:
    """Create DataLoader for training.

    Args:
        dataset: Dataset to load from.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle data (ignored if sampler is provided).
        num_workers: Number of parallel data loading workers.
        pin_memory: Whether to pin memory for faster GPU transfer.
        prefetch_factor: Number of batches to prefetch per worker.
        sampler: Optional sampler for distributed training.

    Returns:
        Configured DataLoader instance.
    """
    effective_shuffle = shuffle if sampler is None else False

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=effective_shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
