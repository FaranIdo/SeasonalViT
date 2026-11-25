"""Sample lookup utilities for finding and loading specific samples.

Supports two data formats:
- Shard files (shard_*.safetensors): indexed by sample_id within each shard
- Raw GeoTIFF files ({sample_id}_{lat}_{lon}.tif): loaded directly with rasterio
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from datasets.shard_io import load_shard
from eval.evaluate_baselines import extract_sample_id

logger = logging.getLogger(__name__)

NODATA_VALUE = -9999.0
CROP_SIZE = 256


# ---------------------------------------------------------------------------
# TIF loading
# ---------------------------------------------------------------------------


def find_tif_by_sample_id(data_dir: Path, sample_id: str) -> Optional[Path]:
    """Find a TIF file by sample ID prefix.

    Args:
        data_dir: Directory containing TIF files.
        sample_id: Sample ID (e.g. "0515794").

    Returns:
        Path to the matching TIF file, or None if not found.
    """
    matches = list(data_dir.glob(f"{sample_id}_*.tif"))
    if not matches:
        return None
    return matches[0]


def load_tif_as_tensor(tif_path: Path) -> dict:
    """Load a GeoTIFF NDVI file and return preprocessed tensors.

    Applies the same preprocessing as convert_to_shards.py:
    center-crop 264->256, fill nodata with zeros, convert to float16.

    Args:
        tif_path: Path to the GeoTIFF file.

    Returns:
        Dict with images (T, H, W), nodata_mask (T, H, W), filename (str).
    """
    import rasterio

    with rasterio.open(tif_path) as src:
        data_raw = src.read()  # (140, 264, 264)

    # Nodata mask
    nodata_mask_full = data_raw == NODATA_VALUE

    # Fill nodata with zeros
    data_filled = np.where(nodata_mask_full, 0.0, data_raw)

    # Center crop to 256x256
    _, h, w = data_filled.shape
    sh = (h - CROP_SIZE) // 2
    sw = (w - CROP_SIZE) // 2
    images = data_filled[:, sh:sh + CROP_SIZE, sw:sw + CROP_SIZE]
    nodata_mask = nodata_mask_full[:, sh:sh + CROP_SIZE, sw:sw + CROP_SIZE]

    return {
        "images": torch.from_numpy(images.astype(np.float32)),
        "nodata_mask": torch.from_numpy(nodata_mask),
        "filename": tif_path.name,
    }


def load_samples_from_tifs(
    sample_ids: List[str],
    data_dir: Path,
) -> Dict[str, dict]:
    """Load samples directly from TIF files.

    Args:
        sample_ids: List of sample IDs to load.
        data_dir: Directory containing TIF files.

    Returns:
        Dict mapping sample_id to {images, nodata_mask, filename}.
    """
    results: Dict[str, dict] = {}

    for sid in sample_ids:
        tif_path = find_tif_by_sample_id(data_dir, sid)
        if tif_path is None:
            logger.warning(f"TIF not found for sample {sid} in {data_dir}")
            continue
        logger.info(f"Loading {tif_path.name}")
        results[sid] = load_tif_as_tensor(tif_path)

    logger.info(f"Loaded {len(results)}/{len(sample_ids)} samples from TIFs")
    return results


# ---------------------------------------------------------------------------
# Shard loading
# ---------------------------------------------------------------------------


def build_sample_index(shard_dir: Path) -> Dict[str, Tuple[Path, int]]:
    """Build an index mapping sample_id -> (shard_path, index_in_shard).

    Scans all shard files and reads their `files` list to build a lookup table.

    Args:
        shard_dir: Directory containing shard_*.safetensors files.

    Returns:
        Dictionary mapping sample_id to (shard_path, index_within_shard).
    """
    shard_files = sorted(shard_dir.glob("shard_*.safetensors"))
    if not shard_files:
        logger.error(f"No shard_*.safetensors files found in {shard_dir}")
        return {}

    logger.info(f"Indexing {len(shard_files)} shards in {shard_dir}")
    index: Dict[str, Tuple[Path, int]] = {}

    for shard_path in shard_files:
        shard_data = load_shard(shard_path)
        files_list = shard_data.get("files", [])

        for idx, filename in enumerate(files_list):
            sample_id = extract_sample_id(filename)
            index[sample_id] = (shard_path, idx)

        del shard_data

    logger.info(f"Indexed {len(index)} samples across {len(shard_files)} shards")
    return index


def load_samples_from_shards(
    sample_ids: List[str],
    index: Dict[str, Tuple[Path, int]],
) -> Dict[str, dict]:
    """Load specific samples by ID from shards, grouping by shard for efficient I/O.

    Args:
        sample_ids: List of sample IDs to load.
        index: Sample index from build_sample_index().

    Returns:
        Dict mapping sample_id to {images, nodata_mask, filename}.
    """
    shard_groups: Dict[Path, List[Tuple[str, int]]] = defaultdict(list)
    missing = []

    for sid in sample_ids:
        if sid not in index:
            missing.append(sid)
            continue
        shard_path, idx = index[sid]
        shard_groups[shard_path].append((sid, idx))

    if missing:
        logger.warning(f"Samples not found in index: {missing}")

    results: Dict[str, dict] = {}

    for shard_path, items in shard_groups.items():
        logger.info(f"Loading {len(items)} samples from {shard_path.name}")
        shard_data = load_shard(shard_path)
        images = shard_data["images"]
        nodata_masks = shard_data["nodata_masks"]
        files_list = shard_data.get("files", [])

        for sid, idx in items:
            filename = files_list[idx] if idx < len(files_list) else f"{sid}.tif"
            results[sid] = {
                "images": images[idx],
                "nodata_mask": nodata_masks[idx],
                "filename": filename,
            }

        del shard_data, images, nodata_masks

    logger.info(f"Loaded {len(results)}/{len(sample_ids)} samples from shards")
    return results
