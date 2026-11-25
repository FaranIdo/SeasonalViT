"""Geographic block splitting for train/val/test sets.

This is the core library module for creating geographically-separated data splits.
It provides reusable functions that can be imported by other scripts.

Related scripts:
- scripts/generate_splits.py: CLI wrapper that requires actual TIF files
- scripts/generate_splits_from_locations.py: Standalone CLI that works from
  locations.json without needing the actual TIF files

Why geographic splits matter:
    Random splits cause data leakage because nearby pixels are highly correlated
    (spatial autocorrelation). Geographic blocking ensures test samples are
    spatially separated from training data.

Split strategies provided:

1. create_geographic_splits():
   Simple train/val/test split using K-means clustering.
   Use for quick experiments or when you don't need temporal separation.

2. create_research_splits():
   Research-grade 5-way split with geographic AND temporal separation,
   following the GreenEarthNet (CVPR 2024) methodology:
   - train: Group A locations, predict 2022 from 1990-2021
   - val_iid: Group A subset, same years as train (IID baseline)
   - ood_t: Group A subset, predict 2023-2024 from 1990-2022 (temporal OOD)
   - ood_s: Group B locations, same years as train (spatial OOD)
   - ood_st: Group B locations, predict 2023-2024 (both OOD)

I/O functions:
- save_splits_to_json(): Save split assignments to JSON manifest (no file copying)
- load_splits_from_json(): Load splits from JSON manifest
- load_splits_from_folder(): Load from JSON or legacy folder structure

Example usage:
    from datasets.splits import create_research_splits, save_splits_to_json

    # Create splits from file paths
    splits = create_research_splits(file_paths=list_of_tif_paths)

    # Save to JSON manifest
    save_splits_to_json(splits, "data/splits.json")

    # Later, load the splits
    splits = load_splits_from_json("data/splits.json")
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


def create_geographic_splits(
    file_paths: Optional[List[Union[str, Path]]] = None,
    coordinates: Optional[List[Tuple[float, float]]] = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    n_blocks: int = 100,
    random_state: int = 42,
) -> Dict[str, np.ndarray]:
    """Split data by geographic clusters to prevent spatial leakage.

    Uses K-means clustering to group samples into spatial blocks,
    then allocates blocks to train/val/test sets. This ensures
    test samples are spatially separated from training data.

    Args:
        file_paths: List of file paths (coordinates extracted from filenames).
            Format: {sample_id}_{lat}_{lon}.tif
        coordinates: List of (lat, lon) tuples. Provide this OR file_paths.
        test_size: Fraction of blocks for test set.
        val_size: Fraction of blocks for validation set.
        n_blocks: Number of spatial clusters (more = finer granularity).
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary with 'train', 'val', 'test' keys containing index arrays.

    Raises:
        ValueError: If neither file_paths nor coordinates provided.
    """
    if file_paths is None and coordinates is None:
        raise ValueError("Provide either file_paths or coordinates")

    # Extract coordinates from file paths if needed
    if coordinates is None:
        coordinates = _extract_coordinates_from_paths(file_paths)

    coords = np.array(coordinates)
    n_samples = len(coords)

    # Adjust n_blocks if we have fewer samples
    n_blocks = min(n_blocks, n_samples // 3)
    if n_blocks < 3:
        n_blocks = 3

    np.random.seed(random_state)

    # Cluster into spatial blocks using K-means
    block_labels = _kmeans_cluster(coords, n_blocks, random_state)

    # Shuffle and allocate blocks
    unique_blocks = np.unique(block_labels)
    np.random.shuffle(unique_blocks)

    n_test = max(1, int(len(unique_blocks) * test_size))
    n_val = max(1, int(len(unique_blocks) * val_size))

    test_blocks = set(unique_blocks[:n_test])
    val_blocks = set(unique_blocks[n_test:n_test + n_val])
    train_blocks = set(unique_blocks[n_test + n_val:])

    train_idx = np.where([b in train_blocks for b in block_labels])[0]
    val_idx = np.where([b in val_blocks for b in block_labels])[0]
    test_idx = np.where([b in test_blocks for b in block_labels])[0]

    return {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }


def _extract_coordinates_from_paths(
    file_paths: List[Union[str, Path]]
) -> List[Tuple[float, float]]:
    """Extract coordinates from filenames.

    Filename format: {sample_id}_{lat}_{lon}.tif
    Example: 0000010_36d3617_114d3828.tif -> (36.3617, 114.3828)

    Args:
        file_paths: List of file paths.

    Returns:
        List of (lat, lon) tuples.
    """
    import re

    pattern = re.compile(r"^(\d+)_(-?\d+)d(\d+)_(-?\d+)d(\d+)\.tif$")
    coordinates = []

    for path in file_paths:
        filename = Path(path).name
        match = pattern.match(filename)

        if match is None:
            raise ValueError(
                f"Filename '{filename}' doesn't match expected pattern"
            )

        lat_int, lat_dec, lon_int, lon_dec = match.groups()[1:]
        lat = float(f"{lat_int}.{lat_dec}")
        lon = float(f"{lon_int}.{lon_dec}")
        coordinates.append((lat, lon))

    return coordinates


def _kmeans_cluster(
    coords: np.ndarray,
    n_clusters: int,
    random_state: int,
    max_iter: int = 300,
) -> np.ndarray:
    """Simple K-means clustering implementation.

    Avoids sklearn dependency for this simple use case.

    Args:
        coords: (N, 2) array of coordinates.
        n_clusters: Number of clusters.
        random_state: Random seed.
        max_iter: Maximum iterations.

    Returns:
        Cluster labels for each sample.
    """
    np.random.seed(random_state)
    n_samples = len(coords)

    # Initialize centroids randomly
    indices = np.random.choice(n_samples, n_clusters, replace=False)
    centroids = coords[indices].copy()

    labels = np.zeros(n_samples, dtype=np.int32)

    for _ in range(max_iter):
        # Assign to nearest centroid
        distances = _compute_distances(coords, centroids)
        new_labels = np.argmin(distances, axis=1)

        # Check for convergence
        if np.array_equal(labels, new_labels):
            break

        labels = new_labels

        # Update centroids
        for k in range(n_clusters):
            mask = labels == k
            if mask.any():
                centroids[k] = coords[mask].mean(axis=0)

    return labels


def _compute_distances(coords: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Compute distances from each point to each centroid.

    Uses Euclidean distance on lat/lon (approximate for small regions).

    Args:
        coords: (N, 2) array of coordinates.
        centroids: (K, 2) array of centroid coordinates.

    Returns:
        (N, K) array of distances.
    """
    # Expand dims for broadcasting
    coords_exp = coords[:, np.newaxis, :]  # (N, 1, 2)
    centroids_exp = centroids[np.newaxis, :, :]  # (1, K, 2)

    # Euclidean distance
    diff = coords_exp - centroids_exp  # (N, K, 2)
    return np.sqrt((diff ** 2).sum(axis=2))  # (N, K)


def get_split_statistics(
    splits: Dict[str, np.ndarray],
    coordinates: List[Tuple[float, float]],
) -> Dict[str, Dict[str, float]]:
    """Compute statistics for each split.

    Args:
        splits: Dictionary with 'train', 'val', 'test' index arrays.
        coordinates: List of (lat, lon) tuples.

    Returns:
        Dictionary with statistics for each split.
    """
    coords = np.array(coordinates)
    stats = {}

    for split_name, indices in splits.items():
        split_coords = coords[indices]
        stats[split_name] = {
            "n_samples": len(indices),
            "lat_min": float(split_coords[:, 0].min()),
            "lat_max": float(split_coords[:, 0].max()),
            "lon_min": float(split_coords[:, 1].min()),
            "lon_max": float(split_coords[:, 1].max()),
            "lat_mean": float(split_coords[:, 0].mean()),
            "lon_mean": float(split_coords[:, 1].mean()),
        }

    return stats


def load_locations_json(
    locations_path: Union[str, Path]
) -> List[Tuple[float, float]]:
    """Load coordinates from locations.json file.

    Args:
        locations_path: Path to locations.json file.

    Returns:
        List of (lat, lon) tuples.
    """
    with open(locations_path) as f:
        locations = json.load(f)

    return [(v[0], v[1]) for v in locations.values()]


# =============================================================================
# Research-Grade Splits (Geographic + Temporal Separation)
# =============================================================================


def create_research_splits(
    file_paths: List[Union[str, Path]],
    coordinates: Optional[List[Tuple[float, float]]] = None,
    train_size: int = 200_000,
    val_size: int = 10_000,
    ood_t_size: int = 15_000,
    ood_spatial_size: int = 25_000,
    n_blocks: int = 100,
    random_state: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """Create research-grade splits with geographic and temporal separation.

    Implements the GreenEarthNet (CVPR 2024) split strategy:
    - Geographic separation via K-means clustering of locations
    - Temporal separation via different input/predict year configurations

    Split structure:
    - train: Group A locations, predict 2022 from 1990-2021
    - val_iid: Group A subset, predict 2022 from 1990-2021 (IID baseline)
    - ood_t: Group A subset, predict 2023-2024 from 1990-2022 (temporal OOD)
    - ood_s: Group B locations, predict 2022 from 1990-2021 (spatial OOD)
    - ood_st: Group B locations, predict 2023-2024 from 1990-2022 (both OOD)

    Args:
        file_paths: List of file paths to GeoTIFF files.
        coordinates: Optional list of (lat, lon) tuples. If not provided,
            coordinates are extracted from file paths.
        train_size: Target number of training samples.
        val_size: Target number of validation samples.
        ood_t_size: Target number of temporal OOD samples.
        ood_spatial_size: Target number of spatial OOD samples (Group B).
        n_blocks: Number of K-means clusters for geographic blocking.
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary with split information:
        {
            "train": {
                "files": [...],
                "input_years": (1990, 2021),
                "predict_years": (2022, 2022),
            },
            "val_iid": {...},
            "ood_t": {...},
            "ood_s": {...},
            "ood_st": {...},
        }
    """
    file_paths = [Path(p) for p in file_paths]
    n_samples = len(file_paths)

    # Extract coordinates if not provided
    if coordinates is None:
        coordinates = _extract_coordinates_from_paths(file_paths)

    coords = np.array(coordinates)

    # Adjust n_blocks if needed
    n_blocks = min(n_blocks, n_samples // 10)
    n_blocks = max(n_blocks, 5)

    np.random.seed(random_state)

    # Step 1: Cluster into spatial blocks
    block_labels = _kmeans_cluster(coords, n_blocks, random_state)

    # Step 2: Allocate blocks to Group A and Group B
    group_a_blocks, group_b_blocks = _allocate_blocks_to_groups(
        block_labels, ood_spatial_size, random_state
    )

    # Step 3: Get sample indices for each group
    group_a_indices = np.where(np.isin(block_labels, list(group_a_blocks)))[0]
    group_b_indices = np.where(np.isin(block_labels, list(group_b_blocks)))[0]

    # Step 4: Within Group A, split into train and val/ood_t subset
    np.random.shuffle(group_a_indices)
    val_ood_size = val_size + ood_t_size
    val_ood_indices = group_a_indices[:val_ood_size]
    train_indices = group_a_indices[val_ood_size:]

    # Limit train to requested size
    if len(train_indices) > train_size:
        train_indices = train_indices[:train_size]

    # Step 5: Split val/ood_t into val_iid and ood_t
    val_iid_indices = val_ood_indices[:val_size]
    ood_t_indices = val_ood_indices  # ood_t uses all val_ood locations

    # Step 6: Group B becomes ood_s and ood_st (same locations, different years)
    ood_s_indices = group_b_indices
    ood_st_indices = group_b_indices  # Same locations as ood_s

    # Build result with file paths and year configurations
    splits = {
        "train": {
            "files": [str(file_paths[i]) for i in train_indices],
            "input_years": (1990, 2021),
            "predict_years": (2022, 2022),
        },
        "val_iid": {
            "files": [str(file_paths[i]) for i in val_iid_indices],
            "input_years": (1990, 2021),
            "predict_years": (2022, 2022),
        },
        "ood_t": {
            "files": [str(file_paths[i]) for i in ood_t_indices],
            "input_years": (1990, 2022),
            "predict_years": (2023, 2024),
        },
        "ood_s": {
            "files": [str(file_paths[i]) for i in ood_s_indices],
            "input_years": (1990, 2021),
            "predict_years": (2022, 2022),
        },
        "ood_st": {
            "files": [str(file_paths[i]) for i in ood_st_indices],
            "input_years": (1990, 2022),
            "predict_years": (2023, 2024),
        },
    }

    return splits


def _allocate_blocks_to_groups(
    block_labels: np.ndarray,
    target_group_b_size: int,
    random_state: int,
) -> Tuple[set, set]:
    """Allocate spatial blocks to Group A and Group B.

    Randomly selects blocks for Group B until reaching target size.

    Args:
        block_labels: Array of block assignments for each sample.
        target_group_b_size: Target number of samples in Group B.
        random_state: Random seed.

    Returns:
        Tuple of (group_a_blocks, group_b_blocks) as sets of block IDs.
    """
    np.random.seed(random_state + 1)  # Different seed for block allocation

    unique_blocks = np.unique(block_labels)
    block_sizes = {b: (block_labels == b).sum() for b in unique_blocks}

    # Shuffle blocks and allocate to Group B until target reached
    shuffled_blocks = list(unique_blocks)
    np.random.shuffle(shuffled_blocks)

    group_b_blocks = set()
    group_b_count = 0

    for block in shuffled_blocks:
        if group_b_count >= target_group_b_size:
            break
        group_b_blocks.add(block)
        group_b_count += block_sizes[block]

    group_a_blocks = set(unique_blocks) - group_b_blocks

    return group_a_blocks, group_b_blocks


def save_splits_to_json(
    splits: Dict[str, Dict[str, Any]],
    output_path: Union[str, Path],
    n_blocks: int = 100,
    random_state: int = 42,
) -> None:
    """Save split assignments to a JSON manifest file.

    This avoids copying files - the JSON just records which files belong
    to each split. The dataloader reads this manifest and loads files
    from the original location.

    Args:
        splits: Dictionary from create_research_splits().
        output_path: Path to save the JSON manifest (e.g., data/splits.json).
        n_blocks: Number of blocks used (for metadata).
        random_state: Random state used (for metadata).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Find common source directory
    all_files = []
    for split_info in splits.values():
        all_files.extend(split_info["files"])

    if all_files:
        # Get the common parent directory
        first_file = Path(all_files[0])
        source_dir = str(first_file.parent.resolve())
    else:
        source_dir = ""

    # Calculate unique total (accounting for shared locations)
    unique_total = (
        len(splits["train"]["files"])
        + len(splits["ood_t"]["files"])  # includes val_iid
        + len(splits["ood_s"]["files"])
    )

    # Build manifest with just filenames (not full paths)
    manifest = {
        "created": datetime.now().isoformat(),
        "source_dir": source_dir,
        "n_blocks": n_blocks,
        "random_state": random_state,
        "total_unique_samples": unique_total,
        "splits": {},
    }

    for split_name, split_info in splits.items():
        # Store just filenames, not full paths
        filenames = [Path(f).name for f in split_info["files"]]
        manifest["splits"][split_name] = {
            "files": filenames,
            "count": len(filenames),
            "input_years": list(split_info["input_years"]),
            "predict_years": list(split_info["predict_years"]),
        }

    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)


def load_splits_from_json(
    manifest_path: Union[str, Path],
    source_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Load splits from a JSON manifest file.

    Args:
        manifest_path: Path to the splits manifest JSON file.
        source_dir: Optional override for source directory. If not provided,
            uses the source_dir recorded in the manifest.

    Returns:
        Dictionary with split information:
        {
            "train": {
                "files": [Path(...), ...],
                "input_years": (1990, 2021),
                "predict_years": (2022, 2022),
            },
            ...
        }
    """
    manifest_path = Path(manifest_path)

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Use provided source_dir or fall back to manifest's source_dir
    if source_dir is not None:
        data_dir = Path(source_dir)
    else:
        data_dir = Path(manifest["source_dir"])

    splits = {}
    for split_name, split_info in manifest["splits"].items():
        # Reconstruct full paths from filenames
        files = [data_dir / filename for filename in split_info["files"]]

        splits[split_name] = {
            "files": files,
            "input_years": tuple(split_info["input_years"]),
            "predict_years": tuple(split_info["predict_years"]),
        }

    return splits


# Keep old function names as aliases for backward compatibility
def organize_splits_to_folders(
    splits: Dict[str, Dict[str, Any]],
    output_dir: Union[str, Path],
    copy_files: bool = True,
    n_blocks: int = 100,
    random_state: int = 42,
) -> None:
    """DEPRECATED: Use save_splits_to_json() instead.

    This function copies files which is slow and wastes disk space.
    Kept for backward compatibility.
    """
    import warnings
    warnings.warn(
        "organize_splits_to_folders() is deprecated. "
        "Use save_splits_to_json() instead to avoid copying files.",
        DeprecationWarning,
        stacklevel=2,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create split folders and copy/link files
    for split_name, split_info in splits.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(exist_ok=True)

        for src_path in split_info["files"]:
            src = Path(src_path)
            dst = split_dir / src.name

            if dst.exists():
                continue

            if copy_files:
                shutil.copy2(src, dst)
            else:
                dst.symlink_to(src.resolve())

    # Also save the JSON manifest for reference
    save_splits_to_json(
        splits,
        output_dir / "splits.json",
        n_blocks=n_blocks,
        random_state=random_state,
    )


def load_splits_from_folder(
    splits_dir: Union[str, Path],
) -> Dict[str, Dict[str, Any]]:
    """Load splits from organized folder structure or JSON manifest.

    Checks for splits.json first, falls back to scanning folders.

    Args:
        splits_dir: Path to splits directory containing either:
            - splits.json manifest file, OR
            - Split folders (train/, val_iid/, etc.)

    Returns:
        Dictionary with split information.
    """
    splits_dir = Path(splits_dir)

    # Prefer JSON manifest if it exists
    json_path = splits_dir / "splits.json"
    if json_path.exists():
        return load_splits_from_json(json_path)

    # Also check if splits_dir is actually a JSON file
    if splits_dir.suffix == ".json" and splits_dir.exists():
        return load_splits_from_json(splits_dir)

    # Fall back to scanning folders (legacy behavior)
    metadata_path = splits_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

        splits = {}
        for split_name, split_meta in metadata["splits"].items():
            split_folder = splits_dir / split_name
            files = sorted(split_folder.glob("*.tif"))

            splits[split_name] = {
                "files": files,
                "input_years": tuple(split_meta["input_years"]),
                "predict_years": tuple(split_meta["predict_years"]),
            }

        return splits

    raise FileNotFoundError(
        f"No splits.json or metadata.json found in {splits_dir}"
    )


def get_research_split_statistics(
    splits: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Compute statistics for research splits.

    Args:
        splits: Dictionary from create_research_splits() or load_splits_from_folder().

    Returns:
        Dictionary with statistics for each split.
    """
    stats = {}

    for split_name, split_info in splits.items():
        files = split_info["files"]
        coords = _extract_coordinates_from_paths(files)
        coords = np.array(coords)

        stats[split_name] = {
            "n_samples": len(files),
            "input_years": split_info["input_years"],
            "predict_years": split_info["predict_years"],
            "lat_min": float(coords[:, 0].min()) if len(coords) > 0 else None,
            "lat_max": float(coords[:, 0].max()) if len(coords) > 0 else None,
            "lon_min": float(coords[:, 1].min()) if len(coords) > 0 else None,
            "lon_max": float(coords[:, 1].max()) if len(coords) > 0 else None,
        }

    return stats
