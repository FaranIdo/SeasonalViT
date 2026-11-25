"""Shard I/O utilities.

Loads shard data from either:
  - .safetensors files (tensors) + .files.json sidecar (filenames), or
  - .pt files (single torch.save dict containing tensors + filenames).

Both formats share the same schema: images, nodata_masks, lats, lons, files.
"""

import json
from pathlib import Path
from typing import Dict, List, Union

import torch
from safetensors.torch import load_file


def _load_safetensors_shard(path: Path) -> Dict[str, Union[torch.Tensor, List[str]]]:
    """Load a shard saved as .safetensors + .files.json sidecar."""
    tensors = load_file(str(path))
    files_path = path.with_suffix("").with_suffix(".files.json")
    with open(files_path) as f:
        files = json.load(f)
    return {
        "images": tensors["images"],
        "nodata_masks": tensors["nodata_masks"].to(torch.bool),
        "lats": tensors["lats"],
        "lons": tensors["lons"],
        "files": files,
    }


def _load_pt_shard(path: Path) -> Dict[str, Union[torch.Tensor, List[str]]]:
    """Load a shard saved as a single torch.save dict."""
    data = torch.load(str(path), map_location="cpu", weights_only=False)
    return {
        "images": data["images"],
        "nodata_masks": data["nodata_masks"].to(torch.bool),
        "lats": data["lats"],
        "lons": data["lons"],
        "files": data["files"],
    }


def load_shard(path: Union[str, Path]) -> Dict[str, Union[torch.Tensor, List[str]]]:
    """Load a shard from either .safetensors or .pt format.

    Dispatches on the file extension. Both formats return the same schema.

    Args:
        path: Path to a shard file (.safetensors or .pt).

    Returns:
        Dict with keys: images, nodata_masks, lats, lons, files.
    """
    path = Path(path)
    if path.suffix == ".safetensors":
        return _load_safetensors_shard(path)
    if path.suffix == ".pt":
        return _load_pt_shard(path)
    raise ValueError(f"Unsupported shard format: {path.suffix} (expected .safetensors or .pt)")


def discover_shards(shard_dir: Union[str, Path]) -> List[Path]:
    """Find shard files in a directory, preferring .safetensors over .pt.

    If a shard exists in both formats, .safetensors wins. Returns paths sorted
    by shard stem so that ordering is stable across formats.

    Args:
        shard_dir: Directory containing shard_*.safetensors and/or shard_*.pt files.

    Returns:
        List of shard file paths. Empty if no shards found.
    """
    shard_dir = Path(shard_dir)
    safetensors = {p.stem: p for p in shard_dir.glob("shard_*.safetensors")}
    pt = {p.stem: p for p in shard_dir.glob("shard_*.pt")}
    merged = {**pt, **safetensors}  # safetensors overrides pt on collision
    return [merged[k] for k in sorted(merged)]
