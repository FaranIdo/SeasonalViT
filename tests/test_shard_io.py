"""Tests for shard_io: load_shard and discover_shards handle both formats."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from datasets.shard_io import discover_shards, load_shard


def _make_shard_dict() -> dict:
    return {
        "images": torch.randn(2, 140, 16, 16, dtype=torch.float16),
        "nodata_masks": torch.zeros(2, 140, 16, 16, dtype=torch.bool),
        "lats": torch.tensor([1.0, 2.0]),
        "lons": torch.tensor([3.0, 4.0]),
        "files": ["a.tif", "b.tif"],
    }


def _save_safetensors(data: dict, path: Path) -> None:
    tensors = {k: (v.to(torch.uint8) if k == "nodata_masks" else v)
               for k, v in data.items() if k != "files"}
    save_file(tensors, str(path))
    sidecar = path.with_suffix("").with_suffix(".files.json")
    with open(sidecar, "w") as f:
        json.dump(data["files"], f)


def _assert_shard_matches(loaded: dict, expected: dict) -> None:
    assert torch.equal(loaded["images"], expected["images"])
    assert torch.equal(loaded["nodata_masks"], expected["nodata_masks"])
    assert loaded["nodata_masks"].dtype == torch.bool
    assert torch.equal(loaded["lats"], expected["lats"])
    assert torch.equal(loaded["lons"], expected["lons"])
    assert loaded["files"] == expected["files"]


def test_load_shard_safetensors(tmp_path):
    data = _make_shard_dict()
    path = tmp_path / "shard_00000.safetensors"
    _save_safetensors(data, path)
    _assert_shard_matches(load_shard(path), data)


def test_load_shard_pt(tmp_path):
    data = _make_shard_dict()
    path = tmp_path / "shard_00000.pt"
    torch.save(data, path)
    _assert_shard_matches(load_shard(path), data)


def test_load_shard_unsupported_format(tmp_path):
    path = tmp_path / "shard_00000.bin"
    path.touch()
    with pytest.raises(ValueError, match="Unsupported shard format"):
        load_shard(path)


def test_discover_shards_pt_only(tmp_path):
    for i in range(3):
        torch.save(_make_shard_dict(), tmp_path / f"shard_{i:05d}.pt")
    found = discover_shards(tmp_path)
    assert [p.suffix for p in found] == [".pt", ".pt", ".pt"]
    assert [p.stem for p in found] == ["shard_00000", "shard_00001", "shard_00002"]


def test_discover_shards_safetensors_preferred_over_pt(tmp_path):
    data = _make_shard_dict()
    torch.save(data, tmp_path / "shard_00000.pt")
    torch.save(data, tmp_path / "shard_00001.pt")
    _save_safetensors(data, tmp_path / "shard_00000.safetensors")

    found = discover_shards(tmp_path)
    by_stem = {p.stem: p.suffix for p in found}
    assert by_stem == {"shard_00000": ".safetensors", "shard_00001": ".pt"}


def test_discover_shards_empty_dir(tmp_path):
    assert discover_shards(tmp_path) == []
