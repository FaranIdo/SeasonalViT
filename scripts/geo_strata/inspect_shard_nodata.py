"""Quick check: do safetensors shards still encode -9999 NoData, or was it stripped?"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def main() -> None:
    shard_path = sys.argv[1]
    print(f"Inspecting {shard_path}")
    if shard_path.endswith(".safetensors"):
        from safetensors.numpy import load_file

        data = load_file(shard_path)
        print("  keys:", list(data.keys()))
        for k, v in data.items():
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
            if v.ndim < 3:
                continue
            print(f"    min={v.min()}, max={v.max()}, has_-9999={(v == -9999).any()}, n_-9999={(v == -9999).sum()}")
            f32 = v.astype(np.float32)
            print(f"    has_nan={np.isnan(f32).any()}, n_nan={np.isnan(f32).sum()}")
            print(f"    band0 min/max across all samples: {v[:, 0].min()}/{v[:, 0].max()}")
    else:
        import torch

        d = torch.load(shard_path, map_location="cpu", weights_only=False)
        if isinstance(d, dict):
            for k, v in d.items():
                print(f"  {k}: type={type(v).__name__}", end="")
                if hasattr(v, "shape"):
                    print(f", shape={tuple(v.shape)}, dtype={v.dtype}")
                    arr = v.float().cpu().numpy()
                    print(f"    min={arr.min()}, max={arr.max()}, has_-9999={(arr == -9999).any()}, n_-9999={(arr == -9999).sum()}")
                else:
                    print()
                if hasattr(v, "shape"):
                    break


if __name__ == "__main__":
    main()
