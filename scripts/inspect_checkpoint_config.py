"""Print exactly what's stored in the `config` key of a checkpoint.

Used to verify whether ablation checkpoints carry the architecture
fields (anchor_mode, temporal_pathway) needed for in-process eval.
"""

import sys
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path

# Make the repo importable so dataclasses pickled with their original
# module path resolve correctly when this script is invoked directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

# Import for unpickling side-effects (registers `model.config.*`).
import model.config  # noqa: F401, E402


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: inspect_checkpoint_config.py <ckpt.pt> [<ckpt2.pt> ...]")
        sys.exit(1)
    for path_str in sys.argv[1:]:
        ckpt_path = Path(path_str)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        print(f"\n=== {ckpt_path} ===")
        cfg = ckpt.get("config")
        if cfg is None:
            print("  (no 'config' key)")
            continue
        print(f"  type: {type(cfg).__name__}")
        if is_dataclass(cfg):
            for f in fields(cfg):
                if f.name in ("anchor_mode", "temporal_pathway",
                              "predict_years", "freeze_spatial",
                              "delta_supervision_weight"):
                    print(f"  {f.name} = {getattr(cfg, f.name)!r}")
            print(f"  all fields: {list(asdict(cfg).keys())}")


if __name__ == "__main__":
    main()
