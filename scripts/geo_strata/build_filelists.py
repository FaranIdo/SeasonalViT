"""Generate plain-text tile lists for the A2 nodata-inventory runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--splits-json", default="data/splits_200k.json")
    p.add_argument("--out-dir", default="data/geo_strata")
    p.add_argument("--global-sample-size", type=int, default=5000)
    p.add_argument("--seed", type=int, default=17)
    return p.parse_args()


def write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n")
    print(f"  wrote {path} ({len(lines)} entries)")


def main() -> None:
    args = parse_args()
    splits = json.loads(Path(args.splits_json).read_text())["splits"]
    ood_s_files = sorted(set(splits["ood_s"]["files"]))
    others = sorted(set(splits["train"]["files"])
                    | set(splits["val_iid"]["files"])
                    | set(splits["ood_t"]["files"]))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    write_lines(out / "filelist_ood_s.txt", ood_s_files)

    rng = np.random.default_rng(args.seed)
    keep = min(args.global_sample_size, len(others))
    idx = rng.choice(len(others), size=keep, replace=False)
    sampled = sorted(others[i] for i in idx.tolist())
    write_lines(out / "filelist_global_sample.txt", sampled)

    union = sorted(set(ood_s_files) | set(sampled))
    write_lines(out / "filelist_combined.txt", union)


if __name__ == "__main__":
    main()
