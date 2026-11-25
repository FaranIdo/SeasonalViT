# Dataset Splits: Geographic + Temporal Separation

This document describes the train/validation/test split strategy used for GloSeaN and SeasonalViT to evaluate spatial and temporal generalization.

## Why Not Random Splits?

Random splits cause **data leakage** and overestimate model performance due to:

### Spatial Autocorrelation
Nearby locations share similar vegetation patterns, climate, and soil characteristics. If a model sees location A during training and location A' (nearby) during testing, it can "cheat" by memorizing local patterns rather than learning generalizable features.

### Temporal Autocorrelation
Global climate patterns (El Nino, La Nina, droughts) affect all locations in a given year. If a model trains on 2020 data from some locations and tests on 2020 data from others, it can exploit shared year-specific patterns rather than learning robust temporal dynamics.

See Ploton et al. (2020) and Roberts et al. (2017) for the effect of spatial structure on validation and for blocking strategies.

## Split Design

We use a two-level grouping strategy inspired by GreenEarthNet (CVPR 2024):

### Geographic Pools via K-means

First, the 200,000 released tiles (`data/locations_200k.json`) are clustered into 100 spatial blocks using K-means on (lat, lon) coordinates. Then whole blocks are allocated to two pools (called Group A / Group B in the code):

```
K-means (100 clusters on lat/lon of 200,000 tiles)
|
+-- Pool A blocks (~88.8%): train + val_iid + ood_t  --> 177,523 tiles
+-- Pool B blocks (~11.2%): ood_s + ood_st           -->  22,477 tiles
```

This ensures **complete geographic separation** between training locations and OOD-spatial test locations.

### Split Definitions

| Split | Location Pool | Input Years | Predict Years | Samples | Purpose |
|-------|---------------|-------------|---------------|---------|---------|
| **train** | A | 1990-2021 | 2022 | 157,523 | Model training |
| **val_iid** | A (subset of ood_t) | 1990-2021 | 2022 | 8,000 | IID validation (hyperparameter tuning) |
| **ood_t** | A | 1990-2022 | 2023-2024 | 20,000 | Temporal generalization |
| **ood_s** | B | 1990-2021 | 2022 | 22,477 | Spatial generalization |
| **ood_st** | B | 1990-2022 | 2023-2024 | 22,477 | Spatial + temporal generalization |

**Total: 200,000 unique tiles** (157,523 train + 20,000 ood_t + 22,477 Pool B). The paper refers to `val_iid` as `val_id`.

### Why Aren't the Numbers Exact?

The sample counts differ from round targets (e.g., 22,477 instead of 22,000) because **blocks are allocated as whole units** to preserve geographic separation.

1. **K-means creates 100 blocks** of varying sizes (depends on location density, not equal)
2. **Blocks are assigned to Pool B whole**, in random order, until Pool B reaches the target:
   - Target: 22,000 (`--ood-spatial-size` default)
   - Actual: 22,477 (where the last block boundary fell)
3. **Pool A gets the remainder**: 200,000 - 22,477 = 177,523
4. **Within Pool A**: 20,000 shuffled tiles (`--ood-t-size`) form ood_t, the first 8,000 of them (`--val-size`) form val_iid, and the rest is train: 177,523 - 20,000 = 157,523 (below the `--train-size` cap of 158,000, so no tiles are dropped)

If we split blocks to hit exact numbers, nearby locations would end up in different pools → **spatial leakage**. The 477-tile difference is the cost of maintaining proper geographic separation.

### How Geographic Separation Works

The separation is **at block level, not regional**. Pool A and Pool B blocks are interleaved across the globe:

```
World map with 100 K-means blocks (simplified):

Europe:     [Block 12: Pool A]  [Block 45: Pool B]  [Block 78: Pool A]
Asia:       [Block 23: Pool B]  [Block 56: Pool A]  [Block 89: Pool A]
Americas:   [Block 34: Pool A]  [Block 67: Pool B]  [Block 91: Pool A]
```

- **Within each block**: locations are geographically close (K-means clusters nearby points)
- **Block allocation**: blocks are shuffled and assigned to Pool B until the target size is reached; the rest go to Pool A
- **Result**: Pool A (train) and Pool B (ood_s) blocks are scattered across all regions

This means ood_s isn't "all in one continent" while train is elsewhere. Instead, both pools have coverage across the globe, but **no single location appears in both pools**.

This tests whether the model learns **generalizable vegetation patterns** vs **memorizing specific locations**.

### Key Constraints

1. **val_iid is a subset of ood_t**: all 8,000 val_iid tiles are in ood_t; the other 12,000 ood_t tiles are listed in `data/shard_lists/shard_list_ood_t_extra.json`. The two splits use different horizons (2022 vs 2023-2024).

2. **ood_s and ood_st share locations**: both use the same 22,477 Pool B tiles, allowing comparison of temporal effects on unseen locations.

3. **No location overlap between pools or with train**: Pool A tiles never appear in ood_s/ood_st, and train shares no tiles with val_iid or ood_t.

## JSON Manifest

The release split manifest is `data/splits_200k.json` (~9.1 MB). **No files are copied** - the manifest just records which files belong to each split, and files are loaded from wherever the GeoTIFFs are stored.

```
data/splits_200k.json         # Split manifest (~9.1 MB)
/path/to/GloSeaN/tiles/       # GeoTIFFs (unchanged)
+-- 0000010_36d3617_114d3828.tif
+-- 0000014_...
+-- ...
```

### splits_200k.json Format

Excerpt (file lists truncated):

```json
{
  "created": "2026-01-28T15:16:22.835952",
  "source_dir": "",
  "n_blocks": 100,
  "random_state": 42,
  "total_unique_samples": 200000,
  "splits": {
    "train": {
      "files": ["0605717_-21d1055_-63d4661.tif", "..."],
      "count": 157523,
      "input_years": [1990, 2021],
      "predict_years": [2022, 2022]
    },
    "val_iid": {
      "files": ["0547207_27d0005_59d9155.tif", "..."],
      "count": 8000,
      "input_years": [1990, 2021],
      "predict_years": [2022, 2022]
    },
    "ood_t": {
      "files": ["0547207_27d0005_59d9155.tif", "..."],
      "count": 20000,
      "input_years": [1990, 2022],
      "predict_years": [2023, 2024]
    },
    "ood_s": {
      "files": ["0000029_27d5668_118d4049.tif", "..."],
      "count": 22477,
      "input_years": [1990, 2021],
      "predict_years": [2022, 2022]
    },
    "ood_st": {
      "files": ["0000029_27d5668_118d4049.tif", "..."],
      "count": 22477,
      "input_years": [1990, 2022],
      "predict_years": [2023, 2024]
    }
  }
}
```

**Note**: The manifest stores only filenames (not full paths). In the release manifest `source_dir` is empty, so pass the GeoTIFF directory yourself: `source_dir=` to `load_splits_from_json`, or `--input-dir` to `scripts/convert_to_shards.py`.

### Shard Lists

`data/shard_lists/` holds four non-overlapping lists derived from the manifest, in the same `{"source_dir", "splits"}` format:

| File | Split key | Tiles | Contents |
|------|-----------|------:|----------|
| `shard_list_train.json` | `train` | 157,523 | train |
| `shard_list_val_iid.json` | `val_iid` | 8,000 | val_iid |
| `shard_list_ood_t_extra.json` | `ood_t_extra` | 12,000 | ood_t minus val_iid |
| `shard_list_ood_spatial.json` | `ood_spatial` | 22,477 | Pool B (ood_s and ood_st) |

## Usage

### Generating Splits

The release manifest is produced by `scripts/generate_splits_from_locations.py`, which needs only the location index (no GeoTIFFs). With its defaults (`--n-blocks 100`, `--random-state 42`, `--ood-spatial-size 22000`, `--ood-t-size 20000`, `--val-size 8000`, `--train-size 158000`) it reproduces the file lists of `data/splits_200k.json`:

```bash
python scripts/generate_splits_from_locations.py \
    --locations data/locations_200k.json \
    --output data/splits_200k.json

# Optional: non-overlapping shard lists
python scripts/create_unique_shard_lists.py \
    --splits data/splits_200k.json \
    --output-dir data/shard_lists
```

`scripts/generate_splits.py` scans a GeoTIFF directory and uses a different procedure (it holds out `--val-size` + `--ood-t-size` tiles from Pool A), so it does not reproduce the release manifest.

### Loading Splits in Python

```python
from datasets.splits import load_splits_from_json

# source_dir is required for the release manifest (its source_dir is "")
splits = load_splits_from_json("data/splits_200k.json", source_dir="/path/to/GloSeaN/tiles")

# Access file paths (source_dir + filenames)
train_files = splits["train"]["files"]    # List of Path objects
val_files = splits["val_iid"]["files"]

# Access year configurations
input_years = splits["train"]["input_years"]   # (1990, 2021)
predict_years = splits["train"]["predict_years"]  # (2022, 2022)
```

### Sharding, Training, and Evaluation

Training and evaluation scripts read shard directories, not the manifest. First convert the GeoTIFFs into one shard directory per split:

```bash
python scripts/convert_to_shards.py \
    --input-dir /path/to/GloSeaN/tiles \
    --output-dir data/shards \
    --splits data/splits_200k.json
# -> data/shards/{train,val_iid,ood_t,ood_s,ood_st}/shard_*.pt
```

Passing a file from `data/shard_lists/` to `--splits` instead creates `train/`, `val_iid/`, `ood_t_extra/`, or `ood_spatial/` without duplicating tiles shared between splits. `--split-only NAME` converts a single split.

Then train on the shard directories:

```bash
# Pre-training
python scripts/pretrain.py --train-dir data/shards/train --val-dir data/shards/val_iid \
    --output-dir runs/pretrain

# Fine-tuning
python scripts/finetune.py --train-dir data/shards/train --val-dir data/shards/val_iid \
    --output-dir runs/finetune --pretrained runs/pretrain/best.pt
```

Evaluate on each split's shard directory:

```bash
python scripts/evaluate.py --checkpoint runs/finetune/best.pt --eval-dir data/shards/val_iid --predict-years 2022
python scripts/evaluate.py --checkpoint runs/finetune/best.pt --eval-dir data/shards/ood_s --predict-years 2022
```

Per-horizon evaluation on the 2023-2024 targets (default split OOD-st) uses `scripts/eval_horizons.py --checkpoint ... --eval-dir ...`.

## Interpreting Results

Expected performance ordering (best to worst):

```
val_iid > ood_t > ood_s > ood_st
```

- **val_iid vs ood_t gap**: Measures temporal generalization (climate shift)
- **val_iid vs ood_s gap**: Measures spatial generalization (geographic transfer)
- **ood_s vs ood_st gap**: Measures temporal generalization on unseen locations

A well-generalizing model should have small gaps between these metrics. Large gaps indicate overfitting to specific locations or time periods.

## References

1. **Benson et al. (2024)** "Multi-modal Learning for Geospatial Vegetation Forecasting" CVPR 2024 (GreenEarthNet)
   - https://arxiv.org/abs/2303.16198
   - Introduced the OOD-s / OOD-t / OOD-st split strategy for vegetation forecasting

2. **Meyer & Pebesma (2021)** "Predicting into unknown space? Estimating the area of applicability of spatial prediction models" Methods in Ecology and Evolution
   - https://doi.org/10.1111/2041-210X.13650
   - Analysis of spatial dependence between training and test sets

3. **Ploton et al. (2020)** "Spatial validation reveals poor predictive performance of large-scale ecological mapping models" Nature Communications
   - https://doi.org/10.1038/s41467-020-18321-y
   - Compares random and spatial cross-validation for large-scale mapping models

4. **Roberts et al. (2017)** "Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure" Ecography
   - https://doi.org/10.1111/ecog.02881
   - Comprehensive review of blocking strategies for ecological data
