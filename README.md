# SeasonalViT and GloSeaN

Official code for **"SeasonalViT: A Dual-Scale Temporal Transformer for NDVI Forecasting"**
(IEEE JSTARS, 2026), and the **GloSeaN** (Global Seasonal NDVI Archive) dataset.

**Paper:** [IEEE Xplore](https://ieeexplore.ieee.org/document/11595919) ·
DOI [10.1109/JSTARS.2026.3710845](https://doi.org/10.1109/JSTARS.2026.3710845)

SeasonalViT forecasts future NDVI from multi-decadal seasonal image sequences. A
Swin-T spatial encoder embeds each seasonal image, and a dual-scale temporal transformer
models both phenological cycles (within a year) and decadal trends (across years) in a
single forward pass.

## GloSeaN dataset

| Attribute | Value |
|---|---|
| Locations | 200,000 tiles (250,000 sampled from [SSL4EO-L](https://arxiv.org/abs/2306.09424), filtered for completeness) |
| Time span | 1990 to 2024, 4 seasons per year (140 timesteps) |
| Tile size | 264 × 264 pixels at 30 m (Landsat) |
| Format | GeoTIFF, one file per location, bands named `{year}_{season}` |
| CRS / NoData | EPSG:4326 / -9999 |

**Download:** <!-- TODO(release): Hugging Face dataset link --> coming soon.

The location index (`data/locations.json`, `data/locations_200k.json`) and the split
manifests used in the paper (`data/splits_200k.json`, `data/shard_lists/`) are included in
this repository. Per-tile ESA WorldCover land-cover fractions (`landcover_stats.json`, used
by the baseline evaluation, figure, and geographic-strata scripts) ship with the dataset
download; place the file at `data/landcover_stats.json`. The full dataset can also be regenerated from Google Earth Engine with
`data/download/download_ndvi_35years.py` (see `data/download/DOWNLOAD_35YEARS.md`).

| Split | Tiles | Purpose |
|---|---|---|
| `train` | 157,523 | Training |
| `val_iid` | 8,000 | In-distribution validation |
| `ood_t` | 20,000 | Temporal generalization (2023 to 2024 targets) |
| `ood_s` | 22,477 | Spatial generalization (held-out regions) |
| `ood_st` | 22,477 | Spatio-temporal generalization |

See `docs/specific/dataset_splits.md` for how the splits were built and
`docs/ndvi_dataset.md` for band naming and preprocessing.

## Pretrained weights

<!-- TODO(release): Hugging Face model link --> Coming soon: the pretrained (MAE) and
fine-tuned (forecasting) checkpoints.

## Installation

```bash
git clone https://github.com/FaranIdo/SeasonalViT.git
cd SeasonalViT
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run the tests to check the install:

```bash
python -m pytest tests/ -q
```

## Usage

### 1. Convert GeoTIFFs to training shards

```bash
python scripts/convert_to_shards.py \
    --input-dir /path/to/GloSeaN/tiles \
    --output-dir data/shards \
    --splits data/splits_200k.json \
    --samples-per-shard 1024
```

Tiles are written in filename order, and the paper's shards hold 1,024 tiles each, so
`--samples-per-shard 1024` gives the same shard contents (154 train shards).

### 2. Pre-train (masked reconstruction)

The commands below use the paper's training settings on 2 GPUs (batch sizes are per GPU).
Several code defaults differ, so pass these flags explicitly.

```bash
python scripts/pretrain.py --train-dir data/shards/train --val-dir data/shards/val_iid \
    --output-dir runs/pretrain --epochs 40 --lr 1e-4 --batch-size 32 \
    --temporal-subset-years 12 --multi-gpu
```

### 3. Fine-tune (forecasting)

```bash
python scripts/finetune.py --train-dir data/shards/train --val-dir data/shards/val_iid \
    --output-dir runs/finetune --pretrained runs/pretrain/best.pt \
    --epochs 20 --batch-size 14 --lr 5e-5 --predict-years 2022 --freeze-spatial \
    --temporal-subset-years 0 --corr-loss-weight 0.5 --delta-supervision-weight 0.2 --multi-gpu
```

Multi-GPU training: launch with `torchrun --nproc_per_node=N` when using `--multi-gpu`.
On Windows, use `--multi-gpu --spawn --num-gpus N` instead of `torchrun`.

### 4. Evaluate

```bash
python scripts/evaluate.py --checkpoint runs/finetune/best.pt --eval-dir data/shards/ood_s \
    --predict-years 2022 --per-season
```

Forecast horizons (1 to 8 seasons ahead): `scripts/eval_horizons.py`.
Baselines (persistence, previous year, climatology, linear trend, Kalman filter, LightGBM, ConvLSTM,
PredRNN, Prithvi): `baselines/` and `eval/evaluate_baselines.py`.

### Visualize a tile

```bash
python visualize_sample.py --input /path/to/tile.tif --output tile.png
```

## Repository layout

```
model/       SeasonalViT architecture (spatial/temporal/location encoders, decoder)
train/       Pre-training and fine-tuning loops, masking, losses, metrics
datasets/    Shard I/O, data loaders, split utilities
baselines/   Statistical and deep-learning baselines
eval/        Evaluation and visualization
scripts/     Command-line entry points and paper figure scripts
data/        GloSeaN download/verification tools, locations, split manifests
figures/     Dataset figure scripts
docs/        Dataset, architecture, and training documentation
tests/       Unit tests
```

## Citation

```bibtex
@article{faran2026seasonalvit,
  title   = {SeasonalViT: A Dual-Scale Temporal Transformer for NDVI Forecasting},
  author  = {Faran, Ido and Netanyahu, Nathan S. and Shoshany, Maxim},
  journal = {IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  volume  = {19},
  pages   = {25566--25588},
  year    = {2026},
  doi     = {10.1109/JSTARS.2026.3710845}
}
```

## License

Code: MIT (see `LICENSE`). <!-- TODO(release): dataset license (planned CC-BY-4.0) -->

## Acknowledgments

This work was supported by the Israel Ministry of Innovation, Science and Technology
through the provision of computational resources under the "Tuning for Deserts" project.
GloSeaN locations are derived from SSL4EO-L ([Stewart et al., 2023](https://arxiv.org/abs/2306.09424)); imagery is from the
USGS Landsat program via Google Earth Engine.
