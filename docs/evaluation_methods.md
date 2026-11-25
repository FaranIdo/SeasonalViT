# Evaluation Methods for SeasonalViT Comparison

This document describes the baseline methods, evaluation splits, metrics, and evaluation scripts used to compare SeasonalViT for long-term NDVI forecasting.

## Method Overview

The paper reports eight baselines plus SeasonalViT:

| Category | Method | Code | Trains? | Spatial? |
|----------|--------|------|---------|----------|
| Non-ML | Persistence | `baselines/naive.py` | No | No |
| Non-ML | Previous Year | `baselines/naive.py` | No | No |
| Non-ML | Climatology | `baselines/naive.py` | No | No |
| ML | Linear Trend | `baselines/statistical.py` | No (per-pixel fit) | No |
| ML | Kalman Filter | `baselines/statistical.py` | No (per-pixel filter) | No |
| ML | LightGBM | `baselines/lightgbm/` | Yes (CPU) | No |
| DL | ConvLSTM | `baselines/convlstm/` | Yes (GPU) | Yes |
| DL | PredRNN | `baselines/predrnn/` | Yes (GPU) | Yes |
| Ours | SeasonalViT (80.65 M parameters) | `model/`, `train/` | Yes (GPU) | Yes |

The code also contains three methods that the paper does not report; see [Additional baselines in the code](#additional-baselines-in-the-code-not-reported-in-the-paper).

## Evaluation Splits

There are four evaluation splits. The scripts use the code names (`--split`). OOD-t and OOD-st cover two target years (2023 and 2024).

| Paper | Code (`--split`) | Shard directories under `--shard-root` | Samples | Default predict years |
|-------|------------------|----------------------------------------|---------|-----------------------|
| val_id | `val_iid` | `val_iid/val_iid` | 8,000 | 2022 |
| OOD-t | `ood_t` | `val_iid/val_iid` + `ood_t_extra/ood_t_extra` | 20,000 | 2023, 2024 |
| OOD-s | `ood_s` | `ood_spatial/ood_spatial` | 22,477 | 2022 |
| OOD-st | `ood_st` | `ood_spatial/ood_spatial` | 22,477 | 2023, 2024 |

The split-to-directory mapping is `SPLIT_CONFIGS` in `eval/evaluate_baselines.py`, which the LightGBM and DL evaluation scripts import. In this mapping `ood_t` is the union of the 8,000 `val_iid` tiles and the 12,000 tiles listed in `data/shard_lists/shard_list_ood_t_extra.json`. Sample counts come from the manifests in `data/shard_lists/`.

When a split has two default predict years, the scripts pool both years into one result. To score 2023 and 2024 separately, pass `--predict-years 2023` or `--predict-years 2024`.

## Metrics

Shards store tiles already center-cropped from 264x264 to 256x256 pixels (`scripts/convert_to_shards.py`). All metrics are therefore computed on the 256x256 crop (65,536 pixels per timestep).

| Metric | Definition | Better |
|--------|------------|--------|
| MAE | mean(\|pred - target\|) | Lower |
| RMSE | sqrt(mean((pred - target)^2)) | Lower |
| r^2 | Squared Pearson correlation between prediction and target. This is not the coefficient of determination. | Higher |
| Bias | Absolute bias, \|mean(pred) - mean(target)\| | Lower |

**Valid pixels.** A pixel counts only if the target is not nodata and the method produced a valid prediction there. For example, Persistence marks a pixel invalid when its source timestep is nodata. Samples with no valid pixels are skipped. r^2 is undefined for a sample when the prediction or target has zero variance, and those samples are left out of the r^2 average.

**Aggregation.** Each metric is computed per sample and then averaged over samples, so every sample has equal weight. The unit of a "sample" depends on the evaluation path:

- **Baseline scripts** (`eval/evaluate_baselines.py`, `eval/evaluate_lightgbm.py`, `eval/evaluate_dl_baselines.py`, which all share `aggregate_from_stats`): a sample's metrics pool all valid pixels across all of its target timesteps, for example the four seasons of the predicted year. Per-season results apply the same procedure to that season's timesteps only.
- **SeasonalViT** (`FinetuningTrainer.validate()` with `PerSampleMetricsTracker` in `train/metrics.py`): each sample is scored on one target map. The target timestep is drawn from the configured predict years with a fixed seed (42).

**Extra outputs.** The baseline scripts also report NSE = 1 - MSE / Var(target), which is the coefficient of determination R^2. `PerSampleMetricsTracker` reports the same quantity as `R2_det`.

**RMSE in the baseline scripts.** `eval/evaluate_baselines.py` and the scripts that reuse its aggregation currently output MAE, r^2, NSE and bias, but not RMSE. The per-sample stats files store `sum_sq_error` and `n_valid` for every (sample, target) record, so RMSE can be recomputed from them.

---

## Method Details

### Context window

Every evaluation script computes the last usable input index as

```
context_end_idx = min(target_indices) - 1 - 4 * (context_gap_years - 1)
```

`--context-gap-years` defaults to 1. With the default `ood_t` / `ood_st` targets (2023 and 2024), context ends at 2022 winter (index 131) for every target. With `--predict-years 2024` and the default gap, context ends at 2023 winter. Use `--context-gap-years 2` to stop at 2022.

### 1. Non-ML Baselines

**Location:** `baselines/naive.py`

These methods have no learned parameters.

**Persistence:** Returns the last observation in the input window, i.e. timestep `min(target_idx - 1, context_end_idx)`. It does not fall back to earlier timesteps: pixels that are nodata at that timestep are excluded.

**Previous Year:** Returns the same season one year before the target (`target_idx - 4`). If that timestep lies after `context_end_idx`, it steps back in whole years until it reaches the context. Like Persistence, it has no nodata fallback.

**Climatology:** Per-pixel mean of all same-season observations up to `context_end_idx`. Nodata observations are skipped.

### 2. ML Baselines

**Location:** `baselines/statistical.py`, `baselines/lightgbm/`

**Linear Trend:** Fits a per-pixel ordinary least squares (OLS) regression on same-season history and extrapolates to the target year. A pixel needs at least 2 valid same-season observations. The fit uses sufficient statistics, so it runs vectorized over the batch.

**Kalman Filter:** A per-pixel local linear trend model with state vector [level, trend]. The forward pass runs over same-season observations and skips the update step at nodata timesteps. The final filtered state is extrapolated to the target year. Process and observation noise are configurable (defaults: level 0.01, trend 0.001, observation 0.05).

**LightGBM:** A gradient-boosted tree model trained on per-pixel features. It is non-spatial: each pixel is treated independently.

Features (9 per pixel):
- Last 5 same-season NDVI values (most recent first)
- Mean of all same-season values
- Standard deviation of all same-season values
- Linear trend slope (OLS)
- Season ID (0-3)

Training samples up to `--max-pixels` pixels from the training shards (default 1M). It uses GBDT with 31 leaves, learning rate 0.05, feature and bagging fraction 0.8, and 500 boosting rounds (`train_lightgbm` in `baselines/lightgbm/train.py`). Early stopping (20 rounds without validation-MAE improvement) is enabled only when `--val-dir` is given. The model is saved as `model.txt` in `--output-dir`. LightGBM is installed from `requirements-extra.txt`.

```bash
python baselines/lightgbm/train.py \
    --train-dir /path/to/train \
    --val-dir /path/to/val \
    --output-dir outputs/lightgbm \
    --predict-years 2022
```

### 3. DL Baselines

**ConvLSTM** (`baselines/convlstm/`): An encoder-decoder built from Convolutional LSTM layers (Shi et al., 2015). It processes the NDVI time series while preserving spatial structure.
- Input: (B, T, 1, H, W)
- Multi-layer ConvLSTM encoder (default hidden dims [32, 64])
- The final hidden state passes through a convolutional decoder
- Output: (B, 1, H, W)

**PredRNN** (`baselines/predrnn/`): PredRNN (Wang et al., 2017) extends ConvLSTM with Spatiotemporal LSTM (ST-LSTM) cells. Besides the standard temporal memory (C), these cells carry a spatiotemporal memory (M) that flows bottom-up through the layers at each timestep (zigzag pattern). The default hidden dims are [64, 64, 64].

**Training (both):** L1 loss on valid pixels, AdamW, a cosine LR schedule with 1-epoch warmup, mixed precision on CUDA, and ShardPrefetchLoader with multi-GPU support (`--multi-gpu`, `--spawn`). `--epochs` defaults to 50 in both scripts. The paper trains both models for 20 epochs, so the examples pass `--epochs 20`.

```bash
python baselines/convlstm/train.py \
    --train-dir /path/to/train \
    --val-dir /path/to/val \
    --output-dir outputs/convlstm \
    --epochs 20 --batch-size 16

python baselines/predrnn/train.py \
    --train-dir /path/to/train \
    --val-dir /path/to/val \
    --output-dir outputs/predrnn \
    --epochs 20 --batch-size 16
```

---

## Additional baselines in the code (not reported in the paper)

**Recent Climatology** (`recent_climatology`, `baselines/naive.py`): Per-pixel mean of the last 5 same-season timesteps up to `context_end_idx`. Nodata observations among those five are skipped.

**Harmonic Regression** (`harmonic_regression`, `baselines/statistical.py`): A CCDC-style per-pixel weighted OLS fit of intercept + linear trend + 2 harmonics (period of 4 seasons). It is fit to all timesteps up to `context_end_idx`, not only to same-season ones, and extrapolated to the target.

**Prithvi-EO-2.0** (`baselines/prithvi/`): Fine-tunes the Prithvi-EO-2.0 300M foundation model (`--model-name prithvi_eo_v2_300`), loaded through TerraTorch.
- Input adapter: a 1x1 convolution projects the 1-channel NDVI to the 6 channels Prithvi expects
- Temporal subsampling: the input sequence is reduced to `--num-input-frames` evenly spaced frames (default 4)
- Center crop to 224x224
- Convolutional prediction head on the encoder output
- Training: differential learning rates (`--backbone-lr` 1e-5, `--head-lr` 1e-3), optional backbone freezing with the last N blocks unfrozen, and gradient accumulation
- Requires `terratorch` (`requirements-extra.txt`)

```bash
python baselines/prithvi/train.py \
    --train-dir /path/to/train \
    --val-dir /path/to/val \
    --output-dir outputs/prithvi \
    --freeze-backbone --unfreeze-last-n 4 \
    --gradient-accumulation 4
```

---

## Evaluation Scripts

All scripts read shards in the `shard_*.safetensors` format, each with a `.files.json` sidecar. In split mode they expect this layout under `--shard-root`:

```
<shard-root>/
├── val_iid/val_iid/            # val_iid; first part of ood_t
├── ood_t_extra/ood_t_extra/    # second part of ood_t
└── ood_spatial/ood_spatial/    # ood_s and ood_st
```

### Non-ML and Statistical Baselines

`eval/evaluate_baselines.py` runs any subset of the seven methods in `baselines/` (`--methods`, default: all of them).

```bash
# Paper baselines on OOD-t, scoring 2023 only
python eval/evaluate_baselines.py \
    --split ood_t \
    --shard-root /path/to/shards \
    --predict-years 2023 \
    --methods persistence previous_year climatology linear_trend kalman_filter

# Add the per-land-cover breakdown
python eval/evaluate_baselines.py \
    --split ood_s \
    --shard-root /path/to/shards \
    --landcover-json data/landcover_stats.json

# Re-aggregate saved per-sample stats without reprocessing shards
python eval/evaluate_baselines.py \
    --load-stats outputs/baselines/ood_s_2022/nonml_per_sample_stats.json \
    --landcover-json data/landcover_stats.json \
    --output-dir outputs/baselines/ood_s_2022
```

| Argument | Description |
|----------|-------------|
| `--split` | `val_iid`, `ood_t`, `ood_s` or `ood_st`. Sets the shard directories and the default predict years |
| `--shard-root` | Root directory with the split folders (required with `--split`) |
| `--methods` | Methods to run: `persistence`, `previous_year`, `climatology`, `recent_climatology`, `linear_trend`, `kalman_filter`, `harmonic_regression` (default: all) |
| `--predict-years` | Target years. Overrides the split default |
| `--context-gap-years` | Years between the end of the context and the first target year (default: 1). See [Context window](#context-window) |
| `--num-workers` | Number of parallel worker processes, one shard each (default: 3; use 1 for sequential) |
| `--device` | `cpu` (default), `cuda` or `cuda:N`. With `cuda` and more than one worker, shards are assigned round-robin to the visible GPUs |
| `--output-dir` | Output directory (default `outputs/baselines`; in split mode this becomes `outputs/baselines/{split}_{years}`) |
| `--max-samples` | Limit the number of samples for quick tests |
| `--landcover-json` | Path to `landcover_stats.json` for the per-land-cover breakdown |
| `--load-stats` | Re-aggregate an existing `nonml_per_sample_stats.json` (skips shard processing) |
| `--eval-dir` | Legacy mode: a directory of shards, or a single shard, used instead of `--split` |
| `--predict-year` | Legacy mode: a single target year |
| `--forecast-horizon` | Legacy mode: predict the last N timesteps (default 4). Ignored when a predict year is set |

Output files:

| File | Contents |
|------|----------|
| `nonml_metrics.json` | Per method: `overall`, `per_season` (`spring`, `summer`, `fall`, `winter`), and `per_landcover` when `--landcover-json` is given. Each block has `mae`, `r2`, `nse`, `bias`, `n_valid_pixels`, `n_samples` |
| `nonml_per_sample_stats.json` | Per method, one record per (sample, target timestep): identifiers, `n_valid`, per-record `mae`/`r2`/`nse`, and sufficient statistics (`sum_pred`, `sum_target`, `sum_pred_sq`, `sum_target_sq`, `sum_pred_target`, `sum_abs_error`, `sum_sq_error`) |
| `nonml_comparison.png` | Bar chart comparing methods on MAE, r^2, NSE and bias |

`--load-stats` rewrites only `nonml_metrics.json` and `nonml_comparison.png`.

**Land cover.** `landcover_stats.json` holds per-tile ESA WorldCover fractions. It ships with the dataset download (place it at `data/landcover_stats.json`), and `data/download/download_landcover_stats.py` can regenerate it from Earth Engine. Samples are grouped by their `dominant_class`. Samples missing from the file are reported as `Unknown`.

| Code | Name |
|------|------|
| 10 | Tree cover |
| 20 | Shrubland |
| 30 | Grassland |
| 40 | Cropland |
| 50 | Built-up |
| 60 | Bare/sparse |
| 70 | Snow/ice |
| 80 | Water |
| 90 | Wetland |
| 95 | Mangroves |
| 100 | Moss/lichen |

### LightGBM

```bash
python eval/evaluate_lightgbm.py \
    --model-path outputs/lightgbm/model.txt \
    --split ood_s \
    --shard-root /path/to/shards
```

Optional flags: `--predict-years`, `--context-gap-years` (default 1), `--batch-size` (default 64), `--device` (for feature extraction; defaults to CUDA when available), `--num-workers` (default 1), and `--output-dir` (default `outputs/lightgbm/eval/{split}_{years}`). The script writes `metrics.json`, `per_sample_stats.json` and `comparison.png`, using the same metric schema as `nonml_metrics.json` without `per_landcover`.

### DL Baselines (ConvLSTM, PredRNN, Prithvi)

```bash
python eval/evaluate_dl_baselines.py \
    --model convlstm \
    --checkpoint outputs/convlstm/best.pt \
    --split ood_t --predict-years 2023 \
    --shard-root /path/to/shards

# PredRNN: --hidden-dims must match training (the eval default is ConvLSTM's 32 64)
python eval/evaluate_dl_baselines.py \
    --model predrnn \
    --checkpoint outputs/predrnn/best.pt \
    --hidden-dims 64 64 64 \
    --split ood_s \
    --shard-root /path/to/shards
```

Optional flags: `--predict-years`, `--context-gap-years` (default 1), `--batch-size` (default 8), `--device`, and `--output-dir` (default `outputs/{model}/eval/{split}_{years}`). The models receive timesteps `0..context_end_idx` as input and take no target conditioning, so every target timestep is scored against the same prediction. Only target nodata pixels are excluded. Outputs are `metrics.json`, `per_sample_stats.json` and `comparison.png`.

### SeasonalViT

Per-sample evaluation on a split goes through `FinetuningTrainer.validate()` in `train/finetune.py`. It runs on the validation set during fine-tuning and is also called by:

- `scripts/eval_ablations_on_split.py --runs-dir RUNS --eval-dir SHARD_DIR --predict-years YEAR [--only NAME ...]`: evaluates `RUNS/<name>/best.pt` for each run directory listed in the script's `ABLATIONS` table and writes a TSV with `val_mae`, `val_rmse`, `val_r2`, `val_R2_det`, `val_bias`, `val_anchor_mae` and `val_delta_abs_mean`.
- The horizon scripts described in [`docs/specific/forecasting_horizon.md`](specific/forecasting_horizon.md#horizon-evaluation).

For each sample, `validate()` draws one target from the configured predict years and feeds the model every timestep before that target (see `docs/specific/forecasting_horizon.md` for how history is truncated within a batch).

`scripts/evaluate.py --checkpoint CKPT --eval-dir SHARD_DIR --predict-years YEAR` evaluates a single checkpoint through the same `validate()` path, loading the model architecture and loss settings saved in the checkpoint. Optional flags: `--per-season` also scores each target season of the predict years on its own (pinned with `predict_target_indices`); `--visualize` saves a grid of example predictions; `--output-dir`, `--batch-size`, `--limit`. It writes `metrics.json` (overall and per-season metrics plus the evaluation settings) and, with `--visualize`, `predictions_grid.png`.

---

## File Structure

```
baselines/
├── naive.py                    # Persistence, Previous Year, Climatology, Recent Climatology
├── statistical.py              # Linear Trend, Kalman Filter, Harmonic Regression
├── lightgbm/
│   ├── model.py                # Feature extraction + predictor wrapper
│   └── train.py                # Training entry point
├── convlstm/
│   ├── model.py                # ConvLSTM cell + encoder-decoder
│   └── train.py                # Training entry point
├── predrnn/
│   ├── model.py                # ST-LSTM cell + PredRNN architecture
│   └── train.py                # Training entry point
└── prithvi/
    ├── model.py                # TerraTorch loading + prediction head
    └── train.py                # Fine-tuning entry point

eval/
├── evaluate_baselines.py       # Non-ML + statistical baselines; shared split config and aggregation
├── evaluate_lightgbm.py        # LightGBM evaluation
└── evaluate_dl_baselines.py    # ConvLSTM, PredRNN, Prithvi evaluation

scripts/
├── evaluate.py                 # SeasonalViT: one checkpoint via FinetuningTrainer.validate()
├── eval_ablations_on_split.py  # SeasonalViT: FinetuningTrainer.validate() on a split
├── eval_horizons.py            # SeasonalViT: horizons h in {1, 2, 4, 8}
└── eval_horizons_fast.py       # SeasonalViT: horizons h = 1..8, fixed input window

train/
└── metrics.py                  # MAE, RMSE, r^2, NSE, bias; PerSampleMetricsTracker
```
