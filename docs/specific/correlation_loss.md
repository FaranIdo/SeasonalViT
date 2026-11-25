# Correlation Loss for r² Optimization

Per-sample Pearson correlation loss added to the fine-tuning objective to directly optimize for spatial pattern quality, measured as r² (squared Pearson correlation).

---

## Motivation

The Smooth L1 (Huber) reconstruction loss computes errors independently per pixel. It provides no gradient signal about spatial relationships within a sample. A model can minimize MAE by predicting near-mean values across a tile while failing to capture spatial patterns (vegetation boundaries, land cover transitions). This leads to low r² despite reasonable MAE.

Specifically, Smooth L1 with `beta=0.1` uses L1 behavior for errors above 0.1 NDVI. Since meaningful spatial variation within a 256x256 tile often produces errors in the 0.1--0.5 range, most informative spatial structure falls in the L1 regime where gradient magnitude is constant regardless of error size. The model has no extra incentive to fix high-contrast spatial features that matter most for correlation.

## How It Works

For each sample in a batch, we compute the Pearson correlation `r` between the predicted and target spatial maps (on valid pixels only), then penalize `1 - r`:

```
L_corr = mean_over_batch(1 - pearson_r(pred_b, target_b))
```

The total fine-tuning loss (`train/losses.py:smooth_l1_forecast_loss`) is:

```
L_total = L_smooth_l1
        + correlation_loss_weight  * L_corr
        + delta_supervision_weight * L_delta_sup
        + delta_reg_weight         * L_delta_reg
```

- `L_smooth_l1`: Smooth L1 between prediction and target (`beta=0.1`), on valid pixels.
- `L_delta_sup`: L1 between the predicted delta and the true delta `(target - anchor)`, both divided by `delta_std`. Pixels where the target or anchor is nodata are excluded. `delta_std` is set with `--delta-std`; the default `0` auto-estimates it before training as 1.4826 × the median absolute deviation of `(target - anchor)` on validation batches.
- `L_delta_reg`: mean squared delta (L2 penalty). `delta_reg_weight` defaults to 0, so this term is off unless set with `--delta-reg-weight`.

The correlation loss gradient pushes each pixel in proportion to how much the target deviates from its spatial mean. High-contrast areas (vegetation boundaries, water bodies) receive stronger gradients. This is exactly the signal missing from per-pixel losses.

## Usage

`correlation_loss_weight` defaults to 0.5 in `FinetuningConfig`, so the correlation term is on unless overridden.

```bash
# Default weight (0.5)
python scripts/finetune.py \
    --train-dir DATA/train --val-dir DATA/val \
    --pretrained MODEL.pt

# Disable the correlation term
python scripts/finetune.py \
    --train-dir DATA/train --val-dir DATA/val \
    --pretrained MODEL.pt \
    --corr-loss-weight 0
```

## Hyperparameter Guide

| Parameter | Code default (`FinetuningConfig`) | Paper run | Flag |
|-----------|-----------------------------------|-----------|------|
| `correlation_loss_weight` | `0.5` | `0.5` | `--corr-loss-weight` |
| `delta_supervision_weight` | `0.1` | `0.2` | `--delta-supervision-weight` |
| `delta_reg_weight` | `0.0` | | `--delta-reg-weight` |
| `delta_std` | `0.0` (auto-estimate) | | `--delta-std` |

Blank cells are not reported for the paper run. Other correlation weights were not evaluated for the paper; larger weights let the correlation term steer optimization more strongly relative to the per-pixel Smooth L1 term.

## Interaction with Architecture

SeasonalViT uses an observation-anchored decoder: `prediction = clamp(anchor + delta, -1, 1)`. The anchor is the per-pixel mean of the last `anchor_years` (default 5) same-season observations in the input, and it already provides strong spatial structure. The correlation loss primarily constrains the delta to make spatially coherent corrections rather than adding noise.

Delta regularization is off by default (`delta_reg_weight = 0`). Without any regularization, the delta can become noisy and degrade r² even while reducing MAE. The correlation loss serves as an alternative form of regularization that is directly aligned with spatial pattern quality.

## Implementation Details

- Computed per-sample, then averaged across the batch
- Samples with zero-variance prediction or target are excluded (avoids division by zero)
- Uses float32 upcast for numerical stability
- Respects nodata masks (only valid pixels contribute to correlation)
- Negligible compute overhead: O(B x H x W) per batch, dwarfed by the model forward pass

The loss is implemented in `train/losses.py` (`per_sample_correlation_loss()`).

## Monitoring

TensorBoard scalars:
- `train/corr_loss_step`: Per-step correlation loss (raw, before weighting)
- `train/corr_loss_epoch`: Epoch-averaged correlation loss
- `val/val_r2`: Per-sample averaged r² (squared Pearson correlation) on the validation set (the target metric)
- `val/val_R2_det`: Per-sample averaged coefficient of determination (R² = 1 - SS_res / SS_tot), logged for reference

A decreasing `corr_loss` and increasing `val_r2` during training confirms the loss is working as intended.
