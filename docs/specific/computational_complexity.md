# Computational Complexity

Training time, parameter counts, and GFLOPs for SeasonalViT.

## Summary

> SeasonalViT has **80.65 M parameters** and requires **1419.65 GFLOPs** for a single forward pass on a 256x256 input with T=136 timesteps (the input length used for the measurement). Pretraining (40 epochs, masked NDVI reconstruction) took **79.6 hours** wall clock (159.2 GPU-hours) and fine-tuning for 2022 forecasting (20 epochs) took **31.7 hours** (63.4 GPU-hours), both on 2x NVIDIA L40S GPUs (46 GB) using PyTorch 2.6.0, CUDA 12.4, and bf16 mixed precision (`torch.amp.autocast` with GradScaler disabled). Per-component parameter counts and the full training configuration are reported in Appendix A of the paper.

## Table A — Model size & per-component compute

Single forward pass, batch size 1, input 1 x 136 x 256 x 256 (NDVI). Measured with `fvcore.nn.FlopCountAnalysis` (`scripts/measure_complexity.py`).

| Component                              | Params (M) | Share |
|---------------------------------------- |-----------:|------:|
| Spatial encoder (Swin-T, patch 4, win 7) |     27.52 | 34.1% |
| Seasonal positional encoding           |      0.72 |  0.9% |
| Location encoder                       |      1.23 |  1.5% |
| Dual-scale temporal transformer (6 layers, 12 heads, d=768) | 46.66 | 57.9% |
| Observation-anchored decoder           |      4.52 |  5.6% |
| **TOTAL (SeasonalViT, fine-tuning)**   | **80.65** | 100.0% |
| **TOTAL (pretraining model)**[^1]      | **75.64** |       |

[^1]: Pretraining model replaces location encoder + decoder with a 0.69 M reconstruction head; the spatial encoder, temporal encoding, and temporal transformer are shared.

**Single forward pass FLOPs:**

| Model                | Input shape       | GFLOPs   |
|----------------------|-------------------|---------:|
| SeasonalViT (fine-tune, T=136) | 1 x 136 x 256 x 256 | 1419.65 |
| Pretraining model (T=140)      | 1 x 140 x 256 x 256 | 1456.78 |

GFLOPs are operation counts from `fvcore.nn.FlopCountAnalysis` on a single forward pass with batch size 1; counts are independent of numerical precision (the same ops are dispatched whether the run is fp32, bf16, or fp16). They include all standard ops traced by fvcore (matmul, conv, attention). fvcore counts one fused multiply-add as one FLOP, so these GFLOPs figures are equivalent to GMACs.

## Table B — Training cost

| Stage      | Hardware       | Batch (per GPU) | Epochs | Wall clock | GPU-hours | LR (init) | Mixed precision |
|------------|----------------|----------------:|-------:|-----------:|----------:|----------:|----------------:|
| Pretrain   | 2x L40S 46 GB  |         64 / 32 |     40 |    79.6 h  |   159.2   |   1e-4    |       bf16      |
| Fine-tune  | 2x L40S 46 GB  |              14 |     20 |    31.7 h  |    63.4   |   5e-5    |       bf16      |
| **Total**  |                |                 |        | **111.3 h**| **222.6** |           |                 |

- **Pretrain wall clock**: pretraining ran in two 20-epoch segments (epochs 1-20, ~19.5 h; epochs 21-40, ~60 h), resumed from the epoch-20 checkpoint with the same masking and LR schedule.
- **Fine-tune wall clock**: a single 20-epoch run. Spatial encoder frozen; trainable parameters: 53.13 M of 80.65 M total.

## Table C — Trainable parameters at fine-tuning

| Configuration                        | Trainable (M) | Frozen (M) |
|--------------------------------------|--------------:|-----------:|
| Full fine-tune                       |        80.65  |       0.00 |
| **Spatial encoder frozen (used)**    |    **53.13**  |  **27.52** |
| Encoder fully frozen (decoder only)  |         5.75  |      74.90 |

## Table D — Hyperparameters (for the appendix / reproducibility)

These are the values used for the paper runs. Some differ from the code defaults in `model/config.py`, so they must be passed as CLI flags to reproduce the runs:

| Stage | Setting | Code default | Paper run | Flag |
|-------|---------|-------------:|----------:|------|
| Pretraining | `learning_rate` | 1.5e-4 | 1e-4 | `--lr 1e-4` |
| Pretraining | `batch_size` (per GPU) | 16 | 64 (epochs 1-20), 32 (epochs 21-40) | `--batch-size` |
| Pretraining | `max_epochs` | 20 | 40 | `--epochs 40` |
| Fine-tuning | `batch_size` (per GPU) | 16 | 14 | `--batch-size 14` |
| Fine-tuning | `delta_supervision_weight` | 0.1 | 0.2 | `--delta-supervision-weight 0.2` |
| Fine-tuning | `freeze_spatial` | False | True | `--freeze-spatial` |
| Fine-tuning | `temporal_subset_years` | 12 | 0 (all input years) | `--temporal-subset-years 0` |

### Architecture

| Hyperparameter                | Value          |
|-------------------------------|----------------|
| Input image size              | 256 x 256      |
| Number of timesteps (max)     | 140 (35 yr x 4 seasons) |
| Spatial patch grid            | 8 x 8 = 64 patches per timestep |
| Embedding dimension           | 768            |
| Spatial encoder backbone      | Swin-T (timm `swin_tiny_patch4_window7_224`) |
| Temporal transformer depth    | 6 layers       |
| Temporal transformer heads    | 12             |
| Temporal MLP ratio            | 4.0            |
| Temporal dropout              | 0.1            |
| Attention pooling heads (decoder) | 8          |
| Decoder hidden dim            | 384            |
| Anchor mode                   | same-season, last 5 yr mean |
| Sensor era boundaries (year idx) | 9, 23, 32   |

### Pretraining

| Hyperparameter                | Value          |
|-------------------------------|----------------|
| Masking strategy              | efficient (Levels 2 + 3) |
| Temporal mask ratio (Level 2) | 0.30           |
| Image mask probability (Level 3) | 0.10        |
| Optimizer                     | AdamW          |
| LR                            | 1e-4 (cosine, min 1e-6) |
| Weight decay                  | 0.05           |
| Warmup epochs                 | 1              |
| Gradient clipping             | 1.0            |
| Temporal subset (years/sample)| 12             |
| Per-GPU batch                 | 32             |
| Effective batch (2 GPUs)      | 64             |

### Fine-tuning (2022 forecasting)

| Hyperparameter                | Value          |
|-------------------------------|----------------|
| Predict years                 | [2022]         |
| Forecast strategy             | use all history before target, indices 128-131 |
| Optimizer                     | AdamW          |
| LR (encoder)                  | 5e-5           |
| LR (decoder + location)       | 5e-4 (10x mult)|
| Weight decay                  | 0.01           |
| Warmup epochs                 | 1              |
| Gradient clipping             | 1.0            |
| Per-GPU batch                 | 14             |
| Effective batch (2 GPUs)      | 28             |
| Frozen components             | Spatial encoder (Swin-T) |
| Temporal subset (years/sample)| disabled (all input years) |
| Loss weights                  | corr 0.5, delta supervision 0.2, delta L2 0.0 |

## How the numbers were obtained

- **Params + GFLOPs**: `python scripts/measure_complexity.py` (uses `fvcore.nn.FlopCountAnalysis`). fvcore counts dispatched ops, not kernel time, so the FLOP total is independent of numerical precision and of CPU vs GPU execution. The script ran on CPU at default fp32 only because that is what `fvcore` defaults to; the same value applies to the bf16-on-GPU training run.
- **Wall clock**: measured wall-clock time of each training stage.
- **Param totals cross-checked** against in-log prints: pretrain log reports 75,636,412 params; fine-tune log reports 80,646,110 / 53,126,750 trainable. Both match the script output to within ~50 K (mask token and unregistered buffers).
