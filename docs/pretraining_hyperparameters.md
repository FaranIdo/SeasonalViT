# Pretraining Hyperparameters

This document describes the hyperparameters used for SeasonalViT pretraining and the rationale behind each choice.

## Configuration Location

All pretraining hyperparameters are defined in `model/config.py` in the `PretrainingConfig` dataclass. `scripts/pretrain.py` exposes a subset of them as CLI flags; the rest are config-only.

## Target Normalization

| Parameter | Default | Description |
|-----------|---------|-------------|
| `normalize_target` | `True` | MAE-style per-patch target normalization |

### Rationale

Following He et al. (2022) "Masked Autoencoders Are Scalable Vision Learners", reconstruction targets are normalized per patch to zero mean and unit variance before computing MSE loss.

**Why this matters:** Without normalization, the model quickly learns to predict the per-patch mean NDVI value (which minimizes raw MSE), then plateaus because gradients for spatial detail within patches vanish. With normalization, the loss function rewards predicting *relative spatial patterns* within each patch, providing sustained learning signal.

**Effect on loss values:** Normalized loss values are not directly comparable to raw MSE. Initial loss will be higher (~1.0, since targets have unit variance), but the model should show continued improvement beyond the first few epochs.

Use `--no-normalize-target` to disable for comparison experiments.

### Normalization Across Training Stages

The table below summarizes what normalization applies at each stage:

| Stage | Input NDVI | Input Projection | Target Normalization |
|-------|-----------|-------------------|----------------------|
| Pretraining | Raw [-1, 1] | Learned Conv2d(1→3) | Per-patch zero-mean/unit-var |
| Fine-tuning | Raw [-1, 1] | Learned Conv2d(1→3) (from pretrained) | None (anchor+delta) |

**Key design decisions:**

1. **No input scaling.** Raw NDVI values in [-1, 1] are never rescaled to [0, 1]. Instead, a learned `Conv2d(1, 3, 1×1)` projection maps the single-channel NDVI input to 3 channels for the Swin-T backbone. It is initialized as an affine transform (`scale = imagenet_std / ndvi_std = 0.904`, `bias = imagenet_mean - scale × ndvi_mean = 0.133`) that maps NDVI statistics to ImageNet-like statistics, then updated by gradient descent during training. See `model/encoders/spatial.py:_init_input_proj`.

2. **Pretraining: per-patch target normalization.** Reconstruction targets are normalized per patch to zero mean and unit variance (MAE-style). This prevents the model from simply predicting mean NDVI per patch and encourages learning spatial detail. Only targets are normalized; predictions are compared directly against the normalized targets.

3. **Fine-tuning: no target normalization.** The observation-anchored decoder (`model/decoder.py:ObservationAnchoredDecoder`) handles scale differently. The anchor is the per-pixel mean of the last `anchor_years` (default 5) same-season observations in the input, excluding nodata (`model/seasonal_vit.py`). The decoder predicts a bounded delta: `clamp(delta_scale × MLP output, -1, 1)`, where `delta_scale` is a fixed (non-learnable) buffer set to 0.5. The delta is upsampled to full resolution and the final prediction is `clamp(anchor + delta, -1, 1)`. This replaces statistical normalization with a structural prior: the anchor provides absolute NDVI levels, and the model only needs to predict bounded corrections.

4. **Why the approaches differ.** Per-patch normalization suits reconstruction because the goal is learning spatial patterns regardless of absolute NDVI level. Anchor-based prediction suits forecasting because the goal is predicting temporal change from a known reference. The decoder is trained from scratch during fine-tuning (separate from the pretrained encoder), so it adapts to the encoder's feature space without needing the same normalization scheme.

## Temporal Subsampling

| Parameter | Default | Flag | Description |
|-----------|---------|------|-------------|
| `temporal_subset_years` | `12` | `--temporal-subset-years` | Complete years (4 seasons each) sampled per training forward pass. `0` uses all 35 years. |
| `val_batch_size` | `None` | `--val-batch-size` | Validation batch size. `None` auto-derives it. |

With the default of 12, each training sample is reduced from 140 to 48 timesteps (12 years × 4 seasons). The same year subset is used for every sample in a batch, and each sampled year keeps all four seasons. This reduces per-sample memory and allows a larger batch size.

Validation always uses the full 140-timestep sequence. Because validation then needs more memory per sample, `effective_val_batch_size` auto-derives the validation batch when `val_batch_size` is `None`: `max(1, batch_size // 4)` when `temporal_subset_years > 0`, otherwise `batch_size`.

## Masking Strategy

| Parameter | Default | Flag | Description |
|-----------|---------|------|-------------|
| `masking_strategy` | `"efficient"` | config only | Which masking approach to use |
| `temporal_mask_ratio` | `0.3` | `--temporal-mask-ratio` | Ratio of timesteps to mask (Level 2) |
| `image_mask_prob` | `0.1` | config only | Probability of additional timestep masking (Level 3) |
| `patch_mask_ratio` | `0.75` | config only | Ratio of patches to mask within images (Level 1, full mode only) |

The paper run used the defaults: `"efficient"` masking, `temporal_mask_ratio` 0.3, `image_mask_prob` 0.1.

### Rationale

**Efficient vs Full Strategy:**
- `"efficient"`: Masks timesteps BEFORE encoding. Only visible timesteps pass through Swin-T, so encoder cost scales with the visible fraction. Loses spatial patch reconstruction but retains temporal learning.
- `"full"`: Encodes ALL timesteps, then applies masks. Slower but includes Level 1 spatial patch masking for learning texture and edge features. It can only be selected by setting `masking_strategy="full"` in `PretrainingConfig` (no CLI flag) and was not used for the paper.

**Visible timestep count:** The efficient mask uses a fixed number of visible timesteps per batch (`train/masking.py:generate_efficient_temporal_mask`):

```
num_visible = T - int(T × temporal_mask_ratio) - int(T × image_mask_prob)
```

With the defaults and temporal subsampling (T = 48), this gives 48 - 14 - 4 = 30 visible and 18 masked timesteps (~37% masked). Validation runs on the full sequence (T = 140): 140 - 42 - 14 = 84 visible timesteps.

**Temporal Mask Ratio (0.3):**
- 30% masking leaves more temporal context, making the reconstruction task easier initially
- Gentler masking helps the model learn meaningful temporal patterns
- Use `--temporal-mask-ratio` to override

**Image Mask Prob (0.1):**
- Rare additional masking forces long-range temporal reasoning
- Helps the model learn to handle completely missing years
- Combined with `temporal_mask_ratio`, gives ~37% total masking

## Optimization and Batching

| Parameter | Code default (`PretrainingConfig`) | Paper run | Flag |
|-----------|-----------------------------------|-----------|------|
| `learning_rate` (peak) | `1.5e-4` | `1e-4` | `--lr` |
| `min_lr` | `1e-6` | `1e-6` | config only |
| `weight_decay` | `0.05` | `0.05` | config only |
| `warmup_epochs` | `1` | `1` | config only |
| `max_epochs` | `20` | `40` | `--epochs` |
| `batch_size` (per GPU) | `16` | `64` in epochs 1-20, `32` in epochs 21-40 (2 GPUs) | `--batch-size` |
| `temporal_subset_years` | `12` | `12` | `--temporal-subset-years` |
| `val_batch_size` | `None` (auto) | | `--val-batch-size` |
| `max_grad_norm` | `1.0` | | config only |
| `gradient_accumulation_steps` | `1` | | config only |
| `train_num_workers` / `val_num_workers` | `2` / `2` | | config only |

Blank cells are not reported for the paper run.

### Rationale

**Learning Rate:**
- Based on SatMAE (1e-4 for batch 256)
- Lower than original ViT (3e-4), which used batch 4096
- High learning rates with small batches can cause instability and val loss divergence

**Minimum Learning Rate (1e-6):**
- Floor for cosine decay so the LR never reaches zero
- Without it, the last epochs run at near-zero LR and waste compute

**Weight Decay (0.05):**
- Standard for transformer pretraining (used in MAE, BEiT, etc.)
- AdamW is applied to all model parameters with a single weight decay value (`train/pretrain.py`)

**Batch Size:**
- `batch_size` is per GPU; with DDP the effective batch is `batch_size × num_gpus`
- Steps per epoch = training samples / effective batch. For example, the paper's 157,523-sample train split at 32 × 2 GPUs gives ~2.5K steps per epoch

**Workers (2 train, 2 val):**
- Low worker count reduces memory overhead from prefetch buffers
- With shard-based loading, IO is dominated by shard reads, not per-sample loading
- Increase if GPU utilization is low due to data loading

### Learning Rate Schedule

The scheduler (`train/pretrain.py`) uses **linear warmup** from 0 to `learning_rate` over `warmup_epochs`, followed by **cosine decay to `min_lr`** over the remaining steps. The schedule is computed per step from `max_epochs`, so the decay stretches automatically when the number of epochs changes; `learning_rate` and `min_lr` do not need adjusting.

```
LR
peak |      /‾‾‾‾‾\
     |     /       \
     |    /         \__
     |   /             \___
min  |__/__________________‾‾‾  ← min_lr floor
     warmup     cosine decay →
```

## Gradient Handling

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_grad_norm` | `1.0` | Maximum gradient norm for clipping |
| `gradient_accumulation_steps` | `1` | Steps to accumulate before update |

### Rationale

**Gradient Clipping (1.0):**
- Prevents exploding gradients during early training
- Standard value for transformer training

**Gradient Accumulation (1):**
- No accumulation by default (update every step)
- Increase to simulate larger batch sizes on limited GPU memory
- Example: `gradient_accumulation_steps=4` with `batch_size=16` simulates a per-GPU batch of 64

## Checkpointing

| Parameter | Default | Description |
|-----------|---------|-------------|
| `checkpoint_every_n_epochs` | `1` | Save checkpoint frequency |

Saving every epoch allows resuming from any point if training is interrupted (`--resume`).
