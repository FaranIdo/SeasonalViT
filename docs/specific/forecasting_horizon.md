# Forecasting Horizon: Variable Target Training

This document describes how SeasonalViT selects forecasting targets and input history during fine-tuning, how the decoder is conditioned on the target, and how forecast horizons are evaluated.

---

## Target Selection

**The model predicts 1 timestep per forward pass. The target timestep is sampled per sample.**

The decoder is conditioned on (target_year, target_season), so one decoder can predict any timestep. This follows DiffusionSat's approach of specifying targets through metadata conditioning.

`FinetuningTrainer._sample_target_indices` (`train/finetune.py`) picks each sample's target uniformly from a candidate list. The list comes from the first of these that is set (same priority as `FinetuningConfig.get_target_indices()` in `model/config.py`):

1. **`predict_target_indices`**: explicit absolute timestep indices, e.g. `[132]` = 2023 spring or `[139]` = 2024 winter. It is set only in config (no CLI flag). `scripts/eval_horizons.py` uses it to pin one target per horizon.
2. **`predict_years`**: all four seasons of the listed years. **The default is `[2022]`**, so fine-tuning draws its targets from timesteps 128-131. `scripts/finetune.py --predict-years` overrides the list.
3. **`forecast_horizon`** (default 8): the last `forecast_horizon` timesteps. This applies only when both `predict_target_indices` and `predict_years` are unset (`predict_years=None` in config).

> **Warning:** With the default `forecast_horizon=8`, the fallback in step 3 samples 2023-2024 targets (indices 132-139). Those are the OOD-t / OOD-st test years, so this mode puts test-period targets into training.

### Training Step (`train/finetune.py`, simplified)

```python
target_idx = self._sample_target_indices(B, T)                     # (B,)
target, target_season, target_year, target_nodata = (
    self._extract_target_from_batch(images, season_ids, year_indices, nodata_mask, target_idx)
)
input_images, input_season_ids, input_year_indices = (
    self._select_input_timesteps(images, season_ids, year_indices, target_idx)
)
# Anchor is computed on the full input window, before subsampling
precomputed_anchor = self._maybe_compute_anchor(input_images, input_season_ids, target_season)
# Training only: keep `temporal_subset_years` complete years (anchor year always included)
input_images, input_season_ids, input_year_indices = (
    self._apply_temporal_subsampling(input_images, input_season_ids, input_year_indices, target_season)
)
```

Validation (`validate()`) samples targets the same way, but with a fixed-seed generator (seed 42). It skips temporal subsampling, so the model sees the whole input window.

### Input History (`_select_input_timesteps`)

- **`use_all_history_before_target=True` (default):** each sample's window ends just before its own target. All samples in a batch share one window length: the shortest history in the batch (the smallest `target_idx`), rounded down to a multiple of 4. Samples with longer histories drop their earliest timesteps. With `predict_years=[2022]` (targets 128-131), every sample therefore gets the 128 timesteps immediately before its target.
- **`use_all_history_before_target=False`:** a fixed window of the first `((T - forecast_horizon) // 4) * 4` timesteps (132 for `forecast_horizon=8`), whatever the target.

### Configuration (`model/config.py`)

```python
@dataclass
class FinetuningConfig:
    forecast_horizon: int = 8                       # fallback only (see priority above)
    use_all_history_before_target: bool = True
    predict_years: Optional[List[int]] = field(default_factory=lambda: [2022])
    predict_target_indices: Optional[List[int]] = None   # takes priority over predict_years
    temporal_subset_years: int = 12                 # training-time year subsampling; 0 disables
```

### Model Forward Pass (`model/seasonal_vit.py`)

```python
def forward(
    self,
    images: torch.Tensor,           # (B, T, H, W)
    season_ids: torch.Tensor,       # (B, T)
    year_indices: torch.Tensor,     # (B, T)
    lat: torch.Tensor,              # (B,)
    lon: torch.Tensor,              # (B,)
    target_season: torch.Tensor,    # (B,) - target season to predict
    target_year: torch.Tensor,      # (B,) - target year to predict
    return_attention_analysis: bool = False,
    seq_positions: Optional[torch.Tensor] = None,
    precomputed_anchor: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    # Returns:
    #   'prediction': (B, 1, H, W) - predicted NDVI
    #   'delta': (B, 1, H, W)
    #   'anchor': (B, 1, H, W)
```

---

## Architecture: Target-Conditioned Decoder

| Component | Multi-step capable? | Notes |
|-----------|:-------------------:|-------|
| Swin-T Spatial Encoder | Yes | Produces spatial features independent of target |
| Temporal Transformer | Yes | Outputs rich features (B, T, N, D) that encode full history |
| Location Encoder | Yes | Location is constant across predictions |
| `target_season` + `target_year` | **Yes** | Conditions decoder on which timestep to predict |
| `TargetTemporalEncoder` | **Yes** | Encodes target (year, season) for conditioning |
| `TemporalAttentionPooling` | **Yes** | Query conditioned on target embedding |
| Decoder output | Single | Outputs (B, 1, H, W) for the specified target |

**Key insight:** Instead of a multi-output decoder, target conditioning tells a single-output decoder WHICH timestep to predict.

---

## Target Conditioning Implementation

### TargetTemporalEncoder (`model/encoders/temporal.py`)

Encodes (target_year, target_season) into a conditioning vector:

```python
class TargetTemporalEncoder(nn.Module):
    """Encodes target (year, season) for decoder conditioning.

    Uses same encoding strategy as SeasonalPositionalEncoding but with
    separate weights, allowing specialization for target vs input encoding.
    """
    def __init__(self, config: SeasonalViTConfig) -> None:
        super().__init__()
        d_model = config.embed_dim

        # 3 components: season_embed, season_cyclic, year encoding
        d_season_embed = d_model // 3
        d_season_cyclic = d_model // 3
        d_year = d_model - d_season_embed - d_season_cyclic

        self.season_embed = nn.Embedding(config.num_seasons, d_season_embed)
        self.season_cyclic_proj = nn.Linear(2, d_season_cyclic)
        self.n_year_freqs = 10
        self.year_proj = nn.Linear(self.n_year_freqs * 2, d_year)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, target_season, target_year) -> torch.Tensor:
        # Returns (B, embed_dim) conditioning vector
        ...
```

### Target-Conditioned Attention Pooling (`model/decoder.py`)

The decoder's attention pooling query is conditioned on the target:

```python
class TemporalAttentionPooling(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 8) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.condition_proj = nn.Linear(d_model, d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, target_condition=None, return_weights=False):
        query = self.query.expand(B * N, 1, D)

        # Condition the query on target
        if target_condition is not None:
            cond = self.condition_proj(target_condition)  # (B, D)
            cond = cond.unsqueeze(1).expand(B, N, D).reshape(B * N, 1, D)
            query = query + cond  # Conditioned query

        pooled, attn_weights = self.attn(query, x_flat, x_flat)
        pooled = self.norm(pooled.squeeze(1).view(B, N, D))
        ...
```

---

## Implemented Decoder (`model/decoder.py`)

`ObservationAnchoredDecoder` predicts NDVI as anchor + delta:

1. `target_encoder` (`TargetTemporalEncoder`) maps (target_season, target_year) to a (B, D) conditioning vector.
2. `temporal_pool` (`TemporalAttentionPooling`) pools the temporal features (B, T, N, D) to (B, N, D) using the conditioned query.
3. `feature_mlp` (Linear-GELU-Linear) refines the pooled features. `delta_mlp` (Linear-GELU-Linear, no output activation) then produces one value per patch, (B, N, 1).
4. The patch deltas are multiplied by `delta_scale` and hard-clamped to [-1, 1]. There is no Tanh. `delta_scale` is a fixed buffer (`config.delta_scale`, default 0.5), not a learnable parameter.
5. The 8x8 delta grid is upsampled by learned blocks to 64x64 and then bilinearly to 256x256, and clamped to [-1, 1] again.
6. `prediction = (anchor + delta).clamp(-1, 1)`. When `anchor_observation` is `None` (`anchor_mode="none"`), the decoder predicts absolute NDVI directly.

---

## Summary

| Aspect | Implementation |
|--------|----------------|
| Predictions per forward pass | 1 (for specified target) |
| Target selection | `predict_target_indices` > `predict_years` (default `[2022]`) > last `forecast_horizon` timesteps |
| Input timesteps | All history before the target, truncated to the shortest history in the batch (multiple of 4), or a fixed window. Training also subsamples `temporal_subset_years` years |
| Decoder conditioning | `TargetTemporalEncoder` encodes (year, season) |
| Attention mechanism | Query conditioned on target embedding |
| Anchor observation | Mean of the last `anchor_years=5` same-season observations in the input window (`anchor_mode="same_season"`, `model/config.py`), excluding nodata |

Multi-output decoding and autoregressive rollout are not implemented. A single target-conditioned decoder covers all horizons.

---

## Horizon Evaluation

Two scripts evaluate one fine-tuned checkpoint at several lead times h. Lead time is counted in seasons after the last input timestep, which defaults to index 131 (2022 winter, `--last-input-idx`), so the target index is `131 + h`. Both write a TSV with `val_mae`, `val_rmse`, `val_r2`, `val_R2_det`, `val_bias`, `val_anchor_mae` and `val_delta_abs_mean` per horizon.

- **`scripts/eval_horizons_fast.py`** (OOD-st, `--eval-dir` = the `ood_spatial` shards): evaluates h = 1..8 (2023 spring through 2024 winter) on a **fixed input window, indices 0-131** (1990 spring through 2022 winter), for every horizon. The encoder runs once per batch, and only the decoder and anchor are recomputed per horizon. Metrics come from `PerSampleMetricsTracker`.

  ```bash
  python scripts/eval_horizons_fast.py \
      --checkpoint checkpoints/seasonalvit_finetune.pt \
      --eval-dir /path/to/shards/ood_spatial/ood_spatial \
      --output horizon_results_ood_st.tsv
  ```

- **`scripts/eval_horizons.py`**: covers h in {1, 2, 4, 8} by default. It sets `predict_target_indices=[131 + h]` and calls `FinetuningTrainer.validate()`. Because `validate()` uses all history before the target, **the input window shifts with the horizon**: for h > 1 it ends at `target_idx - 1` and therefore includes observed seasons after 2022 winter.

---

## Benefits of Variable Target Training

1. **Single decoder learns all horizons** - No need for K separate heads
2. **Target conditioning** - Model learns which historical patterns matter for each target
3. **Interpretable attention** - Can analyze which timesteps matter for different targets
4. **Flexible at inference** - Predict any timestep by specifying target (year, season)
5. **No error accumulation** - Direct prediction, not autoregressive
