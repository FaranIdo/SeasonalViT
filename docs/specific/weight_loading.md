# Weight Loading: Pretraining to Fine-tuning

## Overview

SeasonalViT uses a two-stage training pipeline. Each stage has different model components, and only the shared encoder transfers between stages.

## Pretraining Model Components

`PretrainingModel` (`model/pretrain_model.py`) contains:

| Component | Purpose |
|---|---|
| `spatial_encoder` (Swin-T) | Extract spatial features from NDVI images |
| `temporal_encoding` | Seasonal positional encoding |
| `temporal_transformer` | Learn temporal relationships |
| `reconstruction_head` | MLP to reconstruct masked patches |
| `mask_token` | Learnable token for spatially masked positions |
| `temporal_mask_token` | Learnable token for temporally masked timesteps |

The Swin-T backbone is initialized from **ImageNet pretrained weights** (via `timm`) at the start of pretraining. This provides a strong spatial feature extraction starting point.

## Fine-tuning Model Components

`SeasonalViT` (`model/seasonal_vit.py`) contains:

| Component | Purpose |
|---|---|
| `spatial_encoder` (Swin-T) | Extract spatial features (from pretraining) |
| `temporal_encoding` | Seasonal positional encoding (from pretraining) |
| `temporal_transformer` | Temporal relationships (from pretraining) |
| `location_encoder` | Encode lat/lon for location-aware predictions |
| `decoder` | Observation-anchored decoder for NDVI forecasting |

## What Transfers and What Doesn't

When `SeasonalViT.from_pretrained()` loads a pretraining checkpoint:

| Component | Action | Reason |
|---|---|---|
| Spatial encoder (Swin-T) | **Transferred** | Learned NDVI spatial features |
| Temporal encoding | **Transferred** | Learned seasonal representations |
| Temporal transformer | **Transferred** | Learned temporal dynamics |
| Reconstruction head | **Dropped** | Pretraining-only; reconstructs masked patches |
| Mask tokens | **Dropped** | Pretraining-only; no masking in fine-tuning |
| Location encoder | **Randomly initialized** | New component for forecasting |
| Decoder | **Randomly initialized** | Different task: forecasting vs. reconstruction |

The filtering happens in `from_pretrained()`:

```python
for key, value in pretrain_state_dict.items():
    if key.startswith("reconstruction_head") or key == "mask_token":
        continue
    encoder_state[key] = value
```

`temporal_mask_token` is not filtered explicitly; it has no counterpart in `SeasonalViT`, so `load_state_dict(strict=False)` reports it as an unexpected key and ignores it.

## ImageNet Weight Optimization

When loading from a pretrained checkpoint, `from_pretrained()` sets `swin_pretrained=False` before constructing the model. This skips downloading ImageNet Swin-T weights from `timm`, since they would be immediately overwritten by the checkpoint weights.

When pretraining from scratch (no checkpoint), `swin_pretrained=True` is used to initialize Swin-T with ImageNet weights as the starting point.

## Differential Learning Rates

Because the encoder is pretrained but the decoder and location encoder are randomly initialized, fine-tuning uses differential learning rates (`FinetuningConfig`):

- **Encoder** (pretrained): base `learning_rate`
- **Decoder + location encoder** (random): `learning_rate * decoder_lr_multiplier` (default 10x)

This lets the new components catch up without destabilizing the pretrained encoder.

## Key Files

- `model/pretrain_model.py`: Pretraining model with reconstruction head
- `model/seasonal_vit.py`: Fine-tuning model with `from_pretrained()` classmethod
- `model/encoders/spatial.py`: Swin-T backbone with `timm` and ImageNet init
- `model/config.py`: `swin_pretrained` flag in `SeasonalViTConfig`
- `scripts/finetune.py`: `load_pretrained_model()` entry point
