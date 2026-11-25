# SeasonalViT: Architecture Concepts for Long-Term NDVI Prediction

## Executive Summary

| Property | Why It Matters | How We Exploit It |
|----------|----------------|-------------------|
| **35 years** | Unprecedented temporal depth | Dual-scale temporal modeling (seasonal + trend) |
| **4 seasons/year** | Perfect phenological alignment | Explicit seasonal encoding |
| **256x256 images** | Large spatial context | Hierarchical spatial encoder (Swin) |
| **200K global locations** | Geographic diversity | Location-aware encoding |
| **NDVI only** | Focused signal | Same-season anchored delta prediction |
| **Dense prediction** | Full image output | Patch-based decoder with upsampling |

---

## Core Architecture

```
+-----------------------------------------------------------------------------+
|                              SeasonalViT                                     |
+-----------------------------------------------------------------------------+
|                                                                              |
|  Input: NDVI sequence (B, T, 256, 256) + metadata                           |
|         T = num_years x 4 seasons                                            |
|                                                                              |
|  +---------------------------------------------------------------------+    |
|  | 1. HIERARCHICAL SPATIAL ENCODER (Swin-Tiny)                        |    |
|  |    256x256 -> 8x8 patches -> (B*T, 64, 768)                        |    |
|  +---------------------------------------------------------------------+    |
|                              |                                               |
|  +---------------------------------------------------------------------+    |
|  | 2. POSITIONAL ENCODINGS                                            |    |
|  |    - Season embedding (0-3)                                        |    |
|  |    - Sensor era embedding (0-3)                                    |    |
|  |    - Cyclical season encoding                                      |    |
|  |    - Year encoding (multi-frequency)                               |    |
|  |    - Sequence PE (sinusoidal)                                      |    |
|  |    + Location encoding (3D Cartesian + hemisphere), added to the   |    |
|  |      spatial features                                              |    |
|  +---------------------------------------------------------------------+    |
|                              |                                               |
|  +---------------------------------------------------------------------+    |
|  | 3. DUAL-SCALE TEMPORAL TRANSFORMER                                 |    |
|  |    Seasonal pathway (within year) + Trend pathway (across years)   |    |
|  |    Cross-attention fusion                                          |    |
|  +---------------------------------------------------------------------+    |
|                              |                                               |
|  +---------------------------------------------------------------------+    |
|  | 4. OBSERVATION-ANCHORED DECODER                                    |    |
|  |    - Pool time, keep spatial: (B, T, 64, 768) -> (B, 64, 768)     |    |
|  |    - Per-patch delta MLP: (B, 64, 768) -> (B, 64, 1)              |    |
|  |    - Learned upsample 8x8 -> 64x64, then bilinear -> 256x256       |    |
|  |    - Output = anchor + delta                                       |    |
|  |      (anchor = mean of last 5 same-season observations)            |    |
|  +---------------------------------------------------------------------+    |
|                                                                              |
|  Output: NDVI prediction (B, 1, 256, 256)                                   |
|                                                                              |
+-----------------------------------------------------------------------------+
```

---

## Component Concepts

### 1. Hierarchical Spatial Encoder

**Purpose**: Encode each 256x256 NDVI image into compact spatial features.

**Approach**: Use Swin-Tiny transformer with local (windowed) attention. This is more efficient than global attention for large images.

**Key details**:
- Patch size: 4x4 pixels
- Window size: 7 (timm `swin_tiny_patch4_window7_224`, Swin-T v1)
- Output: 64 spatial tokens of dimension 768 (effectively 8x8 grid)
- Each timestep processed independently, then reshaped to (B, T, 64, 768)

**Why Swin over ViT?** Standard ViT has O(N²) attention complexity. For 256x256 images with 4x4 patches, that's 4096 tokens - prohibitively expensive. Swin uses local windows with shifted connections, scaling linearly with image size.

---

### 2. Positional Encodings

The model uses a **5-component positional encoding** to capture different aspects of the temporal and spatial structure:

| Component | Dimension | Purpose |
|-----------|-----------|---------|
| **Season embedding** | ~153 | Learnable embedding for spring/summer/fall/winter |
| **Sensor era embedding** | ~153 | Handle radiometric differences between Landsat sensors |
| **Cyclical encoding** | ~153 | sin/cos of season angle for smooth seasonal transitions |
| **Year encoding** | ~153 | Multi-frequency (2^0 to 2^9) for long-term patterns |
| **Sequence PE** | ~156 | Standard sinusoidal for absolute position |

**Sensor Era Boundaries**:
- Era 0 (Landsat 5 TM): 1990-1998
- Era 1 (Landsat 7 ETM+): 1999-2012
- Era 2 (Landsat 8 OLI): 2013-2021
- Era 3 (Landsat 9 OLI-2): 2022-2024

This helps the model learn that radiometric properties differ between sensor generations.

---

### 3. Location Encoding

**Purpose**: Make the model location-aware for a global dataset.

**Approach**:
1. Convert lat/lon to 3D Cartesian coordinates (x, y, z on unit sphere)
2. Project through MLP to embedding dimension
3. Add explicit hemisphere embedding (0 = Southern, 1 = Northern; `lat >= 0` is Northern)
4. Combine and project to final encoding

**Why 3D Cartesian?** Avoids discontinuities at the date line (180°/-180°) and poles that occur with raw lat/lon. Points near each other on Earth have similar encodings.

**Why hemisphere embedding?** Seasons are fixed calendar windows for every location, so the "summer" band in the Southern Hemisphere covers austral winter. The hemisphere embedding lets the model learn this inversion.

---

### 4. Dual-Scale Temporal Transformer

**Problem**: 35 years × 4 seasons = 140 timesteps. Standard attention over all 140 is expensive and may miss hierarchical patterns.

**Solution**: Process at two scales simultaneously:

#### Seasonal Pathway (within-year)
- Groups data by year: (B, 35, 4, N, D)
- For each year, applies transformer over 4 seasons
- Learns: "How does spring relate to summer within a year?"

#### Trend Pathway (across-years)
- Groups data by season: (B, 4, 35, N, D)
- For each season, applies transformer over 35 years
- Learns: "How has summer changed over 35 years?"

#### Fusion
- Cross-attention: seasonal features query trend features
- MLP combines both pathways
- Output: features that understand both intra-year cycles and multi-decade trends

---

### 5. Observation-Anchored Decoder

**Key Insight**: NDVI changes slowly between years. Rather than predicting absolute values, predict the *change* from a same-season anchor: the per-pixel mean of the last `anchor_years` (default 5) same-season observations in the input, excluding nodata pixels.

```
prediction = clamp(anchor + delta, -1, 1)
anchor     = mean of last 5 same-season observations
```

**Why this works**:
- Same-season anchor provides high-resolution spatial detail
- Model only needs to predict smooth, low-frequency changes
- Delta is typically small (±0.1 NDVI units)
- The 8x8 per-patch delta is upsampled by three learned `ProgressiveUpsampleBlock`s (ConvTranspose2d, 8→16→32→64), then bilinearly from 64x64 to 256x256

#### Target Conditioning

The decoder is conditioned on the target (year, season) to predict:
1. Encode target timestep (season embedding + cyclical + year encoding)
2. Add conditioning to attention query
3. Model learns which historical timesteps are relevant for each target

This enables **variable target training**: predict any future timestep, not just next-step.

#### Temporal Attention Pooling

Instead of simple averaging over time, use learned attention:
- Learnable query attends over all T timesteps
- Query is conditioned on target timestep
- Output: spatial features weighted by temporal relevance
- Bonus: attention weights are interpretable (which years matter most?)

---

## NoData Handling

Pixels with no valid observation in a seasonal window are stored as NoData (-9999). Mean per-band NoData ranges from 18.0% (1990-1998, Landsat 5 only) to 0.1% (2022-2024, Landsat 8 + 9); see the paper's Appendix B for the per-era and per-climate breakdown.

No temporal interpolation is applied. Missing pixels are filled with a constant and tracked with a boolean mask; they are never used as training targets.

| Component | NoData Handling |
|-----------|-----------------|
| Shards (`scripts/convert_to_shards.py`) | NoData filled with 0.0; a `nodata_masks` tensor is stored alongside the images |
| `SeasonalViT.forward` | Any -9999 sentinel in the input is replaced with `ndvi_mean` (0.35) before spatial encoding |
| Anchor | Averages only non-NoData same-season pixels |
| Pretraining loss | Excludes nodata patches |
| Forecasting loss | Excludes nodata pixels |
| Evaluation | Excludes nodata from metrics |

---

## Training

### Loss Function

Fine-tuning uses `ForecastingLoss` (`train/losses.py`), computed over valid (non-nodata) target pixels:

```
total_loss = smooth_l1(prediction, target; beta=0.1)
           + correlation_loss_weight  * correlation_loss
           + delta_supervision_weight * delta_supervision
           + delta_reg_weight         * delta_regularization
```

| Term | Definition | `FinetuningConfig` default |
|------|------------|---------------------------:|
| Smooth L1 (Huber) | Between prediction and target, `beta=0.1` | 1.0 (fixed) |
| Correlation loss | Per-sample `1 - Pearson r` between prediction and target | 0.5 |
| Delta supervision | L1 between predicted delta and `(target - anchor)`, both divided by `delta_std` (auto-estimated from validation data when `delta_std=0`) | 0.1 |
| Delta regularization | L2 penalty on delta | 0.0 (disabled) |

The paper's fine-tuning run used correlation weight 0.5 and delta-supervision weight 0.2 (`--corr-loss-weight 0.5 --delta-supervision-weight 0.2`).

### Forward Pass Summary

```
1. Flatten images: (B, T, H, W) -> (B*T, 1, H, W)
2. Spatial encode: (B*T, 1, H, W) -> (B*T, 64, 768)
3. Reshape: (B*T, 64, 768) -> (B, T, 64, 768)
4. Add temporal encoding (season, year, etc.)
5. Add location encoding
6. Dual-scale temporal transformer
7. Get same-season anchor from history
8. Decode: attention pool time, predict delta, add to anchor
9. Output: prediction (B, 1, 256, 256)
```

---

## Design Decisions Summary

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Spatial encoder | Swin-Tiny | Efficient local attention for large images |
| Temporal encoding | 5-component | Captures season, sensor era, cycle, year, position |
| Sensor era embedding | 4 eras | Handle Landsat sensor differences |
| Temporal architecture | Dual-scale | Separate seasonal cycles from long-term trends |
| Temporal pooling | Conditioned attention | Learns relevant history per target |
| Decoder | Same-season anchoring (mean of last 5 same-season observations) | Leverage real data for spatial detail |
| Delta resolution | 8x8 patches | Sufficient when anchored to observation |
| Upsampling | Learned ConvTranspose2d 8→64, then bilinear 64→256 | Adequate for smooth delta field |
| NoData | Constant fill + nodata mask | Excluded from anchor and loss |

---

## Interpretability

The attention pooling enables analysis of temporal importance:

- **Effective history length**: How many years back does 90% of attention fall?
- **Recent vs. distant**: What fraction of attention goes to last 5 years?
- **Regional patterns**: Do tropical regions use different history than arid?
- **Seasonal patterns**: Does predicting summer require more/less history?
