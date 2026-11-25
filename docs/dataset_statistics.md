# GloSeaN Dataset Statistics

Generated: 2026-01-29 13:05:38

## Overview

- **Total locations**: 250,000 sampled
- **Unique samples in splits**: 200,000 (after the completeness filter)
- **K-means blocks**: 100

## Train/Validation/Test Splits

`val_iid` in the code and manifest is the split called `val_id` in the paper.

| Split | Samples | Input Years | Predict Years | Purpose |
|-------|---------|-------------|---------------|---------|
| train | 157,523 | 1990-2021 | 2022-2022 | Training |
| val_iid | 8,000 | 1990-2021 | 2022-2022 | In-distribution validation |
| ood_t | 20,000 (8,000 shared with val_iid + 12,000 additional) | 1990-2022 | 2023-2024 | Temporal OOD evaluation |
| ood_s | 22,477 | 1990-2021 | 2022-2022 | Spatial OOD evaluation |
| ood_st | 22,477 | 1990-2022 | 2023-2024 | Spatial + Temporal OOD |

## Land Cover Distribution

The land cover, vegetation fraction, hemisphere, and latitude tables below are computed over all 250,000 sampled locations.

Based on ESA WorldCover v200 (2021) dominant class per tile. Percentages are of 250,000; the listed classes sum to 235,074, and the remaining 14,926 locations have no dominant class assigned.

| Code | Class | Count | Percentage |
|------|-------|------:|----------:|
| 10 | Tree cover | 32,111 | 12.8% |
| 20 | Shrubland | 21,665 | 8.7% |
| 30 | Grassland | 54,049 | 21.6% |
| 40 | Cropland | 43,436 | 17.4% |
| 50 | Built-up | 415 | 0.2% |
| 60 | Bare/sparse vegetation | 59,554 | 23.8% |
| 80 | Permanent water bodies | 22,159 | 8.9% |
| 90 | Herbaceous wetland | 1,624 | 0.7% |
| 95 | Mangroves | 56 | 0.0% |
| 100 | Moss and lichen | 5 | 0.0% |

## Vegetation Fraction

Fraction of each tile covered by vegetation classes (tree cover, shrubland, grassland, cropland, wetland, mangroves, moss/lichen).

- **Mean**: 0.605
- **Median**: 0.919
- **Std Dev**: 0.446
- **Range**: 0.000 - 1.000

### Distribution

| Range | Count | Percentage |
|-------|------:|----------:|
| 0-20% | 84,408 | 33.8% |
| 20-40% | 7,247 | 2.9% |
| 40-60% | 7,287 | 2.9% |
| 60-80% | 10,883 | 4.3% |
| 80-<100% | 130,349 | 52.1% |
| 100% | 9,826 | 3.9% |

## Geographic Distribution

- **Latitude**: -48.0° to 54.04° (mean: 18.06°)
- **Longitude**: -173.28° to 179.82° (mean: 14.29°)

### Hemisphere

- **Northern**: 196,481 (78.6%)
- **Southern**: 53,519 (21.4%)

### Latitude Bands

| Band | Range | Count | Percentage |
|------|-------|------:|----------:|
| Antarctic | -90 to -60° | 0 | 0.0% |
| S Temperate | -60 to -30° | 19,987 | 8.0% |
| S Tropical | -30 to 0° | 33,532 | 13.4% |
| N Tropical | 0 to 30° | 86,835 | 34.7% |
| N Temperate | 30 to 60° | 109,646 | 43.9% |
| Arctic | 60 to 90° | 0 | 0.0% |

## Notes

- Land cover data from ESA WorldCover v200 (2021) at 10m resolution
- NDVI tiles are 264×264 pixels at 30m Landsat resolution (~7.9km × 7.9km)
- Sampling locations from SSL4EO-L dataset
- Geographic splits use K-means clustering to prevent spatial data leakage
