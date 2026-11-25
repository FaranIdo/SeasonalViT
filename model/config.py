"""Configuration dataclass for SeasonalViT model."""

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple


@dataclass
class SeasonalViTConfig:
    """Configuration for SeasonalViT model architecture.

    This configuration covers both pre-training and fine-tuning stages.
    All hyperparameters are centralized here to avoid magic numbers.

    Attributes:
        image_size: Input image dimensions (H, W). Data is center-cropped to this size.
        num_timesteps: Total number of timesteps (35 years × 4 seasons = 140).
        num_years: Number of years in the dataset.
        num_seasons: Number of seasons per year.
        patch_size: Swin-T patch size (for reference - hardcoded in timm model name).
        embed_dim: Embedding dimension for spatial features.
        spatial_grid_size: Patches per dimension after Swin-T 32x downsample (for validation).
        num_spatial_patches: Total spatial patches (spatial_grid_size^2).
        swin_pretrained: Whether to use ImageNet pretrained Swin-T weights.
        temporal_depth: Number of transformer layers in temporal encoder.
        temporal_heads: Number of attention heads in temporal transformer.
        temporal_mlp_ratio: MLP expansion ratio in temporal transformer.
        temporal_dropout: Dropout rate in temporal transformer.
        location_embed_dim: Dimension for location encoding.
        decoder_hidden_dim: Hidden dimension in decoder MLPs.
        delta_scale: Scale factor for delta predictions in decoder.
        nodata_value: Value indicating missing data (for reference - pass to dataset).
    """

    # Data dimensions
    image_size: Tuple[int, int] = (256, 256)
    num_timesteps: int = 140
    num_years: int = 35
    num_seasons: int = 4

    # Sensor era configuration
    # Era 0 (TM): years 0-8 (1990-1998)
    # Era 1 (ETM+): years 9-22 (1999-2012)
    # Era 2 (OLI): years 23-31 (2013-2021)
    # Era 3 (OLI-2): years 32-34 (2022-2024)
    num_sensor_eras: int = 4
    sensor_era_boundaries: Tuple[int, ...] = (9, 23, 32)  # Year indices where eras change

    # Spatial encoder (Swin-T)
    # Note: Swin-T patch_size=4 is hardcoded in timm model name "swin_tiny_patch4_window7_224"
    # Swin-T downsamples by 32x total, giving 8×8=64 patches for 256×256 input
    patch_size: int = 4  # For reference only - hardcoded in timm model name
    embed_dim: int = 768
    spatial_grid_size: int = 8  # image_size // 32 (for validation)
    num_spatial_patches: int = 64  # spatial_grid_size^2
    swin_pretrained: bool = True

    # Temporal transformer
    temporal_depth: int = 6
    temporal_heads: int = 12
    temporal_mlp_ratio: float = 4.0
    temporal_dropout: float = 0.1

    # Location encoding
    location_embed_dim: int = 64

    # Decoder
    decoder_hidden_dim: int = 384
    delta_scale: float = 0.5
    attention_pooling_heads: int = 8  # Number of attention heads for temporal pooling

    # Anchor configuration
    # Number of same-season observations to average for the anchor.
    # 1 = single most recent (previous year baseline), 5 = recent climatology.
    anchor_years: int = 5

    # Anchor mode (for ablations, default preserves existing behavior):
    # - "same_season": average last `anchor_years` same-season observations (default).
    # - "last_step": use the last observation in the input (any season).
    # - "none": no anchor; decoder predicts absolute NDVI directly.
    anchor_mode: Literal["same_season", "last_step", "none"] = "same_season"

    # Temporal transformer pathway selection (for ablations):
    # - "both": dual-scale fusion of cross-season + cross-year (default).
    # - "cross_season": phenological pathway only.
    # - "cross_year": climatological pathway only.
    temporal_pathway: Literal["both", "cross_season", "cross_year"] = "both"

    # Data handling
    # Note: nodata_value is defined here for reference but NDVITimeSeriesDataset
    # uses its own parameter. Pass this value when creating the dataset.
    nodata_value: float = -9999.0

    # Input normalization statistics for NDVI -> ImageNet domain adaptation
    # Used by spatial encoder's input projection initialization
    # NDVI stats (approximate for vegetated areas)
    ndvi_mean: float = 0.35
    ndvi_std: float = 0.25
    # ImageNet stats (averaged across RGB channels)
    imagenet_mean: float = 0.449
    imagenet_std: float = 0.226

    # Memory optimization
    # Gradient checkpointing trades ~30% more compute for ~4-5x less memory.
    # Essential for pre-training where B×T images (e.g., 32×140=4480) pass through
    # Swin-T, which would otherwise OOM storing all intermediate activations.
    use_gradient_checkpointing: bool = True
    spatial_chunk_size: int = 0  # 0 = no chunking, >0 = chunk size for spatial encoder
    # Temporal chunking: process sequences in chunks to avoid OOM in temporal transformer.
    # With B=32, seasonal pathway creates 71,680 sequences (32×35×64) which can OOM.
    # Chunking processes them in smaller batches with identical results.
    temporal_chunk_size: int = 8192  # 0 = no chunking, >0 = chunk size for temporal pathways

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        assert self.num_timesteps == self.num_years * self.num_seasons
        assert self.num_spatial_patches == self.spatial_grid_size ** 2
        assert self.image_size[0] == self.image_size[1], "Only square images supported"


@dataclass
class PretrainingConfig:
    """Configuration for pre-training stage.

    Attributes:
        masking_strategy: Which masking strategy to use:
            - "efficient": Levels 2+3 only. Temporal masking BEFORE encoding.
              Only visible timesteps pass through Swin-T. ~2x faster with 50% masking.
            - "full": All 3 levels. Encodes ALL timesteps, then applies patch+temporal
              masks. Slower but includes spatial patch reconstruction learning.
        temporal_mask_ratio: Ratio of timesteps to mask (Level 2).
        image_mask_prob: Probability of additional timestep masking (Level 3).
        patch_mask_ratio: Ratio of patches to mask within each image (Level 1).
            Only used when masking_strategy="full".
        learning_rate: Initial learning rate.
        weight_decay: Weight decay for AdamW optimizer.
        warmup_epochs: Number of warmup epochs.
        max_epochs: Maximum training epochs.
        batch_size: Training batch size.
        train_num_workers: DataLoader workers for training.
        val_num_workers: DataLoader workers for validation.
        max_grad_norm: Maximum gradient norm for clipping.
        gradient_accumulation_steps: Number of steps to accumulate gradients.
        checkpoint_every_n_epochs: Save checkpoint every N epochs.
    """

    # Masking strategy: "efficient" (Levels 2+3, faster) or "full" (all 3 levels, slower)
    # - "efficient": Only visible timesteps encoded through Swin-T. ~2x speedup.
    #   Loses Level 1 (spatial patch masking) but keeps temporal learning.
    # - "full": Encodes all timesteps, then masks. 4x slower but includes
    #   spatial patch reconstruction for learning texture/edges.
    masking_strategy: Literal["efficient", "full"] = "efficient"

    # Level 2: Temporal masking - mask entire timesteps
    temporal_mask_ratio: float = 0.3  # 30% masking: gentler default, more temporal context

    # Level 3: Image masking - rare additional timestep masking for long-range learning
    image_mask_prob: float = 0.1  # 10% extra masking for long-range learning

    # Level 1: Patch masking - only used when masking_strategy="full"
    patch_mask_ratio: float = 0.75

    # Optimization
    learning_rate: float = 1.5e-4  # Bumped for larger effective batch (128 with temporal subsampling)
    min_lr: float = 1e-6  # Minimum LR at end of cosine decay (prevents stopping learning)
    weight_decay: float = 0.05
    warmup_epochs: int = 1
    max_epochs: int = 20
    batch_size: int = 16  # Effective batch = 32 with 2 GPUs
    train_num_workers: int = 2
    val_num_workers: int = 2

    # Gradient clipping
    max_grad_norm: float = 1.0

    # Gradient accumulation
    gradient_accumulation_steps: int = 1

    # Target normalization (MAE-style)
    # When True, normalize reconstruction targets per patch to zero mean / unit variance
    # before computing MSE. This shifts the task from predicting absolute NDVI values
    # to predicting relative spatial patterns, preventing early plateau.
    normalize_target: bool = True

    # Temporal year subsampling
    # Randomly sample this many complete years (each with 4 seasons) per forward pass
    # instead of using all 35 years. Reduces per-sample memory ~3x with K=12, enabling
    # ~4x larger batch sizes. Set to 0 to disable and use all 35 years.
    temporal_subset_years: int = 12

    # Validation batch size override. When temporal_subset_years > 0, training uses
    # fewer timesteps (e.g. T=48) but validation always uses all T=140. If batch_size
    # is increased to exploit subsampling savings, validation may OOM.
    # None = auto-derive (scale down when subsampling, else use batch_size).
    val_batch_size: Optional[int] = None

    # Checkpointing
    checkpoint_every_n_epochs: int = 1

    @property
    def effective_val_batch_size(self) -> int:
        """Return the batch size to use for validation.

        If val_batch_size is explicitly set, use it. If temporal subsampling
        is active, scale down by 4x (since val uses full T=140). Otherwise
        use the training batch_size.
        """
        if self.val_batch_size is not None:
            return self.val_batch_size
        if self.temporal_subset_years > 0:
            return max(1, self.batch_size // 4)
        return self.batch_size


@dataclass
class FinetuningConfig:
    """Configuration for fine-tuning stage.

    Attributes:
        freeze_encoder: Whether to freeze pre-trained encoder weights.
        forecast_horizon: Number of timesteps from end to sample targets from.
            E.g., 8 = last 2 years (8 seasons). During training, target is
            randomly sampled from the last `forecast_horizon` timesteps.
        learning_rate: Initial learning rate (typically lower than pre-training).
        weight_decay: Weight decay for AdamW optimizer.
        warmup_epochs: Number of warmup epochs.
        max_epochs: Maximum training epochs.
        batch_size: Training batch size.
        num_workers: DataLoader workers.
        delta_reg_weight: Weight for delta regularization loss.
    """

    # Encoder freezing
    freeze_encoder: bool = False
    freeze_spatial: bool = False

    # Forecast horizon for variable target training
    # 8 = sample target from last 8 timesteps (2 years)
    forecast_horizon: int = 8

    # Input selection strategy for variable target training:
    # - True: Use ALL timesteps before target (0 to target_idx-1)
    #   More history for near-end targets, variable length per sample
    # - False: Use FIXED window (0 to T-horizon-1, i.e., first 132 timesteps)
    #   Same input length regardless of target, ignores data between cutoff and target
    use_all_history_before_target: bool = True

    # Explicit year targeting (overrides forecast_horizon when set)
    # E.g., [2022] to predict only 2022, matching baseline evaluation splits.
    # Year Y maps to timesteps [(Y-1990)*4, (Y-1990)*4 + 4).
    predict_years: Optional[List[int]] = field(default_factory=lambda: [2022])

    # Season-level target override (takes priority over predict_years).
    # Use for fine-grained horizon analysis where the target must be a
    # specific season offset rather than a whole year. E.g., [132] = 2023
    # spring (h=1 from 2022 winter); [139] = 2024 winter (h=8). Indices
    # are absolute positions in the T=140 sequence, year Y season s →
    # (Y-1990)*4 + s with s in {0=spring,1=summer,2=fall,3=winter}.
    predict_target_indices: Optional[List[int]] = None

    # Optimization
    learning_rate: float = 5e-5
    min_lr: float = 1e-6  # Minimum LR at end of cosine decay (prevents wasting final epochs)
    weight_decay: float = 0.01
    warmup_epochs: int = 1  # Pre-trained encoder doesn't need long warmup
    max_epochs: int = 20
    batch_size: int = 16

    # Data loading
    train_num_workers: int = 4
    val_num_workers: int = 4

    # Differential learning rate
    # Decoder and location encoder are randomly initialized, so they need higher LR.
    # Encoder (pre-trained) uses base `learning_rate`; decoder uses `learning_rate * decoder_lr_multiplier`.
    decoder_lr_multiplier: float = 10.0

    # Loss weights
    delta_reg_weight: float = 0.0  # L2 penalty on delta. 0 = disabled (recommended: it pushes delta toward zero)
    correlation_loss_weight: float = 0.5  # Weight for per-sample correlation loss (R² optimization)
    delta_supervision_weight: float = 0.1  # Weight for direct delta supervision loss (L1 on MAD-normalized delta)
    delta_std: float = 0.0  # Std of (target - anchor) for delta normalization. 0 = auto-estimate from val data.

    # Gradient clipping
    max_grad_norm: float = 1.0

    # Gradient accumulation
    gradient_accumulation_steps: int = 1

    # Checkpointing
    checkpoint_every_n_epochs: int = 1

    # Temporal year subsampling for training
    # Randomly sample this many complete years (each with 4 seasons) per forward pass
    # instead of using all input years. Reduces memory ~3x with K=12, enabling
    # larger batch sizes. Set to 0 to disable. The anchor year (for the
    # observation-anchored decoder) is always included in the sampled years.
    temporal_subset_years: int = 12

    # Validation batch size override. When temporal_subset_years > 0, training uses
    # fewer timesteps but validation always uses all timesteps. If batch_size is
    # increased to exploit subsampling savings, validation may OOM.
    # None = auto-derive (scale down when subsampling, else use batch_size).
    val_batch_size: Optional[int] = None

    @property
    def effective_val_batch_size(self) -> int:
        """Return the batch size to use for validation.

        If val_batch_size is explicitly set, use it. Otherwise scale down
        by 4x since validation always uses full T=140 timesteps regardless
        of temporal subsampling settings.
        """
        if self.val_batch_size is not None:
            return self.val_batch_size
        return max(1, self.batch_size // 4)

    def get_target_indices(self) -> List[int]:
        """Convert predict_years to timestep indices, or use forecast_horizon.

        Priority: predict_target_indices > predict_years > forecast_horizon.

        Returns:
            List of valid target timestep indices.
        """
        if self.predict_target_indices:
            return list(self.predict_target_indices)
        if self.predict_years:
            indices = []
            for year in self.predict_years:
                start = (year - 1990) * 4
                indices.extend(range(start, start + 4))
            return indices
        return list(range(140 - self.forecast_horizon, 140))
