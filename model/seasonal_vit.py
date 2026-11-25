"""SeasonalViT: Full forecasting model for long-term NDVI prediction.

Combines all components:
- Swin-T spatial encoder (from pre-training)
- Seasonal positional encoding (from pre-training)
- Location encoding (new for forecasting)
- Dual-scale temporal transformer (from pre-training)
- Observation-anchored decoder (new for forecasting)
"""

import logging
from dataclasses import replace
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from model.config import SeasonalViTConfig

logger = logging.getLogger(__name__)
from model.encoders.spatial import HierarchicalSpatialEncoder
from model.encoders.temporal import SeasonalPositionalEncoding
from model.encoders.location import LocationEncoder
from model.temporal_transformer import DualScaleTemporalTransformer
from model.decoder import ObservationAnchoredDecoder


class SeasonalViT(nn.Module):
    """SeasonalViT: Vision Transformer for long-term NDVI forecasting.

    Architecture:
    1. Swin-T encodes each 256×256 image to 64 spatial tokens
    2. Seasonal positional encoding adds temporal context
    3. Location encoding adds geographic context
    4. Dual-scale transformer processes seasonal + trend patterns
    5. Observation-anchored decoder predicts NDVI as anchor + delta

    The model can be initialized from pre-trained weights by loading
    encoder components from PretrainingModel.

    Input:
    - images: (B, T, H, W) NDVI sequence
    - season_ids: (B, T) or (T,) season indices 0-3
    - year_indices: (B, T) or (T,) year indices
    - lat, lon: (B,) geographic coordinates
    - target_season: (B,) season to predict
    - target_year: (B,) year to predict

    Output:
    - prediction: (B, 1, H, W) predicted NDVI
    - delta: (B, 1, H, W) predicted change from anchor
    - anchor: (B, 1, H, W) same-season historical observation

    Attributes:
        config: Model configuration.
        spatial_encoder: Swin-T based spatial encoder.
        temporal_encoding: Seasonal positional encoding.
        location_encoder: Geographic location encoder.
        temporal_transformer: Dual-scale temporal transformer.
        decoder: Observation-anchored decoder.
    """

    def __init__(self, config: SeasonalViTConfig) -> None:
        """Initialize SeasonalViT.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.config = config

        # Spatial encoder (from pre-training)
        self.spatial_encoder = HierarchicalSpatialEncoder(config)

        # Temporal encoding (from pre-training)
        self.temporal_encoding = SeasonalPositionalEncoding(config)

        # Location encoding (new for forecasting)
        self.location_encoder = LocationEncoder(config)

        # Temporal transformer (from pre-training)
        self.temporal_transformer = DualScaleTemporalTransformer(config)

        # Decoder (new for forecasting)
        self.decoder = ObservationAnchoredDecoder(config)

    def _get_climatology_anchor(
        self,
        images: torch.Tensor,
        season_ids: torch.Tensor,
        target_season: torch.Tensor,
    ) -> torch.Tensor:
        """Get anchor observation based on config.anchor_mode.

        - "same_season" (default): average last N same-season observations.
        - "last_step": use the last observation in the input (any season).
        - "none": caller should not invoke this; raises.

        Args:
            images: NDVI images of shape (B, T, H, W).
            season_ids: Season indices of shape (B, T).
            target_season: Target season of shape (B,).

        Returns:
            Anchor of shape (B, 1, H, W).
        """
        mode = getattr(self.config, "anchor_mode", "same_season")
        if mode == "none":
            raise RuntimeError(
                "anchor_mode='none' — anchor should not be computed"
            )
        if mode == "last_step":
            return self._last_valid_anchor(images)

        B, T, H, W = images.shape
        N = self.config.anchor_years
        nodata = self.config.nodata_value

        # Find same-season timesteps and select last N
        season_match = (season_ids == target_season.unsqueeze(1))  # (B, T)
        match_indices, slot_valid = self._select_last_n_matches(season_match, N)

        # Gather the selected observations: (B, N, H, W)
        idx = match_indices.view(B, N, 1, 1).expand(B, N, H, W)
        selected = torch.gather(images, 1, idx)  # (B, N, H, W)

        # Mark padding slots as nodata so they are excluded from averaging
        padding_mask = ~slot_valid.view(B, N, 1, 1).expand(B, N, H, W)
        selected = torch.where(padding_mask, torch.full_like(selected, nodata), selected)

        # Average valid pixels, excluding nodata
        return self._average_valid_pixels(selected, nodata)  # (B, 1, H, W)

    def _last_valid_anchor(self, images: torch.Tensor) -> torch.Tensor:
        """Per-pixel last valid (non-nodata) timestep as anchor.

        For each pixel, walks back from the final timestep and takes the most
        recent non-nodata value. If a pixel is nodata across all timesteps,
        the anchor stays nodata for that pixel.

        Args:
            images: NDVI images of shape (B, T, H, W).

        Returns:
            Anchor of shape (B, 1, H, W).
        """
        nodata = self.config.nodata_value
        T = images.shape[1]

        valid = images != nodata  # (B, T, H, W)
        # Reverse time so argmax finds the most recent valid pixel
        reversed_valid = valid.flip(dims=[1])
        rev_idx = reversed_valid.float().argmax(dim=1, keepdim=True)  # (B, 1, H, W)
        last_valid_idx = (T - 1) - rev_idx  # back to original time ordering

        anchor = torch.gather(images, 1, last_valid_idx)  # (B, 1, H, W)

        # Pixels with no valid timestep remain nodata
        any_valid = valid.any(dim=1, keepdim=True)
        return torch.where(any_valid, anchor, torch.full_like(anchor, nodata))

    def _select_last_n_matches(
        self,
        season_match: torch.Tensor,
        n: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select indices of last N True values per batch row.

        Args:
            season_match: Boolean mask (B, T).
            n: Number of matches to select.

        Returns:
            Tuple of (indices, slot_valid):
            - indices: (B, N) timestep indices (padded with 0 where invalid).
            - slot_valid: (B, N) bool mask, True for real matches.
        """
        B, T = season_match.shape

        # Reversed cumsum to rank matches from the end (last match = 1)
        match_float = season_match.float()
        rank_from_end = match_float.flip(1).cumsum(1).flip(1)  # (B, T)

        # Select positions where rank <= N (the last N matches)
        last_n_mask = (rank_from_end <= n) & season_match  # (B, T)

        # Handle fallback: if no matches, use last timestep
        has_match = season_match.any(dim=1)  # (B,)
        num_fallback = (~has_match).sum().item()
        if num_fallback > 0:
            logger.warning(
                f"{num_fallback}/{B} samples have no same-season observation. "
                "Using last observation as anchor."
            )

        # Collect indices for each batch element
        result = torch.zeros((B, n), device=season_match.device, dtype=torch.long)
        slot_valid = torch.zeros((B, n), device=season_match.device, dtype=torch.bool)
        for b in range(B):
            matched = last_n_mask[b].nonzero(as_tuple=False).squeeze(-1)
            if matched.numel() > 0:
                count = min(matched.numel(), n)
                result[b, :count] = matched[-count:]
                slot_valid[b, :count] = True
            elif not has_match[b]:
                # Fallback: use last timestep
                result[b, 0] = T - 1
                slot_valid[b, 0] = True
        return result, slot_valid

    def _average_valid_pixels(
        self,
        selected: torch.Tensor,
        nodata: float,
    ) -> torch.Tensor:
        """Average observations excluding nodata pixels.

        Args:
            selected: Observations (B, N, H, W).
            nodata: Nodata sentinel value.

        Returns:
            Averaged anchor (B, 1, H, W). Pixels where all N are nodata
            remain nodata.
        """
        valid = (selected != nodata)  # (B, N, H, W)
        safe_values = torch.where(valid, selected, torch.zeros_like(selected))
        valid_count = valid.float().sum(dim=1, keepdim=True)  # (B, 1, H, W)
        avg = safe_values.sum(dim=1, keepdim=True) / valid_count.clamp(min=1)

        # Where all observations are nodata, output nodata
        all_nodata = (valid_count == 0)
        avg = torch.where(all_nodata, torch.full_like(avg, nodata), avg)
        return avg

    def _encode_spatial_chunked(self, images_flat: torch.Tensor) -> torch.Tensor:
        """Encode spatial features in chunks to manage memory.

        Args:
            images_flat: Flattened images of shape (B*T, 1, H, W).

        Returns:
            Spatial features of shape (B*T, N, D).
        """
        chunk_size = self.config.spatial_chunk_size
        if chunk_size <= 0 or images_flat.shape[0] <= chunk_size:
            return self.spatial_encoder(images_flat)

        # Process in chunks
        features_list = []
        for i in range(0, images_flat.shape[0], chunk_size):
            chunk = images_flat[i : i + chunk_size]
            chunk_features = self.spatial_encoder(chunk)
            features_list.append(chunk_features)

        return torch.cat(features_list, dim=0)

    def forward(
        self,
        images: torch.Tensor,
        season_ids: torch.Tensor,
        year_indices: torch.Tensor,
        lat: torch.Tensor,
        lon: torch.Tensor,
        target_season: torch.Tensor,
        target_year: torch.Tensor,
        return_attention_analysis: bool = False,
        seq_positions: Optional[torch.Tensor] = None,
        precomputed_anchor: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for NDVI forecasting.

        Args:
            images: NDVI images of shape (B, T, H, W).
            season_ids: Season indices (0-3) of shape (B, T) or (T,).
            year_indices: Year indices of shape (B, T) or (T,).
            lat: Latitude in degrees of shape (B,).
            lon: Longitude in degrees of shape (B,).
            target_season: Target season to predict of shape (B,).
            target_year: Target year to predict of shape (B,).
            return_attention_analysis: If True, return temporal attention analysis
                for interpretability. Use during evaluation, not training.
            seq_positions: Original timestep indices for sinusoidal PE when using
                temporal subsampling. Shape (B, T). None = default 0..T-1.
            precomputed_anchor: Pre-computed climatology anchor (B, 1, H, W).
                When provided, skips internal anchor computation. Use when anchor
                should be computed on full input before temporal subsampling.

        Returns:
            Dictionary with:
            - 'prediction': Predicted NDVI (B, 1, H, W)
            - 'delta': Predicted change (B, 1, H, W)
            - 'anchor': Same-season observation (B, 1, H, W)
            - 'attention_analysis': Optional TemporalAttentionAnalysis
        """
        B, T, H, W = images.shape
        N = self.config.num_spatial_patches

        # Handle 1D season/year indices
        if season_ids.dim() == 1:
            season_ids = season_ids.unsqueeze(0).expand(B, -1)
        if year_indices.dim() == 1:
            year_indices = year_indices.unsqueeze(0).expand(B, -1)

        # 1. Spatial encoding (with optional chunking for memory management)
        # Replace nodata sentinel (-9999) with 0.0 before encoding to avoid
        # dominating BatchNorm stats. The original images tensor is preserved
        # for _get_climatology_anchor which needs nodata values.
        images_for_encoding = images.clone()
        images_for_encoding[images_for_encoding == self.config.nodata_value] = self.config.ndvi_mean
        images_flat = images_for_encoding.reshape(B * T, 1, H, W)
        spatial_features = self._encode_spatial_chunked(images_flat)  # (B*T, N, D)
        spatial_features = spatial_features.view(B, T, N, -1)  # (B, T, N, D)

        # 2. Temporal encoding
        temporal_enc = self.temporal_encoding(
            season_ids, year_indices, seq_positions=seq_positions
        )  # (B, T, D)

        # 3. Location encoding
        location_enc, hemisphere = self.location_encoder(lat, lon)  # (B, D), (B,)
        # Add location encoding to spatial features (broadcast over T, N)
        spatial_features = spatial_features + location_enc.unsqueeze(1).unsqueeze(2)

        # 4. Temporal transformer
        temporal_features = self.temporal_transformer(
            spatial_features, temporal_enc
        )  # (B, T, N, D)

        # 5. Get anchor (skipped when anchor_mode='none')
        anchor_mode = getattr(self.config, "anchor_mode", "same_season")
        if anchor_mode == "none":
            anchor = None
        elif precomputed_anchor is not None:
            anchor = precomputed_anchor
        else:
            anchor = self._get_climatology_anchor(
                images, season_ids, target_season
            )  # (B, 1, H, W)

        # 6. Decode to prediction with target conditioning
        output = self.decoder(
            temporal_features,
            anchor,
            target_season,
            target_year,
            return_attention_analysis=return_attention_analysis,
        )

        return output

    @classmethod
    def from_pretrained(
        cls,
        pretrain_state_dict: Dict[str, torch.Tensor],
        config: Optional[SeasonalViTConfig] = None,
        strict: bool = False,
    ) -> "SeasonalViT":
        """Create SeasonalViT initialized from pre-trained weights.

        Loads encoder weights from PretrainingModel:
        - spatial_encoder
        - temporal_encoding
        - temporal_transformer

        Location encoder and decoder are randomly initialized.

        Args:
            pretrain_state_dict: State dict from PretrainingModel.
            config: Model configuration. Uses defaults if None.
            strict: Whether to require exact key match.

        Returns:
            SeasonalViT with pre-trained encoder weights.
        """
        if config is None:
            config = SeasonalViTConfig()

        # Skip downloading ImageNet Swin-T weights since they'll be
        # overwritten by the pretrained checkpoint immediately after.
        config = replace(config, swin_pretrained=False)

        model = cls(config)

        # Filter to only encoder keys
        encoder_state = {}
        for key, value in pretrain_state_dict.items():
            # Skip reconstruction head and mask token
            if key.startswith("reconstruction_head") or key == "mask_token":
                continue
            encoder_state[key] = value

        # Load encoder weights
        missing, unexpected = model.load_state_dict(encoder_state, strict=False)

        if strict and missing:
            raise ValueError(f"Missing keys: {missing}")

        return model

    def freeze_encoder(self) -> None:
        """Freeze encoder weights for fine-tuning.

        Freezes:
        - spatial_encoder
        - temporal_encoding
        - temporal_transformer

        Leaves trainable:
        - location_encoder
        - decoder
        """
        for name, param in self.named_parameters():
            if (
                name.startswith("spatial_encoder")
                or name.startswith("temporal_encoding")
                or name.startswith("temporal_transformer")
            ):
                param.requires_grad = False

    def freeze_spatial_encoder(self) -> None:
        """Freeze only the spatial encoder (Swin-T) for fine-tuning.

        Eliminates backward pass through the most expensive component,
        while keeping temporal transformer and decoder trainable.
        """
        for name, param in self.named_parameters():
            if name.startswith("spatial_encoder"):
                param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """Unfreeze encoder weights for full fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True

    def get_num_parameters(self, trainable_only: bool = False) -> int:
        """Count model parameters.

        Args:
            trainable_only: If True, only count trainable parameters.

        Returns:
            Number of parameters.
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


def create_seasonal_vit(
    config: Optional[SeasonalViTConfig] = None,
) -> SeasonalViT:
    """Factory function to create SeasonalViT.

    Args:
        config: Model configuration. Uses defaults if None.

    Returns:
        Configured SeasonalViT instance.
    """
    if config is None:
        config = SeasonalViTConfig()
    return SeasonalViT(config)
