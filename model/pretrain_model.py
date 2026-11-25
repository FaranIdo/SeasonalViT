"""Pre-training Model for MAE-style reconstruction.

Combines the spatial encoder, temporal transformer, and positional encodings
with a simple reconstruction head for masked autoencoder pre-training.
The reconstruction head is discarded after pre-training.

Supports two masking modes:
1. Efficient temporal masking (default): Only visible timesteps pass through
   Swin-T, giving ~2x speedup with 50% masking. Mask tokens are inserted
   for masked timesteps before the temporal transformer.
2. Legacy mode: All timesteps encoded, then masked. Less efficient but
   allows patch-level masking within images.
"""

from typing import Dict, Optional, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import SeasonalViTConfig
from model.encoders.spatial import HierarchicalSpatialEncoder
from model.encoders.temporal import SeasonalPositionalEncoding
from model.temporal_transformer import DualScaleTemporalTransformer

if TYPE_CHECKING:
    from train.masking import EfficientMaskOutput


class ReconstructionHead(nn.Module):
    """Simple MLP head for patch reconstruction.

    Predicts the original pixel values for masked patches.
    This head is discarded after pre-training.

    Input: (B, T, N, D) encoded features
    Output: (B, T, N, patch_size^2) reconstructed patch pixels
    """

    def __init__(self, config: SeasonalViTConfig) -> None:
        """Initialize reconstruction head.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.config = config
        d_model = config.embed_dim
        hidden_dim = config.decoder_hidden_dim

        # Compute output dimension (pixels per patch)
        # Swin-T with 256x256 input → 8x8 patches → each patch is 32x32 pixels
        self.patch_pixels = 32 * 32  # 1024 pixels per patch (grayscale)

        self.head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.patch_pixels),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights for better training stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct patch pixels from encoded features.

        Args:
            x: Encoded features of shape (B, T, N, D).

        Returns:
            Reconstructed patch pixels of shape (B, T, N, patch_pixels).
        """
        return self.head(x)


class PretrainingModel(nn.Module):
    """MAE-style pre-training model for NDVI sequences.

    Combines:
    - Swin-T spatial encoder (pretrained weights loaded)
    - Seasonal positional encoding
    - Dual-scale temporal transformer
    - Simple reconstruction head (discarded after pre-training)

    The model learns to reconstruct masked patches from visible context,
    enabling self-supervised learning on unlabeled NDVI data.

    Input: NDVI sequence (B, T, H, W) + season/year metadata
    Output: Reconstructed patches + encoded features

    Attributes:
        config: Model configuration.
        spatial_encoder: Swin-T based spatial encoder.
        temporal_encoding: Seasonal positional encoding.
        temporal_transformer: Dual-scale temporal transformer.
        reconstruction_head: MLP for patch reconstruction.
    """

    def __init__(self, config: SeasonalViTConfig) -> None:
        """Initialize pre-training model.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.config = config

        # Spatial encoder
        self.spatial_encoder = HierarchicalSpatialEncoder(config)

        # Temporal encoding
        self.temporal_encoding = SeasonalPositionalEncoding(config)

        # Temporal transformer
        self.temporal_transformer = DualScaleTemporalTransformer(config)

        # Reconstruction head (discarded after pre-training)
        self.reconstruction_head = ReconstructionHead(config)

        # Mask token for patch-level masking (legacy mode)
        # Shape: (1, 1, 1, D) - broadcast to (B, T, N, D)
        self.mask_token = nn.Parameter(torch.randn(1, 1, 1, config.embed_dim) * 0.02)

        # Mask token for temporal masking (efficient mode)
        # Shape: (1, 1, N, D) - one token per spatial patch for masked timesteps
        self.temporal_mask_token = nn.Parameter(
            torch.randn(1, 1, config.num_spatial_patches, config.embed_dim) * 0.02
        )

    def _apply_mask(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply mask by replacing masked positions with mask token.

        Args:
            features: Spatial features of shape (B, T, N, D).
            mask: Boolean mask of shape (B, T, N), True = masked.

        Returns:
            Features with masked positions replaced by mask token.
        """
        # Expand mask token to match feature dimensions
        mask_tokens = self.mask_token.expand(
            features.shape[0], features.shape[1], features.shape[2], -1
        )

        # Apply mask: where mask is True, use mask_token; else use features
        masked_features = torch.where(
            mask.unsqueeze(-1),  # (B, T, N, 1)
            mask_tokens,
            features,
        )

        return masked_features

    def _patchify(self, images: torch.Tensor) -> torch.Tensor:
        """Convert images to patches for reconstruction target.

        Divides each image into 8x8 patches of 32x32 pixels each.

        Args:
            images: Images of shape (B, T, H, W).

        Returns:
            Patches of shape (B, T, N, patch_pixels) where N=64, patch_pixels=1024.
        """
        B, T, H, W = images.shape
        patch_size = 32
        grid_size = H // patch_size  # 8

        # Reshape to patches
        # (B, T, H, W) → (B, T, grid_size, patch_size, grid_size, patch_size)
        patches = images.view(B, T, grid_size, patch_size, grid_size, patch_size)
        # (B, T, grid_size, grid_size, patch_size, patch_size)
        patches = patches.permute(0, 1, 2, 4, 3, 5)
        # (B, T, N, patch_pixels)
        patches = patches.reshape(B, T, grid_size * grid_size, patch_size * patch_size)

        return patches

    def forward_efficient(
        self,
        images: torch.Tensor,
        season_ids: torch.Tensor,
        year_indices: torch.Tensor,
        efficient_mask: "EfficientMaskOutput",
        seq_positions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Efficient forward pass with temporal masking BEFORE spatial encoding.

        Only visible timesteps pass through Swin-T, giving ~2x speedup with
        50% masking ratio. Mask tokens are inserted for masked timesteps.

        Args:
            images: NDVI images of shape (B, T, H, W).
            season_ids: Season indices (0-3) of shape (B, T) or (T,).
            year_indices: Year indices of shape (B, T) or (T,).
            efficient_mask: EfficientMaskOutput with visible/masked indices.
            seq_positions: Original timestep indices for sinusoidal PE when using
                temporal year subsampling. Shape (B, T) or None for default 0..T-1.

        Returns:
            Dictionary with:
            - 'reconstruction': Predicted patches (B, T, N, patch_pixels)
            - 'target': Ground truth patches (B, T, N, patch_pixels)
            - 'features': Encoded features (B, T, N, D)
            - 'temporal_mask': Boolean mask (B, T), True = masked timestep
        """
        B, T, H, W = images.shape
        N = self.config.num_spatial_patches
        D = self.config.embed_dim

        num_visible = efficient_mask.num_visible
        visible_indices = efficient_mask.visible_indices  # (B, num_visible)
        masked_indices = efficient_mask.masked_indices  # (B, num_masked)

        # 1. Extract only visible timesteps for encoding
        # Expand indices to gather from images: (B, num_visible) -> (B, num_visible, H, W)
        visible_idx_expanded = visible_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        visible_images = torch.gather(images, 1, visible_idx_expanded)  # (B, num_visible, H, W)

        # 2. Spatial encoding - ONLY visible timesteps (the speedup!)
        visible_flat = visible_images.reshape(B * num_visible, 1, H, W)
        visible_features = self.spatial_encoder(visible_flat)  # (B*num_visible, N, D)
        visible_features = visible_features.view(B, num_visible, N, D)

        # 3. Create full feature tensor with mask tokens for masked positions
        # Initialize all positions with mask tokens
        all_features = self.temporal_mask_token.expand(B, T, -1, -1).clone()  # (B, T, N, D)

        # Scatter visible features into their original positions
        visible_idx_feat = visible_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, N, D)
        all_features.scatter_(1, visible_idx_feat, visible_features)

        # 4. Temporal encoding (for all positions)
        temporal_enc = self.temporal_encoding(
            season_ids, year_indices, seq_positions=seq_positions
        )  # (B, T, D)

        # 5. Temporal transformer processes all positions
        features = self.temporal_transformer(all_features, temporal_enc)  # (B, T, N, D)

        # 6. Reconstruction
        reconstruction = self.reconstruction_head(features)  # (B, T, N, patch_pixels)

        # 7. Create target patches (for all timesteps, loss computed on masked only)
        target = self._patchify(images)  # (B, T, N, patch_pixels)

        # 8. Create temporal mask for loss computation (True = masked = compute loss)
        temporal_mask = ~efficient_mask.visible_mask  # (B, T)

        return {
            "reconstruction": reconstruction,
            "target": target,
            "features": features,
            "temporal_mask": temporal_mask,
        }

    def forward(
        self,
        images: torch.Tensor,
        season_ids: torch.Tensor,
        year_indices: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        seq_positions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Legacy forward pass for pre-training (encodes all, then masks).

        Args:
            images: NDVI images of shape (B, T, H, W).
            season_ids: Season indices (0-3) of shape (B, T) or (T,).
            year_indices: Year indices of shape (B, T) or (T,).
            mask: Optional boolean mask of shape (B, T, N), True = masked.
                  If None, no masking is applied.
            seq_positions: Original timestep indices for sinusoidal PE when using
                temporal year subsampling. Shape (B, T) or None for default 0..T-1.

        Returns:
            Dictionary with:
            - 'reconstruction': Predicted patches (B, T, N, patch_pixels)
            - 'target': Ground truth patches (B, T, N, patch_pixels)
            - 'features': Encoded features (B, T, N, D)
            - 'mask': Applied mask (B, T, N) or None
        """
        B, T, H, W = images.shape
        N = self.config.num_spatial_patches

        # 1. Spatial encoding
        # Flatten to (B*T, 1, H, W) for spatial encoder
        images_flat = images.reshape(B * T, 1, H, W)
        spatial_features = self.spatial_encoder(images_flat)  # (B*T, N, D)
        spatial_features = spatial_features.view(B, T, N, -1)  # (B, T, N, D)

        # 2. Apply mask if provided
        if mask is not None:
            spatial_features = self._apply_mask(spatial_features, mask)

        # 3. Temporal encoding
        temporal_enc = self.temporal_encoding(
            season_ids, year_indices, seq_positions=seq_positions
        )  # (B, T, D)

        # 4. Temporal transformer
        features = self.temporal_transformer(spatial_features, temporal_enc)

        # 5. Reconstruction
        reconstruction = self.reconstruction_head(features)  # (B, T, N, patch_pixels)

        # 6. Create target patches
        target = self._patchify(images)  # (B, T, N, patch_pixels)

        return {
            "reconstruction": reconstruction,
            "target": target,
            "features": features,
            "mask": mask,
        }

    def get_encoder_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get state dict for encoder components only.

        Returns state dict containing:
        - spatial_encoder
        - temporal_encoding
        - temporal_transformer

        Excludes reconstruction_head and mask_token.

        Returns:
            State dict for encoder components.
        """
        encoder_state = {}
        for name, param in self.named_parameters():
            if not name.startswith("reconstruction_head") and name != "mask_token":
                encoder_state[name] = param.data.clone()
        return encoder_state


def create_pretrain_model(
    config: Optional[SeasonalViTConfig] = None,
) -> PretrainingModel:
    """Factory function to create pre-training model.

    Args:
        config: Model configuration. Uses defaults if None.

    Returns:
        Configured PretrainingModel instance.
    """
    if config is None:
        config = SeasonalViTConfig()
    return PretrainingModel(config)
