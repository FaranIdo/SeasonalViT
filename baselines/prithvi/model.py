"""Prithvi-EO-2.0 fine-tuning baseline for NDVI forecasting.

Loads the Prithvi-EO-2.0 (300M) foundation model from HuggingFace via TerraTorch,
modifies the input projection for 1-channel NDVI input, and adds a convolutional
prediction head for NDVI forecasting.

Architecture:
    Input: (B, T, 1, H, W) NDVI sequence
    -> Modified input projection (1ch -> Prithvi embed dim)
    -> Prithvi-EO-2.0 ViT encoder (frozen or partially fine-tuned)
    -> Conv prediction head -> (B, 1, H, W)

The model uses Prithvi's temporal encoding to capture multi-year dynamics.
Season indices are mapped to approximate day-of-year values for Prithvi's
temporal position encoding.

Requirements:
    pip install terratorch
"""

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Season to approximate day-of-year mapping for Prithvi temporal encoding
# Spring=90, Summer=180, Fall=270, Winter=360
SEASON_TO_DOY = {0: 90, 1: 180, 2: 270, 3: 360}


class PrithviInputAdapter(nn.Module):
    """Adapts 1-channel NDVI input to Prithvi's expected multi-band input.

    Prithvi expects 6-band HLS input. We use a learned 1->6 channel projection
    to map NDVI into the space Prithvi was trained on, then leverage the
    pre-trained patch embedding.
    """

    def __init__(self, out_channels: int = 6):
        """Initialize input adapter.

        Args:
            out_channels: Number of output channels (must match Prithvi input bands).
        """
        super().__init__()
        self.projection = nn.Conv2d(1, out_channels, kernel_size=1, bias=True)

        # Initialize to replicate NDVI across channels (better starting point
        # than random, since NDVI correlates with all vegetation-sensitive bands)
        nn.init.ones_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project 1-channel input to multi-channel.

        Args:
            x: Input (B*T, 1, H, W).

        Returns:
            Projected input (B*T, C_out, H, W).
        """
        return self.projection(x)


class PrithviPredictionHead(nn.Module):
    """Convolutional prediction head for NDVI forecasting.

    Takes the encoder output features and produces a single-channel
    NDVI prediction at the original spatial resolution.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, image_size: int = 224):
        """Initialize prediction head.

        Args:
            input_dim: Encoder output dimension.
            hidden_dim: Hidden dimension for conv layers.
            image_size: Target spatial resolution.
        """
        super().__init__()
        self.image_size = image_size

        # Spatial tokens from ViT are typically (num_patches, embed_dim)
        # Reshape to spatial grid, then upsample to target resolution
        self.head = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Generate prediction from encoder features.

        Args:
            features: Encoder output. Shape depends on backbone:
                - If 3D (B, num_patches, embed_dim): reshape to spatial grid
                - If 4D (B, C, H, W): use directly

        Returns:
            Prediction (B, 1, H, W).
        """
        if features.dim() == 3:
            features = self._reshape_to_spatial(features)

        out = self.head(features)

        # Upsample to target resolution if needed
        if out.shape[-1] != self.image_size:
            out = nn.functional.interpolate(
                out, size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False,
            )
        return out

    def _reshape_to_spatial(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape flattened patch tokens to spatial grid.

        Args:
            x: (B, num_patches, embed_dim).

        Returns:
            (B, embed_dim, grid_h, grid_w).
        """
        B, N, C = x.shape
        grid_size = int(N ** 0.5)
        return x.permute(0, 2, 1).reshape(B, C, grid_size, grid_size)


class PrithviForecaster(nn.Module):
    """Prithvi-EO-2.0 based NDVI forecaster.

    Wraps the Prithvi foundation model with input adaptation and prediction head
    for the NDVI forecasting task.
    """

    def __init__(
        self,
        model_name: str = "prithvi_eo_v2_300",
        pretrained: bool = True,
        num_input_frames: int = 4,
        freeze_backbone: bool = False,
        unfreeze_last_n: int = 4,
        image_size: int = 224,
    ):
        """Initialize Prithvi forecaster.

        Args:
            model_name: TerraTorch backbone name.
            pretrained: Whether to load pre-trained weights.
            num_input_frames: Number of temporal frames for Prithvi.
                Input sequence is temporally subsampled to this many frames.
            freeze_backbone: Whether to freeze the backbone entirely.
            unfreeze_last_n: If freeze_backbone=True, unfreeze last N transformer blocks.
            image_size: Spatial input size (Prithvi uses 224).
        """
        super().__init__()
        self.num_input_frames = num_input_frames
        self.image_size = image_size

        # Input adapter: 1ch NDVI -> 6ch (Prithvi's expected input)
        self.input_adapter = PrithviInputAdapter(out_channels=6)

        # Load Prithvi backbone via TerraTorch
        self.backbone = self._load_backbone(model_name, pretrained, num_input_frames)
        embed_dim = self._get_embed_dim()

        # Prediction head
        self.prediction_head = PrithviPredictionHead(
            input_dim=embed_dim, hidden_dim=256, image_size=image_size,
        )

        # Optionally freeze backbone
        if freeze_backbone:
            self._freeze_backbone(unfreeze_last_n)

    def _load_backbone(
        self, model_name: str, pretrained: bool, num_frames: int,
    ) -> nn.Module:
        """Load Prithvi backbone from TerraTorch registry."""
        try:
            from terratorch import BACKBONE_REGISTRY
            backbone = BACKBONE_REGISTRY.build(
                model_name, pretrained=pretrained, num_frames=num_frames,
            )
            logger.info(f"Loaded Prithvi backbone: {model_name} (pretrained={pretrained})")
            return backbone
        except ImportError:
            logger.warning(
                "TerraTorch not installed. Using placeholder backbone. "
                "Install with: pip install terratorch"
            )
            return self._create_placeholder_backbone()

    def _create_placeholder_backbone(self) -> nn.Module:
        """Create a simple placeholder when TerraTorch is not available.

        This allows testing the training pipeline without TerraTorch installed.
        """
        return nn.Sequential(
            nn.Conv2d(6, 256, 14, stride=14),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(16),
        )

    def _get_embed_dim(self) -> int:
        """Detect backbone embedding dimension."""
        if hasattr(self.backbone, "embed_dim"):
            return self.backbone.embed_dim
        if hasattr(self.backbone, "num_features"):
            return self.backbone.num_features
        # Fallback: probe with dummy input
        return 256  # Placeholder default

    def _freeze_backbone(self, unfreeze_last_n: int) -> None:
        """Freeze backbone, optionally keeping last N blocks trainable."""
        for param in self.backbone.parameters():
            param.requires_grad = False

        if unfreeze_last_n > 0 and hasattr(self.backbone, "blocks"):
            for block in self.backbone.blocks[-unfreeze_last_n:]:
                for param in block.parameters():
                    param.requires_grad = True

        frozen = sum(1 for p in self.backbone.parameters() if not p.requires_grad)
        total = sum(1 for p in self.backbone.parameters())
        logger.info(f"Backbone: {frozen}/{total} params frozen, last {unfreeze_last_n} blocks unfrozen")

    def _subsample_temporal(
        self, images: torch.Tensor,
    ) -> torch.Tensor:
        """Subsample temporal sequence to num_input_frames.

        Selects evenly-spaced frames from the input sequence.

        Args:
            images: (B, T, 1, H, W) input sequence.

        Returns:
            (B, num_input_frames, 1, H, W) subsampled sequence.
        """
        B, T, C, H, W = images.shape
        if T <= self.num_input_frames:
            return images

        # Evenly space frames
        indices = torch.linspace(0, T - 1, self.num_input_frames).long()
        return images[:, indices]

    def _center_crop(self, x: torch.Tensor) -> torch.Tensor:
        """Center crop spatial dims to image_size."""
        _, _, h, w = x.shape
        if h == self.image_size and w == self.image_size:
            return x
        start_h = (h - self.image_size) // 2
        start_w = (w - self.image_size) // 2
        return x[:, :, start_h:start_h + self.image_size, start_w:start_w + self.image_size]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: NDVI sequence (B, T, 1, H, W) or (B, T, H, W).

        Returns:
            Prediction (B, 1, H, W).
        """
        if x.dim() == 4:
            x = x.unsqueeze(2)

        B, T, C, H, W = x.shape

        # Temporal subsampling
        x = self._subsample_temporal(x)
        T_sub = x.shape[1]

        # Reshape for input adapter: (B*T, 1, H, W)
        x_flat = x.reshape(B * T_sub, C, H, W)

        # Center crop to Prithvi's expected size
        x_flat = self._center_crop(x_flat)

        # Adapt 1ch -> 6ch
        x_adapted = self.input_adapter(x_flat)  # (B*T, 6, 224, 224)

        # Reshape for backbone: depends on expected format
        # TerraTorch expects (B, C, T, H, W) for video ViT
        x_backbone = x_adapted.reshape(B, T_sub, 6, self.image_size, self.image_size)
        x_backbone = x_backbone.permute(0, 2, 1, 3, 4)  # (B, 6, T, H, W)

        # Forward through backbone
        features = self.backbone(x_backbone)

        # Handle list output (multi-scale features)
        if isinstance(features, (list, tuple)):
            features = features[-1]  # Use last scale

        # Prediction
        return self.prediction_head(features)

    def get_num_parameters(self, trainable_only: bool = False) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
