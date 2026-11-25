"""Hierarchical Spatial Encoder using Swin Transformer.

Wraps a Swin-T backbone to extract spatial features from NDVI images.
Adapts the standard 3-channel ImageNet model for single-channel NDVI input
using a learnable 1x1 convolution with statistics-aware initialization.

See docs/input_channel_adaptation.md for details on the adaptation approach.
"""

from typing import Optional

import torch
import torch.nn as nn
import timm

from model.config import SeasonalViTConfig


class HierarchicalSpatialEncoder(nn.Module):
    """Swin-T based spatial encoder for NDVI images.

    Extracts hierarchical spatial features from single-channel NDVI images.
    Uses a pre-trained Swin-T backbone with adapted input projection.

    Input: (B*T, 1, H, W) - Flattened batch of grayscale NDVI images
    Output: (B*T, N, D) - Spatial features where N=64 (8×8 grid), D=768

    Attributes:
        config: Model configuration.
        input_proj: Projects 1-channel input to 3-channel for Swin-T.
        backbone: Swin-T backbone from timm.
    """

    def __init__(self, config: SeasonalViTConfig) -> None:
        """Initialize the spatial encoder.

        Args:
            config: Model configuration containing image_size, embed_dim, etc.
        """
        super().__init__()
        self.config = config

        # Project single-channel NDVI to 3-channel for Swin-T compatibility
        # Uses bias to handle distribution shift between NDVI and ImageNet
        self.input_proj = nn.Conv2d(
            in_channels=1,
            out_channels=3,
            kernel_size=1,
            bias=True,
        )
        self._init_input_proj()

        # Load Swin-T backbone
        self.backbone = self._create_swin_backbone()

        # Enable gradient checkpointing to reduce memory when processing B×T images
        if config.use_gradient_checkpointing:
            self.backbone.set_grad_checkpointing(enable=True)

    def _init_input_proj(self) -> None:
        """Initialize input projection to map NDVI to ImageNet-like distribution.

        NDVI values typically range [-1, 1] with vegetation around [0.2, 0.9].
        ImageNet RGB channels have mean ~0.449 and std ~0.226.

        The initialization applies an affine transform:
            output = scale * input + bias

        Where scale maps NDVI std to ImageNet std, and bias shifts the mean.
        """
        scale = self.config.imagenet_std / self.config.ndvi_std
        bias = self.config.imagenet_mean - scale * self.config.ndvi_mean

        nn.init.constant_(self.input_proj.weight, scale)
        nn.init.constant_(self.input_proj.bias, bias)

    def _create_swin_backbone(self) -> nn.Module:
        """Create and configure Swin-T backbone from timm.

        Returns:
            Configured Swin-T model without classification head.
        """
        model = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=self.config.swin_pretrained,
            num_classes=0,  # Remove classification head
            img_size=self.config.image_size[0],
        )
        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial features from NDVI images.

        Args:
            x: Input tensor of shape (B*T, 1, H, W).
               Values should be in [-1, 1] range (valid NDVI).

        Returns:
            Spatial features of shape (B*T, N, D) where:
            - N = num_spatial_patches (64 for 8×8 grid)
            - D = embed_dim (768 for Swin-T)
        """
        # Project 1-channel to 3-channel
        # Let mixed precision handle dtypes naturally - input_proj supports both fp16/fp32
        x = self.input_proj(x)  # (B*T, 3, H, W)

        # Extract features through Swin-T
        # Swin-T returns (B, H/32, W/32, D) for 256x256 input -> (B, 8, 8, 768)
        features = self.backbone.forward_features(x)

        # Flatten spatial dimensions: (B, 8, 8, 768) -> (B, 64, 768)
        batch_size = features.shape[0]
        features = features.view(batch_size, -1, self.config.embed_dim)

        return features

    def get_output_shape(self, batch_size: int) -> tuple:
        """Return expected output shape for given batch size.

        Args:
            batch_size: Number of images in batch (B*T).

        Returns:
            Tuple of (batch_size, num_patches, embed_dim).
        """
        return (
            batch_size,
            self.config.num_spatial_patches,
            self.config.embed_dim,
        )


def create_spatial_encoder(
    config: Optional[SeasonalViTConfig] = None,
) -> HierarchicalSpatialEncoder:
    """Factory function to create a spatial encoder.

    Args:
        config: Model configuration. Uses defaults if None.

    Returns:
        Configured HierarchicalSpatialEncoder instance.
    """
    if config is None:
        config = SeasonalViTConfig()
    return HierarchicalSpatialEncoder(config)
