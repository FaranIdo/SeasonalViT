"""Location Encoding for geographic coordinates.

Encodes latitude/longitude with:
- 3D Cartesian coordinates on unit sphere
- Hemisphere embedding for seasonal inversion awareness
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from model.config import SeasonalViTConfig


class LocationEncoder(nn.Module):
    """Geographic location encoder with hemisphere awareness.

    Converts lat/lon coordinates to a learned embedding that captures:
    1. 3D position on Earth's surface (Cartesian coordinates)
    2. Hemisphere (important for season inversion)

    This is critical for global models where seasons are inverted
    between northern and southern hemispheres.

    Input: lat (B,), lon (B,) in degrees
    Output: location_encoding (B, D), hemisphere (B,)

    Attributes:
        config: Model configuration.
        coord_proj: MLP to project 3D Cartesian coordinates.
        hemisphere_embed: Learnable embedding for N/S hemisphere.
        combine: Linear layer to combine coord and hemisphere encodings.
    """

    def __init__(self, config: SeasonalViTConfig) -> None:
        """Initialize location encoder.

        Args:
            config: Model configuration with embed_dim, location_embed_dim, etc.
        """
        super().__init__()
        self.config = config
        d_model = config.embed_dim
        d_location = config.location_embed_dim

        # 3D Cartesian coordinate projection
        self.coord_proj = nn.Sequential(
            nn.Linear(3, d_location),
            nn.GELU(),
            nn.Linear(d_location, d_model),
        )

        # Hemisphere embedding (0 = Southern, 1 = Northern)
        self.hemisphere_embed = nn.Embedding(2, d_model)

        # Combine coordinate and hemisphere encodings
        self.combine = nn.Linear(d_model * 2, d_model)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights for better training stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _lat_lon_to_cartesian(
        self, lat: torch.Tensor, lon: torch.Tensor
    ) -> torch.Tensor:
        """Convert lat/lon to 3D Cartesian coordinates on unit sphere.

        Args:
            lat: Latitude in degrees (-90 to 90).
            lon: Longitude in degrees (-180 to 180).

        Returns:
            Cartesian coordinates (x, y, z) of shape (B, 3).
        """
        lat_rad = lat * (math.pi / 180.0)
        lon_rad = lon * (math.pi / 180.0)

        x = torch.cos(lat_rad) * torch.cos(lon_rad)
        y = torch.cos(lat_rad) * torch.sin(lon_rad)
        z = torch.sin(lat_rad)

        return torch.stack([x, y, z], dim=-1)

    def forward(
        self, lat: torch.Tensor, lon: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode geographic location.

        Args:
            lat: Latitude in degrees of shape (B,) or scalar.
            lon: Longitude in degrees of shape (B,) or scalar.

        Returns:
            Tuple of:
            - location_encoding: Shape (B, embed_dim)
            - hemisphere: Shape (B,), 0=South, 1=North
        """
        # Ensure batch dimension
        if lat.dim() == 0:
            lat = lat.unsqueeze(0)
        if lon.dim() == 0:
            lon = lon.unsqueeze(0)

        # Convert to 3D Cartesian
        coords = self._lat_lon_to_cartesian(lat, lon)  # (B, 3)

        # Project coordinates
        coord_enc = self.coord_proj(coords)  # (B, d_model)

        # Determine hemisphere (lat >= 0 is Northern)
        hemisphere = (lat >= 0).long()  # (B,)
        hemi_enc = self.hemisphere_embed(hemisphere)  # (B, d_model)

        # Combine encodings
        combined = torch.cat([coord_enc, hemi_enc], dim=-1)  # (B, d_model*2)
        location_encoding = self.combine(combined)  # (B, d_model)

        return location_encoding, hemisphere

    def get_output_dim(self) -> int:
        """Return output embedding dimension.

        Returns:
            Embedding dimension (same as config.embed_dim).
        """
        return self.config.embed_dim


def create_location_encoder(
    config: Optional[SeasonalViTConfig] = None,
) -> LocationEncoder:
    """Factory function to create location encoder.

    Args:
        config: Model configuration. Uses defaults if None.

    Returns:
        Configured LocationEncoder instance.
    """
    if config is None:
        config = SeasonalViTConfig()
    return LocationEncoder(config)
