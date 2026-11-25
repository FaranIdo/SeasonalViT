"""Dual-Scale Temporal Transformer for phenological and climatological modeling.

Processes temporal sequences at two scales:
1. Cross-season pathway (phenological): Within-year attention (4 seasons)
2. Cross-year pathway (climatological): Across-year attention (n_years timesteps)
3. Cross-attention fusion to combine both pathways
"""

from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from model.config import SeasonalViTConfig


class DualScaleTemporalTransformer(nn.Module):
    """Dual-scale temporal transformer for long-term NDVI sequences.

    Processes 35 years × 4 seasons = 140 timesteps using two parallel pathways:
    1. Cross-season pathway (phenological): Captures within-year patterns
       (spring→summer→fall→winter)
    2. Cross-year pathway (climatological): Captures across-year patterns
       (same season across years)

    The two pathways are fused via cross-attention to combine phenological
    patterns with climatological trends.

    Input: (B, T, N, D) - spatial features with temporal dimension
    Output: (B, T, N, D) - temporally-aware features

    Attributes:
        config: Model configuration.
        cross_season_transformer: Encoder for within-year patterns (phenological).
        cross_year_transformer: Encoder for across-year patterns (climatological).
        cross_attn: Cross-attention for pathway fusion.
        fusion_mlp: MLP to combine fused features.
    """

    def __init__(self, config: SeasonalViTConfig) -> None:
        """Initialize dual-scale temporal transformer.

        Args:
            config: Model configuration with temporal_depth, temporal_heads, etc.
        """
        super().__init__()
        self.config = config
        d_model = config.embed_dim
        n_heads = config.temporal_heads
        n_layers = config.temporal_depth
        n_seasons = config.num_seasons
        dropout = config.temporal_dropout
        mlp_ratio = config.temporal_mlp_ratio

        self.n_seasons = n_seasons
        self.use_checkpointing = config.use_gradient_checkpointing
        self.chunk_size = config.temporal_chunk_size  # 0 = no chunking
        self.pathway = getattr(config, "temporal_pathway", "both")

        # Single-pathway ablations get all n_layers (vs n_layers/2 for dual).
        # For "both" mode keep the original half-split to preserve checkpoint
        # compatibility with existing pre-trained weights.
        dual_layers = n_layers // 2
        single_layers = n_layers

        # Cross-season pathway (phenological): within-year attention
        if self.pathway in ("both", "cross_season"):
            cross_season_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=int(d_model * mlp_ratio),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,  # Pre-LN for better training stability
            )
            layers = dual_layers if self.pathway == "both" else single_layers
            self.cross_season_transformer = nn.TransformerEncoder(
                cross_season_layer, num_layers=layers
            )

        # Cross-year pathway (climatological): across-year attention
        if self.pathway in ("both", "cross_year"):
            cross_year_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=int(d_model * mlp_ratio),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            layers = dual_layers if self.pathway == "both" else single_layers
            self.cross_year_transformer = nn.TransformerEncoder(
                cross_year_layer, num_layers=layers
            )

        # Fusion only needed when both pathways are active
        if self.pathway == "both":
            # Cross-attention for fusion
            self.cross_attn = nn.MultiheadAttention(
                d_model, n_heads, dropout=dropout, batch_first=True
            )

            # Fusion MLP
            self.fusion_mlp = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
            )

            # Layer norm for fusion
            self.fusion_norm = nn.LayerNorm(d_model)

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

    def _chunked_transformer(
        self,
        x: torch.Tensor,
        transformer: nn.TransformerEncoder,
    ) -> torch.Tensor:
        """Apply transformer with optional chunking to avoid OOM.

        Args:
            x: Input tensor of shape (num_sequences, seq_len, D).
            transformer: The transformer encoder to apply.

        Returns:
            Transformed tensor of same shape as input.
        """
        if self.chunk_size <= 0 or x.shape[0] <= self.chunk_size:
            return transformer(x)

        # Process in chunks
        outputs = []
        for i in range(0, x.shape[0], self.chunk_size):
            chunk = x[i : i + self.chunk_size]
            outputs.append(transformer(chunk))
        return torch.cat(outputs, dim=0)

    def _process_cross_season(
        self, x: torch.Tensor, n_years: int
    ) -> torch.Tensor:
        """Process input through cross-season (phenological) pathway.

        Args:
            x: Input tensor of shape (B, n_years, n_seasons, N, D).
            n_years: Number of years.

        Returns:
            Cross-season features of shape (B, n_years, n_seasons, N, D).
        """
        B, _, n_seasons, N, D = x.shape

        # Vectorized: process all years at once
        # (B, n_years, 4, N, D) -> (B, n_years, N, 4, D) -> (B*n_years*N, 4, D)
        x_flat = x.permute(0, 1, 3, 2, 4).contiguous().view(B * n_years * N, n_seasons, D)

        # Apply cross-season transformer (with optional chunking to avoid OOM)
        cross_season_features = self._chunked_transformer(
            x_flat, self.cross_season_transformer
        )

        # Reshape back: (B*n_years*N, 4, D) -> (B, n_years, N, 4, D) -> (B, n_years, 4, N, D)
        cross_season_features = cross_season_features.view(B, n_years, N, n_seasons, D)
        cross_season_features = cross_season_features.permute(0, 1, 3, 2, 4).contiguous()

        return cross_season_features

    def _process_cross_year(
        self, x: torch.Tensor, n_years: int
    ) -> torch.Tensor:
        """Process input through cross-year (climatological) pathway.

        Args:
            x: Input tensor of shape (B, n_years, n_seasons, N, D).
            n_years: Number of years.

        Returns:
            Cross-year features of shape (B, n_years, n_seasons, N, D).
        """
        B, _, n_seasons, N, D = x.shape

        # Vectorized: process all seasons at once
        # (B, n_years, 4, N, D) -> (B, 4, N, n_years, D) -> (B*4*N, n_years, D)
        x_flat = x.permute(0, 2, 3, 1, 4).contiguous().view(B * n_seasons * N, n_years, D)

        # Apply cross-year transformer (with optional chunking to avoid OOM)
        cross_year_features = self._chunked_transformer(
            x_flat, self.cross_year_transformer
        )

        # Reshape back: (B*4*N, n_years, D) -> (B, 4, N, n_years, D) -> (B, n_years, 4, N, D)
        cross_year_features = cross_year_features.view(B, n_seasons, N, n_years, D)
        cross_year_features = cross_year_features.permute(0, 3, 1, 2, 4).contiguous()

        return cross_year_features

    def _fuse_pathways(
        self,
        cross_season_features: torch.Tensor,
        cross_year_features: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse cross-season and cross-year pathways via cross-attention.

        Args:
            cross_season_features: Shape (B, T, N, D).
            cross_year_features: Shape (B, T, N, D).

        Returns:
            Fused features of shape (B, T, N, D).
        """
        B, T, N, D = cross_season_features.shape

        # Flatten for cross-attention: (B*T, N, D)
        cross_season_q = cross_season_features.view(B * T, N, D)
        cross_year_kv = cross_year_features.view(B * T, N, D)

        # Cross-attention: cross-season queries attend to cross-year keys/values
        fused, _ = self.cross_attn(cross_season_q, cross_year_kv, cross_year_kv)
        fused = fused.view(B, T, N, D)

        # Combine via MLP
        combined = torch.cat([cross_season_features, fused], dim=-1)  # (B, T, N, 2D)
        output = self.fusion_mlp(combined)  # (B, T, N, D)
        output = self.fusion_norm(output + cross_season_features)  # Residual

        return output

    def forward(
        self,
        spatial_features: torch.Tensor,
        temporal_encoding: torch.Tensor,
    ) -> torch.Tensor:
        """Process spatial features through dual-scale temporal transformer.

        Args:
            spatial_features: Shape (B, T, N, D) where T = n_years * n_seasons.
            temporal_encoding: Shape (B, T, D) positional encoding.

        Returns:
            Temporally-aware features of shape (B, T, N, D).
        """
        B, T, N, D = spatial_features.shape
        n_years = T // self.n_seasons

        # Add temporal encoding (broadcast over spatial dimension)
        x = spatial_features + temporal_encoding.unsqueeze(2)  # (B, T, N, D)

        # Reshape to (B, n_years, n_seasons, N, D)
        x_reshaped = x.view(B, n_years, self.n_seasons, N, D)

        run_season = self.pathway in ("both", "cross_season")
        run_year = self.pathway in ("both", "cross_year")

        # Process through active pathway(s) (with optional gradient checkpointing)
        use_ckpt = self.use_checkpointing and self.training
        cross_season_out = None
        cross_year_out = None
        if run_season:
            if use_ckpt:
                cross_season_out = checkpoint(
                    self._process_cross_season, x_reshaped, n_years,
                    use_reentrant=False,
                )
            else:
                cross_season_out = self._process_cross_season(x_reshaped, n_years)
        if run_year:
            if use_ckpt:
                cross_year_out = checkpoint(
                    self._process_cross_year, x_reshaped, n_years,
                    use_reentrant=False,
                )
            else:
                cross_year_out = self._process_cross_year(x_reshaped, n_years)

        # Single-pathway ablations: return the active pathway directly
        if self.pathway == "cross_season":
            assert cross_season_out is not None
            return cross_season_out.view(B, T, N, D)
        if self.pathway == "cross_year":
            assert cross_year_out is not None
            return cross_year_out.view(B, T, N, D)

        # Both: flatten and fuse
        assert cross_season_out is not None and cross_year_out is not None
        cross_season_flat = cross_season_out.view(B, T, N, D)
        cross_year_flat = cross_year_out.view(B, T, N, D)
        return self._fuse_pathways(cross_season_flat, cross_year_flat)

    def get_output_shape(
        self, batch_size: int, num_timesteps: int, num_patches: int
    ) -> Tuple[int, int, int, int]:
        """Return expected output shape.

        Args:
            batch_size: Batch size.
            num_timesteps: Number of timesteps (T).
            num_patches: Number of spatial patches (N).

        Returns:
            Tuple of (batch_size, num_timesteps, num_patches, embed_dim).
        """
        return (batch_size, num_timesteps, num_patches, self.config.embed_dim)


def create_temporal_transformer(
    config: SeasonalViTConfig = None,
) -> DualScaleTemporalTransformer:
    """Factory function to create temporal transformer.

    Args:
        config: Model configuration. Uses defaults if None.

    Returns:
        Configured DualScaleTemporalTransformer instance.
    """
    if config is None:
        config = SeasonalViTConfig()
    return DualScaleTemporalTransformer(config)
