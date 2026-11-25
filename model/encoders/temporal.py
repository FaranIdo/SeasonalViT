"""Seasonal Positional Encoding for temporal sequences.

Encodes temporal position with seasonal awareness using:
- Learnable season embeddings (0-3 for spring/summer/fall/winter)
- Learnable sensor era embeddings (0-3 for TM/ETM+/OLI/OLI-2)
- Cyclical encoding for smooth season transitions
- Multi-frequency year encoding for long-term patterns
- Sinusoidal sequence positional encoding
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from model.config import SeasonalViTConfig


class SeasonalPositionalEncoding(nn.Module):
    """Temporal positional encoding with seasonal awareness.

    Combines five encoding components into a unified temporal representation:
    1. Season embedding: Learnable embedding for each season (0-3)
    2. Sensor era embedding: Learnable embedding for Landsat sensors (TM/ETM+/OLI/OLI-2)
    3. Cyclical encoding: sin/cos of season angle for smooth transitions
    4. Year encoding: Multi-frequency encoding for long-term patterns
    5. Sequence PE: Standard sinusoidal positional encoding

    Input: season_ids (B, T), year_indices (B, T)
    Output: (B, T, D) temporal encoding

    Attributes:
        config: Model configuration.
        season_embed: Learnable season embeddings.
        sensor_era_embed: Learnable sensor era embeddings.
        sensor_era_boundaries: Year indices where sensor eras change.
        season_cyclic_proj: Projects 2D cyclical encoding to d_model/5.
        year_proj: Projects multi-frequency year encoding to d_model/5.
        seq_proj: Projects sinusoidal PE to d_model/5.
        output_proj: Combines all components to final d_model.
    """

    def __init__(self, config: SeasonalViTConfig) -> None:
        """Initialize seasonal positional encoding.

        Args:
            config: Model configuration with embed_dim, num_seasons, etc.
        """
        super().__init__()
        self.config = config
        d_model = config.embed_dim
        n_seasons = config.num_seasons
        max_timesteps = config.num_timesteps

        # Dimension allocation (5 components)
        d_season_embed = d_model // 5
        d_sensor_era = d_model // 5
        d_season_cyclic = d_model // 5
        d_year = d_model // 5
        d_sequence = d_model - d_season_embed - d_sensor_era - d_season_cyclic - d_year

        # 1. Season embedding (learnable)
        self.season_embed = nn.Embedding(n_seasons, d_season_embed)

        # 2. Sensor era embedding (learnable)
        self.sensor_era_embed = nn.Embedding(config.num_sensor_eras, d_sensor_era)
        self.register_buffer(
            "sensor_era_boundaries",
            torch.tensor(config.sensor_era_boundaries, dtype=torch.long)
        )

        # 3. Cyclical season encoding
        self.season_cyclic_proj = nn.Linear(2, d_season_cyclic)

        # 4. Year encoding (multi-frequency)
        self.n_year_freqs = 10
        self.year_proj = nn.Linear(self.n_year_freqs * 2, d_year)

        # 5. Sequence positional encoding (sinusoidal)
        self.seq_proj = nn.Linear(d_model, d_sequence)
        self.register_buffer(
            "seq_pe",
            self._generate_sinusoidal_pe(max_timesteps, d_model)
        )

        # Output projection to combine all components
        total_dim = d_season_embed + d_sensor_era + d_season_cyclic + d_year + d_sequence
        self.output_proj = nn.Linear(total_dim, d_model)

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

    def _generate_sinusoidal_pe(
        self, max_len: int, d_model: int
    ) -> torch.Tensor:
        """Generate standard sinusoidal positional encoding.

        Args:
            max_len: Maximum sequence length.
            d_model: Embedding dimension.

        Returns:
            Positional encoding tensor of shape (max_len, d_model).
        """
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def _encode_years(self, years: torch.Tensor) -> torch.Tensor:
        """Encode year indices with multi-frequency sinusoids.

        Uses exponentially spaced frequencies to capture both short-term
        and long-term temporal patterns.

        Args:
            years: Year indices of shape (B, T).

        Returns:
            Year encoding of shape (B, T, d_year).
        """
        # Frequencies: 2^0, 2^1, ..., 2^(n_year_freqs-1)
        freqs = (2 ** torch.arange(
            self.n_year_freqs, device=years.device, dtype=torch.float
        ))
        # Normalize years to reasonable range
        normalized_years = years.float() / self.config.num_years
        angles = 2 * math.pi * normalized_years.unsqueeze(-1) * freqs
        enc = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.year_proj(enc)

    def _encode_season_cyclical(
        self, season_ids: torch.Tensor
    ) -> torch.Tensor:
        """Encode seasons as points on unit circle.

        Maps season 0-3 to angles 0, π/2, π, 3π/2 for smooth transitions.

        Args:
            season_ids: Season indices of shape (B, T), values 0-3.

        Returns:
            Cyclical encoding of shape (B, T, d_season_cyclic).
        """
        n_seasons = self.config.num_seasons
        season_angle = 2 * math.pi * season_ids.float() / n_seasons
        season_cyclic = torch.stack([
            torch.sin(season_angle),
            torch.cos(season_angle)
        ], dim=-1)
        return self.season_cyclic_proj(season_cyclic)

    def _get_sensor_era(self, year_indices: torch.Tensor) -> torch.Tensor:
        """Map year indices to sensor era indices (0-3).

        Sensor eras correspond to different Landsat sensors:
        - Era 0 (TM): years 0-8 (1990-1998)
        - Era 1 (ETM+): years 9-22 (1999-2012)
        - Era 2 (OLI): years 23-31 (2013-2021)
        - Era 3 (OLI-2): years 32-34 (2022-2024)

        Args:
            year_indices: Year indices of shape (B, T).

        Returns:
            Sensor era indices of shape (B, T), values 0-3.
        """
        era_indices = torch.zeros_like(year_indices)
        for i, boundary in enumerate(self.sensor_era_boundaries):
            era_indices = torch.where(year_indices >= boundary, i + 1, era_indices)
        return era_indices

    def forward(
        self,
        season_ids: torch.Tensor,
        year_indices: torch.Tensor,
        seq_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute temporal positional encoding.

        Args:
            season_ids: Season indices (0-3) of shape (B, T) or (T,).
            year_indices: Year indices of shape (B, T) or (T,).
            seq_positions: Optional explicit sequence positions of shape (B, T).
                          If None, uses 0, 1, 2, ..., T-1.

        Returns:
            Combined temporal encoding of shape (B, T, embed_dim).
        """
        # Handle 1D inputs (broadcast to batch)
        if season_ids.dim() == 1:
            season_ids = season_ids.unsqueeze(0)
        if year_indices.dim() == 1:
            year_indices = year_indices.unsqueeze(0)

        B, T = season_ids.shape

        # 1. Season embedding
        season_emb = self.season_embed(season_ids)  # (B, T, d_season_embed)

        # 2. Sensor era embedding
        sensor_era_ids = self._get_sensor_era(year_indices)
        sensor_era_emb = self.sensor_era_embed(sensor_era_ids)  # (B, T, d_sensor_era)

        # 3. Cyclical season encoding
        season_cyclic = self._encode_season_cyclical(season_ids)

        # 4. Year encoding
        year_enc = self._encode_years(year_indices)  # (B, T, d_year)

        # 5. Sequence positional encoding
        if seq_positions is None:
            seq_positions = torch.arange(T, device=season_ids.device)
            seq_positions = seq_positions.unsqueeze(0).expand(B, -1)
        seq_enc = self.seq_proj(self.seq_pe[seq_positions])  # (B, T, d_sequence)

        # Combine all components
        combined = torch.cat([
            season_emb, sensor_era_emb, season_cyclic, year_enc, seq_enc
        ], dim=-1)

        return self.output_proj(combined)

    def get_output_dim(self) -> int:
        """Return output embedding dimension.

        Returns:
            Embedding dimension (same as config.embed_dim).
        """
        return self.config.embed_dim


class TargetTemporalEncoder(nn.Module):
    """Encodes target (year, season) for decoder conditioning.

    Produces a conditioning vector that tells the decoder WHICH future
    timestep to predict. Uses same encoding strategy as SeasonalPositionalEncoding
    but with separate weights, allowing specialization for target vs input encoding.

    This enables variable target training where the model learns to predict
    different future timesteps based on the target conditioning.

    Input: target_season (B,), target_year (B,)
    Output: (B, D) conditioning vector

    Attributes:
        config: Model configuration.
        season_embed: Learnable season embedding.
        season_cyclic_proj: Projects cyclical season encoding.
        year_proj: Projects multi-frequency year encoding.
        output_proj: Combines components to final embedding.
    """

    def __init__(self, config: SeasonalViTConfig) -> None:
        """Initialize target temporal encoder.

        Args:
            config: Model configuration with embed_dim, num_seasons, etc.
        """
        super().__init__()
        self.config = config
        d_model = config.embed_dim
        n_seasons = config.num_seasons

        # Dimension allocation (3 components for target)
        d_season_embed = d_model // 3
        d_season_cyclic = d_model // 3
        d_year = d_model - d_season_embed - d_season_cyclic

        # 1. Season embedding (learnable)
        self.season_embed = nn.Embedding(n_seasons, d_season_embed)

        # 2. Cyclical season encoding (smooth transitions)
        self.season_cyclic_proj = nn.Linear(2, d_season_cyclic)

        # 3. Year encoding (multi-frequency)
        self.n_year_freqs = 10
        self.year_proj = nn.Linear(self.n_year_freqs * 2, d_year)

        # Output projection
        total_dim = d_season_embed + d_season_cyclic + d_year
        self.output_proj = nn.Linear(total_dim, d_model)

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

    def forward(
        self,
        target_season: torch.Tensor,
        target_year: torch.Tensor,
    ) -> torch.Tensor:
        """Encode target timestep for decoder conditioning.

        Args:
            target_season: Target season indices (0-3) of shape (B,).
            target_year: Target year indices of shape (B,).

        Returns:
            Target conditioning vector of shape (B, embed_dim).
        """
        # 1. Season embedding
        season_emb = self.season_embed(target_season)  # (B, d_season_embed)

        # 2. Cyclical season encoding
        n_seasons = self.config.num_seasons
        season_angle = 2 * math.pi * target_season.float() / n_seasons
        season_cyclic = torch.stack([
            torch.sin(season_angle),
            torch.cos(season_angle)
        ], dim=-1)  # (B, 2)
        season_cyclic = self.season_cyclic_proj(season_cyclic)  # (B, d_season_cyclic)

        # 3. Year encoding (multi-frequency)
        freqs = (2 ** torch.arange(
            self.n_year_freqs, device=target_year.device, dtype=torch.float
        ))
        normalized_years = target_year.float() / self.config.num_years
        angles = 2 * math.pi * normalized_years.unsqueeze(-1) * freqs
        year_enc = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        year_enc = self.year_proj(year_enc)  # (B, d_year)

        # Combine all components
        combined = torch.cat([season_emb, season_cyclic, year_enc], dim=-1)
        return self.output_proj(combined)  # (B, d_model)


def create_temporal_encoding(
    config: Optional[SeasonalViTConfig] = None,
) -> SeasonalPositionalEncoding:
    """Factory function to create temporal encoding.

    Args:
        config: Model configuration. Uses defaults if None.

    Returns:
        Configured SeasonalPositionalEncoding instance.
    """
    if config is None:
        config = SeasonalViTConfig()
    return SeasonalPositionalEncoding(config)


def create_target_encoder(
    config: Optional[SeasonalViTConfig] = None,
) -> TargetTemporalEncoder:
    """Factory function to create target temporal encoder.

    Args:
        config: Model configuration. Uses defaults if None.

    Returns:
        Configured TargetTemporalEncoder instance.
    """
    if config is None:
        config = SeasonalViTConfig()
    return TargetTemporalEncoder(config)
