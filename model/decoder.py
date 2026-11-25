"""Observation-Anchored Decoder for forecasting.

Predicts NDVI by anchoring to same-season historical observation
and predicting a small delta. This approach:
1. Preserves spatial detail from real observations
2. Only needs to learn temporal change patterns (delta)
3. Delta is bounded and smooth, suitable for bilinear upsampling

The decoder uses attention-weighted temporal pooling which enables
interpretability analysis of temporal importance patterns.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import SeasonalViTConfig
from model.encoders.temporal import TargetTemporalEncoder


@dataclass
class TemporalAttentionAnalysis:
    """Analysis results from temporal attention weights.

    Attributes:
        attention_weights: Raw attention weights (B, T) summed across heads/patches.
        effective_history_90: Timesteps needed for 90% cumulative attention (B,).
        effective_history_50: Timesteps needed for 50% cumulative attention (B,).
        recent_5yr_weight: Sum of attention on last 5 years (20 timesteps) (B,).
        attention_entropy: Entropy of attention distribution (B,). Higher = more uniform.
        peak_timestep: Timestep with maximum attention (B,).
    """

    attention_weights: torch.Tensor
    effective_history_90: torch.Tensor
    effective_history_50: torch.Tensor
    recent_5yr_weight: torch.Tensor
    attention_entropy: torch.Tensor
    peak_timestep: torch.Tensor

    def to_dict(self) -> Dict[str, torch.Tensor]:
        """Convert to dictionary for easy logging."""
        return {
            "attention_weights": self.attention_weights,
            "effective_history_90": self.effective_history_90,
            "effective_history_50": self.effective_history_50,
            "recent_5yr_weight": self.recent_5yr_weight,
            "attention_entropy": self.attention_entropy,
            "peak_timestep": self.peak_timestep,
        }

    def summary_stats(self) -> Dict[str, float]:
        """Compute summary statistics across batch."""
        return {
            "effective_history_90_mean": self.effective_history_90.float().mean().item(),
            "effective_history_50_mean": self.effective_history_50.float().mean().item(),
            "recent_5yr_weight_mean": self.recent_5yr_weight.mean().item(),
            "attention_entropy_mean": self.attention_entropy.mean().item(),
        }


class TemporalAttentionPooling(nn.Module):
    """Attention-weighted temporal pooling with target conditioning.

    Uses a learnable query conditioned on target timestep to attend over
    temporal features, producing a weighted sum. The target condition tells
    the model which future timestep it should predict, enabling variable
    target training.

    Attention weights can be extracted for interpretability analysis.
    """

    def __init__(self, d_model: int, num_heads: int = 8) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        # Learnable query for pooling
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Project target condition into query space
        self.condition_proj = nn.Linear(d_model, d_model)

        # Multi-head attention
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
        )

        # Layer norm for stability
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        target_condition: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Pool temporal features using target-conditioned attention.

        Args:
            x: Temporal features of shape (B, T, N, D).
            target_condition: Optional target encoding of shape (B, D).
                When provided, conditions the attention query to focus on
                features relevant for predicting the specified target timestep.
            return_weights: Whether to return attention weights.

        Returns:
            pooled: Spatial features of shape (B, N, D).
            weights: Optional attention weights of shape (B, N, T).
        """
        B, T, N, D = x.shape

        # Reshape to (B*N, T, D) for attention over time per spatial location
        x_flat = x.permute(0, 2, 1, 3).reshape(B * N, T, D)

        # Expand query for batch
        query = self.query.expand(B * N, 1, D)

        # Add target condition to query if provided
        if target_condition is not None:
            # Project condition and expand for each spatial patch
            cond = self.condition_proj(target_condition)  # (B, D)
            cond = cond.unsqueeze(1).expand(B, N, D)  # (B, N, D)
            cond = cond.reshape(B * N, 1, D)  # (B*N, 1, D)
            query = query + cond  # Condition the query

        # Attend over temporal dimension
        pooled, attn_weights = self.attn(
            query, x_flat, x_flat, need_weights=return_weights
        )

        # pooled: (B*N, 1, D) -> (B, N, D)
        pooled = pooled.squeeze(1).view(B, N, D)
        pooled = self.norm(pooled)

        if return_weights and attn_weights is not None:
            # attn_weights: (B*N, 1, T) -> (B, N, T)
            attn_weights = attn_weights.squeeze(1).view(B, N, T)
            return pooled, attn_weights

        return pooled, None


class ProgressiveUpsampleBlock(nn.Module):
    """Residual 2x upsample block: learned upsample + refinement + skip connection.

    Skip path uses bilinear 2x to match spatial dimensions.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            channels, channels, kernel_size=4, stride=2, padding=1,
        )
        self.refine = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        out = self.act(self.upsample(x))
        out = self.act(self.refine(out))
        return out + skip


class ObservationAnchoredDecoder(nn.Module):
    """Decoder that predicts NDVI as anchor + delta with target conditioning.

    Formula: prediction = same_season_observation + delta * scale

    The decoder:
    1. Encodes target (year, season) to conditioning vector
    2. Pools temporal features using target-conditioned attention
    3. Applies per-patch MLP to predict delta values
    4. Upsamples delta from 8×8 to 256×256 using bilinear interpolation
    5. Adds delta to the anchor observation

    Key design choices:
    - Target conditioning tells model WHICH timestep to predict
    - Attention pooling learns which historical timesteps matter for each target
    - Attention weights can be analyzed for temporal importance insights
    - Delta is bounded by hard clamp to [-1, 1] range (avoids Tanh vanishing gradients)
    - Delta is scaled by learnable parameter (initialized small)
    - Bilinear upsampling is sufficient because delta is smooth

    Input: temporal_features (B, T, N, D), anchor observation (B, 1, H, W),
           target_season (B,), target_year (B,)
    Output: prediction (B, 1, H, W), delta (B, 1, H, W)

    Attributes:
        config: Model configuration.
        grid_size: Spatial grid size (8 for 64 patches).
        target_encoder: Encodes target (year, season) for conditioning.
        temporal_pool: Target-conditioned temporal pooling module.
        feature_mlp: MLP to refine spatial features before delta prediction.
        delta_mlp: MLP to predict per-patch delta.
        delta_scale: Learnable scale for delta (initialized small).
    """

    def __init__(self, config: SeasonalViTConfig) -> None:
        """Initialize observation-anchored decoder.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.config = config
        d_model = config.embed_dim
        hidden_dim = config.decoder_hidden_dim
        n_patches = config.num_spatial_patches
        self.grid_size = int(n_patches ** 0.5)  # 8

        # Target encoder for conditioning
        self.target_encoder = TargetTemporalEncoder(config)

        # Attention-weighted temporal pooling (now target-conditioned)
        self.temporal_pool = TemporalAttentionPooling(
            d_model=d_model,
            num_heads=config.attention_pooling_heads,
        )

        # Feature refinement MLP (refines spatial features before delta prediction)
        self.feature_mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

        # Delta prediction MLP (per-patch)
        # No Tanh — use hard clamp after scaling to avoid vanishing gradients.
        # Tanh squashes gradients near saturation, making it hard for the delta
        # pathway to learn. Hard clamping provides gradient=1 everywhere inside [-1, 1].
        self.delta_mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # Fixed delta scale (registered as buffer so it's not learnable)
        # Freezing prevents delta collapse where the optimizer suppresses
        # deltas by driving delta_scale toward zero.
        self.register_buffer(
            "delta_scale", torch.ones(1) * config.delta_scale
        )

        # Progressive upsampling: 8->16->32->64 (learned), then 64->256 (bilinear)
        upsample_channels = 32
        self.upsample_in = nn.Conv2d(1, upsample_channels, kernel_size=3, padding=1)
        self.upsample_blocks = nn.Sequential(
            ProgressiveUpsampleBlock(upsample_channels),  # 8->16
            ProgressiveUpsampleBlock(upsample_channels),  # 16->32
            ProgressiveUpsampleBlock(upsample_channels),  # 32->64
        )
        self.upsample_out = nn.Conv2d(upsample_channels, 1, kernel_size=1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights for better training stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _analyze_attention_weights(
        self, weights: torch.Tensor
    ) -> TemporalAttentionAnalysis:
        """Analyze attention weights for interpretability.

        Args:
            weights: Attention weights of shape (B, N, T).

        Returns:
            TemporalAttentionAnalysis with various metrics.
        """
        B, N, T = weights.shape

        # Average across spatial patches for per-sample temporal importance
        avg_weights = weights.mean(dim=1)  # (B, T)

        # Effective history: timesteps from end needed for X% cumulative attention
        # Reverse so we count from most recent
        reversed_weights = avg_weights.flip(dims=[1])
        cumsum = reversed_weights.cumsum(dim=1)

        # Find first index where cumsum >= threshold
        effective_90 = (cumsum >= 0.9).float().argmax(dim=1) + 1  # (B,)
        effective_50 = (cumsum >= 0.5).float().argmax(dim=1) + 1  # (B,)

        # Recent 5 years weight (last 20 timesteps = 5 years * 4 seasons)
        recent_timesteps = min(20, T)
        recent_5yr_weight = avg_weights[:, -recent_timesteps:].sum(dim=1)  # (B,)

        # Attention entropy (higher = more uniform distribution)
        # Add small epsilon for numerical stability
        eps = 1e-10
        entropy = -(avg_weights * (avg_weights + eps).log()).sum(dim=1)  # (B,)

        # Peak timestep (which timestep gets most attention)
        peak_timestep = avg_weights.argmax(dim=1)  # (B,)

        return TemporalAttentionAnalysis(
            attention_weights=avg_weights,
            effective_history_90=effective_90,
            effective_history_50=effective_50,
            recent_5yr_weight=recent_5yr_weight,
            attention_entropy=entropy,
            peak_timestep=peak_timestep,
        )

    def _upsample_delta(self, delta: torch.Tensor) -> torch.Tensor:
        """Upsample delta from patch grid to full resolution using progressive upsampling.

        Learned ConvTranspose2d blocks: 8->16->32->64, then bilinear 64->256.

        Args:
            delta: Per-patch delta of shape (B, N, 1) or (B, 1, grid, grid).

        Returns:
            Upsampled delta of shape (B, 1, H, W).
        """
        B = delta.shape[0]
        H, W = self.config.image_size

        # Reshape to (B, 1, grid_size, grid_size)
        if delta.dim() == 3:
            delta = delta.view(B, self.grid_size, self.grid_size, 1)
            delta = delta.permute(0, 3, 1, 2)  # (B, 1, grid, grid)

        # Progressive learned upsampling: 8->16->32->64
        x = self.upsample_in(delta)          # (B, 32, 8, 8)
        x = self.upsample_blocks(x)          # (B, 32, 64, 64)
        delta = self.upsample_out(x)         # (B, 1, 64, 64)

        # Final bilinear upsampling: 64->256 (delta is smooth at 64x64)
        delta = F.interpolate(
            delta,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )

        return delta  # (B, 1, H, W)

    def _pool_and_refine(
        self,
        temporal_features: torch.Tensor,
        target_season: torch.Tensor,
        target_year: torch.Tensor,
        return_weights: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run target conditioning, temporal pooling, and feature refinement.

        Returns patch-level pre-output features (B, N, 1) from delta_mlp and
        the optional attention weights for interpretability.
        """
        target_cond = self.target_encoder(target_season, target_year)  # (B, D)
        spatial_features, attn_weights = self.temporal_pool(
            temporal_features,
            target_condition=target_cond,
            return_weights=return_weights,
        )  # (B, N, D), optional (B, N, T)
        refined = self.feature_mlp(spatial_features)  # (B, N, D)
        patches = self.delta_mlp(refined)  # (B, N, 1)
        return patches, attn_weights

    def _predict_anchored(
        self, patches: torch.Tensor, anchor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Default mode: prediction = (anchor + scaled delta).clamp(-1, 1).

        The post-upsample clamp on `delta` matches the symmetry of
        `_predict_absolute` and bounds the per-batch delta magnitude to the
        NDVI-valid range. Without it, a randomly-initialized ConvTranspose2d
        upsample stack can produce out-of-range activations that drive the
        (anchor + delta) add to overflow under bf16 — the failure mode that
        crashed the `last_step` ablation in ep 1 where the target-anchor gap
        is ~2.5x larger and sustains larger gradients through the decoder.
        """
        delta_patches = (patches * self.delta_scale).clamp(-1, 1)
        delta = self._upsample_delta(delta_patches).clamp(-1, 1)
        prediction = (anchor + delta).clamp(-1, 1)
        return prediction, delta, anchor

    def _predict_absolute(
        self, patches: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Anchor-free mode: prediction is absolute NDVI (clamped to [-1, 1]).

        Delta and anchor are synthesized so downstream loss plumbing
        (delta stats, anchor_mae logging) still works without branches:
        anchor=0, delta=prediction.
        """
        abs_patches = patches.clamp(-1, 1)
        prediction = self._upsample_delta(abs_patches).clamp(-1, 1)
        anchor = torch.zeros_like(prediction)
        return prediction, prediction, anchor

    def forward(
        self,
        temporal_features: torch.Tensor,
        anchor_observation: Optional[torch.Tensor],
        target_season: torch.Tensor,
        target_year: torch.Tensor,
        return_attention_analysis: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Predict NDVI with optional anchor.

        Args:
            temporal_features: Encoded features of shape (B, T, N, D).
            anchor_observation: Anchor observation of shape (B, 1, H, W), or
                None when anchor_mode='none' (direct absolute prediction).
            target_season: Target season indices (0-3) of shape (B,).
            target_year: Target year indices of shape (B,).
            return_attention_analysis: If True, return attention analysis.

        Returns:
            Dict with 'prediction', 'delta', 'anchor' (B, 1, H, W) and
            optional 'attention_analysis'.
        """
        patches, attn_weights = self._pool_and_refine(
            temporal_features, target_season, target_year,
            return_weights=return_attention_analysis,
        )

        if anchor_observation is None:
            prediction, delta, anchor = self._predict_absolute(patches)
        else:
            prediction, delta, anchor = self._predict_anchored(
                patches, anchor_observation,
            )

        result = {"prediction": prediction, "delta": delta, "anchor": anchor}

        # 9. Optionally add attention analysis
        if return_attention_analysis and attn_weights is not None:
            result["attention_analysis"] = self._analyze_attention_weights(
                attn_weights
            )

        return result

    def get_delta_statistics(
        self, delta: torch.Tensor
    ) -> Dict[str, float]:
        """Compute statistics for delta predictions.

        Useful for monitoring training and debugging.

        Args:
            delta: Delta tensor of shape (B, 1, H, W).

        Returns:
            Dictionary with delta statistics.
        """
        return {
            "delta_mean": delta.mean().item(),
            "delta_std": delta.std().item(),
            "delta_min": delta.min().item(),
            "delta_max": delta.max().item(),
            "delta_abs_mean": delta.abs().mean().item(),
        }


def create_decoder(
    config: Optional[SeasonalViTConfig] = None,
) -> ObservationAnchoredDecoder:
    """Factory function to create decoder.

    Args:
        config: Model configuration. Uses defaults if None.

    Returns:
        Configured ObservationAnchoredDecoder instance.
    """
    if config is None:
        config = SeasonalViTConfig()
    return ObservationAnchoredDecoder(config)
