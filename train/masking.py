"""Masking Strategies for MAE-style Pre-training.

Supports two modes:
1. Efficient temporal masking (default): Only visible timesteps pass through
   the spatial encoder, giving ~2x speedup with 50% masking ratio.
2. Legacy 3-level masking: Encodes all timesteps, then masks. Less efficient
   but allows patch-level masking within images.
"""

from dataclasses import dataclass
from typing import Tuple

import torch

from model.config import PretrainingConfig


@dataclass
class EfficientMaskOutput:
    """Output from efficient temporal masking.

    In efficient mode, we mask entire timesteps BEFORE spatial encoding,
    so the encoder only processes visible timesteps.

    Attributes:
        visible_mask: Shape (B, T), True = visible (will be encoded)
        masked_indices: Shape (B, num_masked), indices of masked timesteps
        visible_indices: Shape (B, num_visible), indices of visible timesteps
        num_visible: Number of visible timesteps per sample
        num_masked: Number of masked timesteps per sample
    """

    visible_mask: torch.Tensor
    masked_indices: torch.Tensor
    visible_indices: torch.Tensor
    num_visible: int
    num_masked: int


def generate_efficient_temporal_mask(
    batch_size: int,
    num_timesteps: int,
    config: PretrainingConfig = None,
    device: torch.device = None,
) -> EfficientMaskOutput:
    """Generate efficient temporal mask (Levels 2+3) for pre-encoding masking.

    Combines Level 2 (temporal masking) and Level 3 (rare image masking) into
    a single temporal mask applied BEFORE the spatial encoder. Only visible
    timesteps are encoded, giving significant speedup.

    Args:
        batch_size: Number of samples in batch.
        num_timesteps: Number of temporal positions (T).
        config: Pre-training config with mask ratios. Uses defaults if None.
        device: Device for tensor creation.

    Returns:
        EfficientMaskOutput with visible/masked indices and counts.
    """
    if config is None:
        config = PretrainingConfig()

    # Level 2: Base temporal masking
    base_num_masked = int(num_timesteps * config.temporal_mask_ratio)

    # Level 3: Additional rare masking (applied per-sample)
    # We add extra masked timesteps with probability image_mask_prob
    extra_mask_per_sample = (
        torch.rand(batch_size, num_timesteps, device=device) < config.image_mask_prob
    )

    # Generate random permutation for base masking
    noise = torch.rand(batch_size, num_timesteps, device=device)
    ids_sorted = torch.argsort(noise, dim=-1)

    # Start with base temporal mask
    base_masked_indices = ids_sorted[:, :base_num_masked]
    visible_mask = torch.ones(batch_size, num_timesteps, dtype=torch.bool, device=device)
    visible_mask.scatter_(1, base_masked_indices, False)

    # Apply Level 3: additional masking
    visible_mask = visible_mask & ~extra_mask_per_sample

    # Use fixed num_visible based on config ratios to avoid GPU sync
    # This is conservative: accounts for base masking + expected Level 3 masking
    # The priority-based selection handles cases where actual visible differs
    expected_extra = int(num_timesteps * config.image_mask_prob)
    num_visible = num_timesteps - base_num_masked - expected_extra
    num_visible = max(1, num_visible)  # ensure at least 1 visible
    num_masked = num_timesteps - num_visible

    # Vectorized approach: use random noise to select which visible indices to keep
    # For samples with more visible than min, we randomly drop some by assigning them
    # high noise values that push them out of the top-k selection

    # Create noise for tie-breaking: visible positions get low noise, masked get high
    # We use the existing visible_mask to prioritize visible positions
    selection_noise = torch.rand(batch_size, num_timesteps, device=device)

    # Visible positions: noise in [0, 1), masked positions: noise in [1, 2)
    # This ensures all visible positions are selected before any masked ones
    selection_priority = torch.where(
        visible_mask,
        selection_noise,  # [0, 1) for visible
        selection_noise + 1.0,  # [1, 2) for masked (will never be selected as visible)
    )

    # Sort by priority to get indices
    sorted_indices = torch.argsort(selection_priority, dim=-1)

    # First num_visible indices are visible, rest are masked
    visible_indices = sorted_indices[:, :num_visible]  # (B, num_visible)
    masked_indices = sorted_indices[:, num_visible:]  # (B, num_masked)

    # Sort indices within each sample for consistent ordering
    visible_indices = visible_indices.sort(dim=-1)[0]
    masked_indices = masked_indices.sort(dim=-1)[0]

    # Build final visible mask from selected indices
    final_visible_mask = torch.zeros(batch_size, num_timesteps, dtype=torch.bool, device=device)
    # Use scatter to set True at visible indices
    final_visible_mask.scatter_(1, visible_indices, True)

    return EfficientMaskOutput(
        visible_mask=final_visible_mask,
        masked_indices=masked_indices,
        visible_indices=visible_indices,
        num_visible=num_visible,
        num_masked=num_masked,
    )


@dataclass
class MaskOutput:
    """Output from mask generation.

    Attributes:
        patch_mask: Shape (B, T, N), True = masked patch
        temporal_mask: Shape (B, T), True = fully masked timestep
        combined_mask: Shape (B, T, N), combined mask for all levels
    """

    patch_mask: torch.Tensor
    temporal_mask: torch.Tensor
    combined_mask: torch.Tensor


def generate_patch_mask(
    batch_size: int,
    num_timesteps: int,
    num_patches: int,
    mask_ratio: float,
    device: torch.device = None,
) -> torch.Tensor:
    """Generate random patch-level mask.

    Randomly masks a percentage of patches within each image independently.
    This is the primary masking level for local spatial reconstruction.

    Args:
        batch_size: Number of samples in batch.
        num_timesteps: Number of temporal positions.
        num_patches: Number of spatial patches per image.
        mask_ratio: Fraction of patches to mask (e.g., 0.75).
        device: Device for tensor creation.

    Returns:
        Boolean mask of shape (B, T, N), True = masked.
    """
    num_mask = int(num_patches * mask_ratio)

    # Generate random indices for each image
    noise = torch.rand(batch_size, num_timesteps, num_patches, device=device)
    # Sort and take top num_mask indices as masked
    ids_sorted = torch.argsort(noise, dim=-1)
    mask = ids_sorted < num_mask  # True for masked positions

    return mask


def generate_temporal_mask(
    batch_size: int,
    num_timesteps: int,
    mask_ratio: float,
    device: torch.device = None,
) -> torch.Tensor:
    """Generate temporal-level mask (entire timesteps).

    Randomly masks a percentage of entire timesteps, forcing the model
    to reconstruct full images from temporal context only.

    Args:
        batch_size: Number of samples in batch.
        num_timesteps: Number of temporal positions.
        mask_ratio: Fraction of timesteps to mask (e.g., 0.25).
        device: Device for tensor creation.

    Returns:
        Boolean mask of shape (B, T), True = masked timestep.
    """
    num_mask = int(num_timesteps * mask_ratio)

    # Generate random indices for each sample
    noise = torch.rand(batch_size, num_timesteps, device=device)
    ids_sorted = torch.argsort(noise, dim=-1)
    mask = ids_sorted < num_mask

    return mask


def generate_image_mask(
    batch_size: int,
    num_timesteps: int,
    mask_prob: float,
    device: torch.device = None,
) -> torch.Tensor:
    """Generate additional image-level mask (low probability).

    Occasionally masks entire additional timesteps beyond temporal masking.
    This encourages long-range temporal learning by removing more context.

    Args:
        batch_size: Number of samples in batch.
        num_timesteps: Number of temporal positions.
        mask_prob: Probability of masking each additional timestep (e.g., 0.1).
        device: Device for tensor creation.

    Returns:
        Boolean mask of shape (B, T), True = masked timestep.
    """
    # Bernoulli sampling for each timestep
    mask = torch.rand(batch_size, num_timesteps, device=device) < mask_prob

    return mask


def generate_mask(
    batch_size: int,
    num_timesteps: int,
    num_patches: int,
    config: PretrainingConfig = None,
    device: torch.device = None,
) -> MaskOutput:
    """Generate 3-level hierarchical mask for pre-training.

    Combines three masking levels:
    1. Patch masking: Random patches within each image (75%)
    2. Temporal masking: Entire timesteps fully masked (25%)
    3. Image masking: Additional rare full timestep masking (10% chance)

    The combined mask is the union of all levels.

    Args:
        batch_size: Number of samples in batch.
        num_timesteps: Number of temporal positions (T).
        num_patches: Number of spatial patches per image (N).
        config: Pre-training configuration with mask ratios.
        device: Device for tensor creation.

    Returns:
        MaskOutput with patch_mask, temporal_mask, and combined_mask.
    """
    if config is None:
        config = PretrainingConfig()

    # Level 1: Patch masking (75% of patches per image)
    patch_mask = generate_patch_mask(
        batch_size, num_timesteps, num_patches,
        mask_ratio=config.patch_mask_ratio,
        device=device,
    )

    # Level 2: Temporal masking (25% of timesteps fully masked)
    temporal_mask = generate_temporal_mask(
        batch_size, num_timesteps,
        mask_ratio=config.temporal_mask_ratio,
        device=device,
    )

    # Level 3: Image masking (10% chance for additional full masking)
    image_mask = generate_image_mask(
        batch_size, num_timesteps,
        mask_prob=config.image_mask_prob,
        device=device,
    )

    # Combine temporal and image masks
    full_timestep_mask = temporal_mask | image_mask  # (B, T)

    # Combined mask: OR patch mask with fully-masked timesteps
    # Expand full_timestep_mask to (B, T, 1) and broadcast - no clone needed
    # since OR creates a new tensor
    combined_mask = patch_mask | full_timestep_mask.unsqueeze(-1)

    return MaskOutput(
        patch_mask=patch_mask,
        temporal_mask=full_timestep_mask,
        combined_mask=combined_mask,
    )


def get_visible_indices(mask: torch.Tensor) -> torch.Tensor:
    """Get indices of visible (unmasked) patches.

    Args:
        mask: Boolean mask of shape (B, T, N), True = masked.

    Returns:
        Tensor of visible indices, padded to max visible count.
    """
    visible = ~mask  # True = visible
    return visible


def compute_mask_stats(mask_output: MaskOutput) -> dict:
    """Compute statistics about the generated mask.

    Useful for logging and debugging mask generation.

    Args:
        mask_output: Output from generate_mask.

    Returns:
        Dictionary with mask statistics.
    """
    combined = mask_output.combined_mask.float()
    temporal = mask_output.temporal_mask.float()
    patch = mask_output.patch_mask.float()

    stats = {
        "combined_mask_ratio": combined.mean().item(),
        "temporal_mask_ratio": temporal.mean().item(),
        "patch_mask_ratio": patch.mean().item(),
        "num_fully_masked_timesteps": temporal.sum(dim=-1).mean().item(),
    }

    return stats
