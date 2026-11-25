"""Data augmentation transforms for NDVI time series.

Provides both temporal and spatial augmentations suitable for
satellite imagery time series.
"""

from typing import Callable, Dict, List, Optional

import torch


class Compose:
    """Compose multiple transforms together.

    Attributes:
        transforms: List of transforms to apply sequentially.
    """

    def __init__(self, transforms: List[Callable]) -> None:
        """Initialize with list of transforms.

        Args:
            transforms: List of callable transforms.
        """
        self.transforms = transforms

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply all transforms sequentially.

        Args:
            sample: Sample dictionary.

        Returns:
            Transformed sample.
        """
        for transform in self.transforms:
            sample = transform(sample)
        return sample


class TemporalDropout:
    """Randomly drop temporal bands by setting them to zero.

    Improves robustness to missing data and temporal gaps.

    Attributes:
        drop_prob: Probability of dropping each temporal band.
    """

    def __init__(self, drop_prob: float = 0.1) -> None:
        """Initialize with drop probability.

        Args:
            drop_prob: Probability of dropping each temporal band.
        """
        self.drop_prob = drop_prob

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply temporal dropout.

        Args:
            sample: Sample dictionary with 'images' key.

        Returns:
            Sample with some temporal bands zeroed out.
        """
        images = sample["images"]  # (T, H, W)
        T = images.shape[0]

        # Create mask for which timesteps to keep
        keep_mask = torch.rand(T) > self.drop_prob
        keep_mask = keep_mask.view(-1, 1, 1).expand_as(images)

        sample["images"] = torch.where(keep_mask, images, torch.zeros_like(images))
        return sample


class TemporalNoise:
    """Add Gaussian noise to simulate sensor uncertainty.

    Typical NDVI sensor uncertainty is ±0.02-0.05.

    Attributes:
        noise_std: Standard deviation of Gaussian noise.
    """

    def __init__(self, noise_std: float = 0.03) -> None:
        """Initialize with noise standard deviation.

        Args:
            noise_std: Standard deviation of Gaussian noise.
        """
        self.noise_std = noise_std

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply temporal noise.

        Args:
            sample: Sample dictionary with 'images' key.

        Returns:
            Sample with added noise, clamped to [-1, 1].
        """
        images = sample["images"]
        noise = torch.randn_like(images) * self.noise_std
        sample["images"] = torch.clamp(images + noise, -1, 1)
        return sample


class SeasonalJitter:
    """Swap adjacent seasons within years.

    Simulates phenological variability where seasons may shift
    slightly between years.

    Attributes:
        swap_prob: Probability of swapping adjacent seasons in each year.
    """

    def __init__(self, swap_prob: float = 0.5) -> None:
        """Initialize with swap probability.

        Args:
            swap_prob: Probability of swapping in each year.
        """
        self.swap_prob = swap_prob

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply seasonal jitter.

        Args:
            sample: Sample dictionary with 'images' key.

        Returns:
            Sample with some adjacent seasons swapped.
        """
        images = sample["images"].clone()  # (T, H, W)
        T = images.shape[0]

        # Process each year (4 seasons)
        for year_start in range(0, T, 4):
            if torch.rand(1).item() < self.swap_prob:
                # Pick random adjacent pair within the year
                idx1 = year_start + torch.randint(0, 3, (1,)).item()
                idx2 = idx1 + 1

                if idx2 < T:
                    # Swap the two timesteps
                    images[[idx1, idx2]] = images[[idx2, idx1]]

        sample["images"] = images
        return sample


class RandomHorizontalFlip:
    """Randomly flip images horizontally.

    Attributes:
        p: Probability of flipping.
    """

    def __init__(self, p: float = 0.5) -> None:
        """Initialize with flip probability.

        Args:
            p: Probability of flipping.
        """
        self.p = p

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply horizontal flip.

        Args:
            sample: Sample dictionary with 'images' key.

        Returns:
            Sample with images optionally flipped.
        """
        if torch.rand(1).item() < self.p:
            sample["images"] = torch.flip(sample["images"], dims=[2])
            if "nodata_mask" in sample:
                sample["nodata_mask"] = torch.flip(sample["nodata_mask"], dims=[2])
        return sample


class RandomVerticalFlip:
    """Randomly flip images vertically.

    Attributes:
        p: Probability of flipping.
    """

    def __init__(self, p: float = 0.5) -> None:
        """Initialize with flip probability.

        Args:
            p: Probability of flipping.
        """
        self.p = p

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply vertical flip.

        Args:
            sample: Sample dictionary with 'images' key.

        Returns:
            Sample with images optionally flipped.
        """
        if torch.rand(1).item() < self.p:
            sample["images"] = torch.flip(sample["images"], dims=[1])
            if "nodata_mask" in sample:
                sample["nodata_mask"] = torch.flip(sample["nodata_mask"], dims=[1])
        return sample


class RandomRotation90:
    """Randomly rotate images by 90-degree increments.

    Preserves pixel alignment unlike arbitrary rotations.
    """

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply random 90-degree rotation.

        Args:
            sample: Sample dictionary with 'images' key.

        Returns:
            Sample with images rotated 0, 90, 180, or 270 degrees.
        """
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            sample["images"] = torch.rot90(sample["images"], k, dims=[1, 2])
            if "nodata_mask" in sample:
                sample["nodata_mask"] = torch.rot90(
                    sample["nodata_mask"], k, dims=[1, 2]
                )
        return sample


class Normalize:
    """Normalize NDVI values.

    Supports multiple normalization strategies.

    Attributes:
        method: Normalization method ('minmax', 'standardize', 'robust').
    """

    def __init__(self, method: str = "minmax") -> None:
        """Initialize with normalization method.

        Args:
            method: One of 'minmax', 'standardize', 'robust'.
        """
        self.method = method

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply normalization.

        Args:
            sample: Sample dictionary with 'images' key.

        Returns:
            Sample with normalized images.
        """
        images = sample["images"]

        if self.method == "minmax":
            # Scale from [-1, 1] to [0, 1]
            sample["images"] = (images + 1) / 2
        elif self.method == "standardize":
            # Z-score normalization
            mean = images.mean()
            std = images.std() + 1e-8
            sample["images"] = (images - mean) / std
        elif self.method == "robust":
            # Robust scaling using percentiles
            low = torch.quantile(images, 0.02)
            high = torch.quantile(images, 0.98)
            images = torch.clamp(images, low, high)
            sample["images"] = (images - low) / (high - low + 1e-8)

        return sample


def get_train_transforms(
    temporal_dropout_prob: float = 0.1,
    temporal_noise_std: float = 0.03,
    seasonal_jitter_prob: float = 0.3,
    flip_prob: float = 0.5,
    rotation: bool = True,
    normalize: Optional[str] = None,
) -> Compose:
    """Get standard training transforms.

    Args:
        temporal_dropout_prob: Probability of temporal dropout.
        temporal_noise_std: Standard deviation of temporal noise.
        seasonal_jitter_prob: Probability of seasonal jitter.
        flip_prob: Probability of horizontal/vertical flip.
        rotation: Whether to apply random 90-degree rotation.
        normalize: Normalization method (None, 'minmax', 'standardize', 'robust').

    Returns:
        Composed transform pipeline.
    """
    transforms = []

    if temporal_dropout_prob > 0:
        transforms.append(TemporalDropout(temporal_dropout_prob))

    if temporal_noise_std > 0:
        transforms.append(TemporalNoise(temporal_noise_std))

    if seasonal_jitter_prob > 0:
        transforms.append(SeasonalJitter(seasonal_jitter_prob))

    if flip_prob > 0:
        transforms.append(RandomHorizontalFlip(flip_prob))
        transforms.append(RandomVerticalFlip(flip_prob))

    if rotation:
        transforms.append(RandomRotation90())

    if normalize:
        transforms.append(Normalize(normalize))

    return Compose(transforms)


def get_val_transforms(normalize: Optional[str] = None) -> Compose:
    """Get standard validation transforms.

    No augmentation, only optional normalization.

    Args:
        normalize: Normalization method (None, 'minmax', 'standardize', 'robust').

    Returns:
        Composed transform pipeline.
    """
    transforms = []

    if normalize:
        transforms.append(Normalize(normalize))

    return Compose(transforms)
