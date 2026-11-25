"""Tests for LocationEncoder."""

import math

import pytest
import torch

from model.encoders.location import LocationEncoder, create_location_encoder
from model.config import SeasonalViTConfig


class TestLocationEncoder:
    """Test suite for LocationEncoder."""

    def test_output_shape(self, model_config: SeasonalViTConfig) -> None:
        """Test that output has correct shape."""
        encoder = LocationEncoder(model_config)
        batch_size = 4

        lat = torch.rand(batch_size) * 180 - 90  # [-90, 90]
        lon = torch.rand(batch_size) * 360 - 180  # [-180, 180]

        location_enc, hemisphere = encoder(lat, lon)

        assert location_enc.shape == (batch_size, model_config.embed_dim)
        assert hemisphere.shape == (batch_size,)

    def test_output_shape_single(self, model_config: SeasonalViTConfig) -> None:
        """Test with single location (scalar inputs)."""
        encoder = LocationEncoder(model_config)

        lat = torch.tensor(45.0)
        lon = torch.tensor(-122.0)

        location_enc, hemisphere = encoder(lat, lon)

        assert location_enc.shape == (1, model_config.embed_dim)
        assert hemisphere.shape == (1,)

    def test_gradient_flow(self, model_config: SeasonalViTConfig) -> None:
        """Test that gradients flow through the encoder."""
        encoder = LocationEncoder(model_config)

        lat = torch.tensor([45.0, -30.0], requires_grad=False)
        lon = torch.tensor([-122.0, 150.0], requires_grad=False)

        location_enc, _ = encoder(lat, lon)
        loss = location_enc.sum()
        loss.backward()

        # Check gradients exist for learnable parameters
        assert encoder.hemisphere_embed.weight.grad is not None
        assert not torch.isnan(encoder.hemisphere_embed.weight.grad).any()

    def test_hemisphere_detection_northern(
        self, model_config: SeasonalViTConfig
    ) -> None:
        """Test that northern hemisphere is correctly detected."""
        encoder = LocationEncoder(model_config)

        # Northern hemisphere locations
        lat = torch.tensor([45.0, 60.0, 10.0, 0.0])  # 0 is Northern by convention
        lon = torch.tensor([0.0, 0.0, 0.0, 0.0])

        _, hemisphere = encoder(lat, lon)

        # All should be Northern (1)
        assert (hemisphere == 1).all()

    def test_hemisphere_detection_southern(
        self, model_config: SeasonalViTConfig
    ) -> None:
        """Test that southern hemisphere is correctly detected."""
        encoder = LocationEncoder(model_config)

        # Southern hemisphere locations
        lat = torch.tensor([-45.0, -60.0, -10.0, -0.1])
        lon = torch.tensor([0.0, 0.0, 0.0, 0.0])

        _, hemisphere = encoder(lat, lon)

        # All should be Southern (0)
        assert (hemisphere == 0).all()

    def test_different_hemispheres_produce_different_encodings(
        self, model_config: SeasonalViTConfig
    ) -> None:
        """Test that different hemispheres produce different encodings."""
        encoder = LocationEncoder(model_config)
        encoder.eval()

        with torch.no_grad():
            # Same longitude, different hemispheres
            north_enc, _ = encoder(torch.tensor([45.0]), torch.tensor([0.0]))
            south_enc, _ = encoder(torch.tensor([-45.0]), torch.tensor([0.0]))

        assert not torch.allclose(north_enc, south_enc)

    def test_different_locations_produce_different_encodings(
        self, model_config: SeasonalViTConfig
    ) -> None:
        """Test that different locations produce different encodings."""
        encoder = LocationEncoder(model_config)
        encoder.eval()

        with torch.no_grad():
            loc1, _ = encoder(torch.tensor([45.0]), torch.tensor([-122.0]))
            loc2, _ = encoder(torch.tensor([51.5]), torch.tensor([-0.1]))  # London
            loc3, _ = encoder(torch.tensor([-33.9]), torch.tensor([151.2]))  # Sydney

        assert not torch.allclose(loc1, loc2)
        assert not torch.allclose(loc2, loc3)
        assert not torch.allclose(loc1, loc3)

    def test_deterministic_output(self, model_config: SeasonalViTConfig) -> None:
        """Test that same input produces same output in eval mode."""
        encoder = LocationEncoder(model_config)
        encoder.eval()

        lat = torch.tensor([45.0, -30.0])
        lon = torch.tensor([-122.0, 150.0])

        with torch.no_grad():
            output1, hemi1 = encoder(lat, lon)
            output2, hemi2 = encoder(lat, lon)

        assert torch.allclose(output1, output2)
        assert torch.equal(hemi1, hemi2)

    def test_factory_function(self) -> None:
        """Test create_location_encoder factory function."""
        config = SeasonalViTConfig(swin_pretrained=False)
        encoder = create_location_encoder(config)

        assert isinstance(encoder, LocationEncoder)
        assert encoder.get_output_dim() == config.embed_dim

    def test_factory_function_default_config(self) -> None:
        """Test factory function with default config."""
        encoder = create_location_encoder()

        assert isinstance(encoder, LocationEncoder)
        assert encoder.get_output_dim() == 768  # Default embed_dim

    def test_get_output_dim(self, model_config: SeasonalViTConfig) -> None:
        """Test get_output_dim method."""
        encoder = LocationEncoder(model_config)

        assert encoder.get_output_dim() == model_config.embed_dim


class TestLocationEncoderEdgeCases:
    """Edge case tests for location encoder."""

    def test_poles(self, model_config: SeasonalViTConfig) -> None:
        """Test encoding at poles."""
        encoder = LocationEncoder(model_config)

        # North pole
        north_enc, north_hemi = encoder(torch.tensor([90.0]), torch.tensor([0.0]))
        assert not torch.isnan(north_enc).any()
        assert north_hemi.item() == 1

        # South pole
        south_enc, south_hemi = encoder(torch.tensor([-90.0]), torch.tensor([0.0]))
        assert not torch.isnan(south_enc).any()
        assert south_hemi.item() == 0

    def test_equator(self, model_config: SeasonalViTConfig) -> None:
        """Test encoding at equator."""
        encoder = LocationEncoder(model_config)

        # Equator (lat=0 is considered Northern by convention)
        enc, hemi = encoder(torch.tensor([0.0]), torch.tensor([0.0]))
        assert not torch.isnan(enc).any()
        assert hemi.item() == 1  # lat >= 0 is Northern

    def test_dateline(self, model_config: SeasonalViTConfig) -> None:
        """Test encoding at international date line."""
        encoder = LocationEncoder(model_config)
        encoder.eval()

        with torch.no_grad():
            # Points just west and east of dateline should be close
            west, _ = encoder(torch.tensor([0.0]), torch.tensor([179.9]))
            east, _ = encoder(torch.tensor([0.0]), torch.tensor([-179.9]))

        # These should be very close (nearly same point on Earth)
        assert torch.allclose(west, east, atol=0.5)

    def test_prime_meridian(self, model_config: SeasonalViTConfig) -> None:
        """Test encoding at prime meridian."""
        encoder = LocationEncoder(model_config)
        encoder.eval()

        with torch.no_grad():
            # Points just west and east of prime meridian should be close
            west, _ = encoder(torch.tensor([0.0]), torch.tensor([-0.1]))
            east, _ = encoder(torch.tensor([0.0]), torch.tensor([0.1]))

        # These should be very close
        assert torch.allclose(west, east, atol=0.5)

    def test_cartesian_conversion_north_pole(
        self, model_config: SeasonalViTConfig
    ) -> None:
        """Test that north pole converts to (0, 0, 1) in Cartesian."""
        encoder = LocationEncoder(model_config)

        lat = torch.tensor([90.0])
        lon = torch.tensor([0.0])
        coords = encoder._lat_lon_to_cartesian(lat, lon)

        expected = torch.tensor([[0.0, 0.0, 1.0]])
        assert torch.allclose(coords, expected, atol=1e-6)

    def test_cartesian_conversion_south_pole(
        self, model_config: SeasonalViTConfig
    ) -> None:
        """Test that south pole converts to (0, 0, -1) in Cartesian."""
        encoder = LocationEncoder(model_config)

        lat = torch.tensor([-90.0])
        lon = torch.tensor([0.0])
        coords = encoder._lat_lon_to_cartesian(lat, lon)

        expected = torch.tensor([[0.0, 0.0, -1.0]])
        assert torch.allclose(coords, expected, atol=1e-6)

    def test_cartesian_conversion_equator_prime_meridian(
        self, model_config: SeasonalViTConfig
    ) -> None:
        """Test that (0, 0) converts to (1, 0, 0) in Cartesian."""
        encoder = LocationEncoder(model_config)

        lat = torch.tensor([0.0])
        lon = torch.tensor([0.0])
        coords = encoder._lat_lon_to_cartesian(lat, lon)

        expected = torch.tensor([[1.0, 0.0, 0.0]])
        assert torch.allclose(coords, expected, atol=1e-6)

    def test_cartesian_conversion_unit_sphere(
        self, model_config: SeasonalViTConfig
    ) -> None:
        """Test that all Cartesian coordinates lie on unit sphere."""
        encoder = LocationEncoder(model_config)

        # Random locations
        lat = torch.rand(100) * 180 - 90
        lon = torch.rand(100) * 360 - 180
        coords = encoder._lat_lon_to_cartesian(lat, lon)

        # Check magnitude is 1 for all points
        magnitudes = torch.norm(coords, dim=1)
        expected = torch.ones(100)
        assert torch.allclose(magnitudes, expected, atol=1e-6)

    def test_large_batch(self, model_config: SeasonalViTConfig) -> None:
        """Test with large batch size."""
        encoder = LocationEncoder(model_config)
        batch_size = 256

        lat = torch.rand(batch_size) * 180 - 90
        lon = torch.rand(batch_size) * 360 - 180

        location_enc, hemisphere = encoder(lat, lon)

        assert location_enc.shape == (batch_size, model_config.embed_dim)
        assert hemisphere.shape == (batch_size,)
        assert not torch.isnan(location_enc).any()
