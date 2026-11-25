"""Encoder components for SeasonalViT.

Components:
- HierarchicalSpatialEncoder: Swin-T based spatial feature extraction
- SeasonalPositionalEncoding: Temporal position encodings
- LocationEncoder: Geographic location encodings

Imports are lazy to avoid errors during incremental development.
"""

__all__ = [
    "HierarchicalSpatialEncoder",
    "create_spatial_encoder",
    "SeasonalPositionalEncoding",
    "create_temporal_encoding",
    "LocationEncoder",
    "create_location_encoder",
]


def __getattr__(name: str):
    """Lazy import to allow incremental development."""
    if name == "HierarchicalSpatialEncoder":
        from model.encoders.spatial import HierarchicalSpatialEncoder
        return HierarchicalSpatialEncoder
    elif name == "create_spatial_encoder":
        from model.encoders.spatial import create_spatial_encoder
        return create_spatial_encoder
    elif name == "SeasonalPositionalEncoding":
        from model.encoders.temporal import SeasonalPositionalEncoding
        return SeasonalPositionalEncoding
    elif name == "create_temporal_encoding":
        from model.encoders.temporal import create_temporal_encoding
        return create_temporal_encoding
    elif name == "LocationEncoder":
        from model.encoders.location import LocationEncoder
        return LocationEncoder
    elif name == "create_location_encoder":
        from model.encoders.location import create_location_encoder
        return create_location_encoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
