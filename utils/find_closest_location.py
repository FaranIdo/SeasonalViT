#!/usr/bin/env python3
"""Find the closest location in locations.json to a target coordinate."""

import json
import math

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points in kilometers."""
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))

    # Radius of Earth in kilometers
    r = 6371
    return c * r

# Target coordinates (31°59'55"N 35°13'45"E)
target_lat = 31 + 59/60 + 55/3600
target_lon = 35 + 13/60 + 45/3600

print(f"Target coordinates: {target_lat:.6f}°N, {target_lon:.6f}°E")
print()

# Load locations
with open('data/locations.json', 'r') as f:
    locations = json.load(f)

# Find closest location
min_distance = float('inf')
closest_id = None
closest_coords = None

for sample_id, coords in locations.items():
    lat, lon = coords
    distance = haversine_distance(target_lat, target_lon, lat, lon)

    if distance < min_distance:
        min_distance = distance
        closest_id = sample_id
        closest_coords = coords

print(f"Closest location:")
print(f"  Sample ID: {closest_id}")
print(f"  Coordinates: {closest_coords[0]:.6f}°N, {closest_coords[1]:.6f}°E")
print(f"  Distance: {min_distance:.2f} km")
