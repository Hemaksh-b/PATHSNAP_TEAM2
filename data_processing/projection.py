"""CRS projection utilities — project lon/lat GPS points to UTM metres.

Stub implementation used until Member 1 delivers the full version.
The API pipeline calls project_trip() which adds 'x', 'y' columns (metres)
to the track, nodes, and arcs DataFrames.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from pyproj import Transformer
    _HAS_PYPROJ = True
except ImportError:  # pragma: no cover
    _HAS_PYPROJ = False


def _lon_lat_to_utm(lon: np.ndarray, lat: np.ndarray):
    """Convert WGS-84 lon/lat arrays to approximate UTM metres."""
    if _HAS_PYPROJ:
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)
        x, y = transformer.transform(lon, lat)
        return np.asarray(x), np.asarray(y)
    # Flat-earth approximation fallback
    lat0 = float(np.mean(lat))
    cos_lat = np.cos(np.radians(lat0))
    x = np.asarray(lon, dtype=float) * 111_320 * cos_lat
    y = np.asarray(lat, dtype=float) * 111_320
    return x, y


def project_trip(trip: dict) -> dict:
    """Add 'x', 'y' columns (metres) to track/nodes DataFrames.

    Returns a *new* dict; the input is not modified.
    """
    trip = dict(trip)  # shallow copy

    # Project track
    track = trip["track"].copy()
    x, y = _lon_lat_to_utm(track["lon"].to_numpy(), track["lat"].to_numpy())
    track["x"] = x
    track["y"] = y
    trip["track"] = track

    # Project nodes
    nodes = trip["nodes"].copy()
    xn, yn = _lon_lat_to_utm(nodes["lon"].to_numpy(), nodes["lat"].to_numpy())
    nodes["x"] = xn
    nodes["y"] = yn
    trip["nodes"] = nodes

    return trip
