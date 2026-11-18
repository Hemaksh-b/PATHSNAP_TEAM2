"""Per-trip metric projection (Issue #1).

Trips are spread over several countries/continents, so one fixed projected
CRS would distort distances badly. Each trip is projected to the WGS 84 UTM
zone that contains its mean position (distortion < 0.1 % inside a zone).
"""
from __future__ import annotations

import numpy as np
from pyproj import CRS, Transformer


def utm_epsg_for(lon: float, lat: float) -> int:
    """EPSG code of the WGS 84 UTM zone containing (lon, lat)."""
    if not (-180.0 <= lon <= 180.0 and -80.0 <= lat <= 84.0):
        raise ValueError(f"({lon}, {lat}) is outside the UTM domain")
    zone = min(int((lon + 180.0) // 6.0) + 1, 60)
    return (32600 if lat >= 0 else 32700) + zone


def trip_transformer(trip: dict) -> Transformer:
    """lon/lat -> UTM metres transformer for a trip (zone from the GPS mean)."""
    track = trip["track"]
    epsg = utm_epsg_for(float(track["lon"].mean()), float(track["lat"].mean()))
    return Transformer.from_crs("EPSG:4326", CRS.from_epsg(epsg), always_xy=True)


def project_trip(trip: dict) -> dict:
    """Return a NEW trip dict with metric columns added (input is not modified).

    Adds ``x``, ``y`` (metres) to ``track`` and ``nodes``, ``length_m`` to
    ``arcs`` and the EPSG code under ``crs_epsg``.
    """
    tr = trip_transformer(trip)
    track = trip["track"].copy()
    nodes = trip["nodes"].copy()
    arcs = trip["arcs"].copy()

    track["x"], track["y"] = tr.transform(track["lon"].to_numpy(), track["lat"].to_numpy())
    nodes["x"], nodes["y"] = tr.transform(nodes["lon"].to_numpy(), nodes["lat"].to_numpy())
    nx_, ny_ = nodes["x"].to_numpy(), nodes["y"].to_numpy()
    u, v = arcs["from_node_id"].to_numpy(), arcs["to_node_id"].to_numpy()
    arcs["length_m"] = np.hypot(nx_[v] - nx_[u], ny_[v] - ny_[u])

    epsg = tr.target_crs.to_epsg()
    return {**trip, "track": track, "nodes": nodes, "arcs": arcs, "crs_epsg": epsg}
