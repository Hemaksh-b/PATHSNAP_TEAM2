"""Per-trip metric projection (Issue #1).

Trips are spread over several countries/continents, so one fixed projected
CRS would distort distances badly. Each trip is projected to the WGS 84 UTM
zone that contains its mean position (distortion < 0.1 % inside a zone).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from pyproj import CRS, Transformer

    _HAS_PYPROJ = True
except ImportError:  # pragma: no cover
    CRS = None  # type: ignore
    Transformer = None  # type: ignore
    _HAS_PYPROJ = False


@dataclass(frozen=True)
class _TargetCRS:
    epsg: int

    def to_epsg(self) -> int:
        return self.epsg


class _FallbackUTMTransformer:
    """Metric local tangent-plane / UTM approximation when pyproj is unavailable."""

    def __init__(self, epsg: int, ref_lat: float = 0.0) -> None:
        self.target_crs = _TargetCRS(epsg)
        self._cos_lat = float(np.cos(np.radians(ref_lat)))

    def transform(self, lon: Any, lat: Any) -> tuple[np.ndarray, np.ndarray]:
        lon_arr = np.asarray(lon, dtype=float)
        lat_arr = np.asarray(lat, dtype=float)
        x = lon_arr * 111_320.0 * self._cos_lat
        y = lat_arr * 111_132.0
        return x, y


def utm_epsg_for(lon: float, lat: float) -> int:
    """EPSG code of the WGS 84 UTM zone containing (lon, lat)."""
    if not (-180.0 <= lon <= 180.0 and -80.0 <= lat <= 84.0):
        raise ValueError(f"({lon}, {lat}) is outside the UTM domain")
    zone = min(int((lon + 180.0) // 6.0) + 1, 60)
    return (32600 if lat >= 0 else 32700) + zone


def trip_transformer(trip: dict) -> Any:
    """lon/lat -> UTM metres transformer for a trip (zone from the GPS mean)."""
    track = trip["track"]
    mean_lon = float(track["lon"].mean())
    mean_lat = float(track["lat"].mean())
    epsg = utm_epsg_for(mean_lon, mean_lat)
    if _HAS_PYPROJ:
        return Transformer.from_crs("EPSG:4326", CRS.from_epsg(epsg), always_xy=True)
    return _FallbackUTMTransformer(epsg=epsg, ref_lat=mean_lat)


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
