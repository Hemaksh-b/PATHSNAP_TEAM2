"""Data-quality checks for one trip (Issue #1).

Counts problems; never modifies data. EDA aggregates these per partition.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

LARGE_GAP_S = 10.0  # gaps above this are reported as "large"


@dataclass(frozen=True)
class TripReport:
    trip_id: str
    n_obs: int
    n_missing_values: int
    n_bad_coords: int
    n_duplicate_points: int  # same lon/lat/timestamp row repeated
    n_repeated_timestamps: int  # dt == 0
    n_backward_timestamps: int  # dt < 0
    n_large_gaps: int  # dt > LARGE_GAP_S
    max_gap_s: float
    share_1s_steps: float
    n_nodes: int
    n_arcs: int
    n_invalid_arc_endpoints: int
    n_self_loop_arcs: int
    route_len: int | None
    n_invalid_route_arcs: int | None
    route_connected_share: float | None  # consecutive route arcs joined head-to-tail

    def as_dict(self) -> dict:
        return asdict(self)


def _bad_coords(lon: np.ndarray, lat: np.ndarray) -> int:
    return int(np.sum(~((np.abs(lon) <= 180) & (np.abs(lat) <= 90))))


def _route_checks(trip: dict) -> tuple[int | None, int | None, float | None]:
    route = trip["route"]
    if route is None:
        return None, None, None
    arcs = trip["arcs"]
    ids = route["arc_id"].to_numpy()
    valid = (ids >= 0) & (ids < len(arcs))
    n_invalid = int(np.sum(~valid))
    if len(ids) < 2 or n_invalid:
        return len(ids), n_invalid, None
    to_node = arcs["to_node_id"].to_numpy()[ids[:-1]]
    from_node = arcs["from_node_id"].to_numpy()[ids[1:]]
    joined = (to_node == from_node) | (ids[:-1] == ids[1:])
    return len(ids), 0, float(np.mean(joined))


def validate_trip(trip: dict) -> TripReport:
    track, nodes, arcs = trip["track"], trip["nodes"], trip["arcs"]
    ts = track["timestamp"].to_numpy(dtype=float)
    dt = np.diff(ts) if len(ts) > 1 else np.array([])
    u, v = arcs["from_node_id"].to_numpy(), arcs["to_node_id"].to_numpy()
    n_nodes = len(nodes)
    route_len, n_bad_route, connected = _route_checks(trip)

    return TripReport(
        trip_id=str(trip["trip_id"]),
        n_obs=len(track),
        n_missing_values=int(track.isna().sum().sum() + nodes.isna().sum().sum()
                             + arcs.isna().sum().sum()),
        n_bad_coords=_bad_coords(track["lon"].to_numpy(), track["lat"].to_numpy())
        + _bad_coords(nodes["lon"].to_numpy(), nodes["lat"].to_numpy()),
        n_duplicate_points=int(track.duplicated().sum()),
        n_repeated_timestamps=int(np.sum(dt == 0)),
        n_backward_timestamps=int(np.sum(dt < 0)),
        n_large_gaps=int(np.sum(dt > LARGE_GAP_S)),
        max_gap_s=float(dt.max()) if dt.size else 0.0,
        share_1s_steps=float(np.mean(dt == 1)) if dt.size else 0.0,
        n_nodes=n_nodes,
        n_arcs=len(arcs),
        n_invalid_arc_endpoints=int(np.sum((u < 0) | (u >= n_nodes) | (v < 0) | (v >= n_nodes))),
        n_self_loop_arcs=int(np.sum(u == v)),
        route_len=route_len,
        n_invalid_route_arcs=n_bad_route,
        route_connected_share=connected,
    )
