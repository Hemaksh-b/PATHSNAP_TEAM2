"""Utilities to load .track, .nodes, .arcs, and .route files for a trip (Issue #1).

Returned dict keys (stable — used by other modules):
    trip_id, track (lon, lat, timestamp), nodes (lon, lat, node_id),
    arcs (from_node_id, to_node_id, arc_id), route (DataFrame[arc_id] or None)
"""
from pathlib import Path

import pandas as pd

from data_processing.config import EVALUATOR_TEST_DIR


def _read(path: Path, names: list[str]) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=names)


def is_evaluator_trip(trip_dir: Path) -> bool:
    """True if ``trip_dir`` lies inside the reserved evaluator-testing partition."""
    try:
        Path(trip_dir).resolve().relative_to(EVALUATOR_TEST_DIR.resolve())
        return True
    except ValueError:
        return False


def load_trip(trip_dir: Path, include_route: bool = True) -> dict:
    """Load one trip folder.

    ``include_route=False`` never touches the ``.route`` file (use it on every
    inference path). Reading the route of an evaluator-testing trip raises
    ``PermissionError`` -- ground truth there is for server-side scoring only.
    """
    trip_dir = Path(trip_dir)
    trip_id = trip_dir.name
    track_df = _read(trip_dir / f"{trip_id}.track", ["lon", "lat", "timestamp"])
    nodes_df = _read(trip_dir / f"{trip_id}.nodes", ["lon", "lat"])
    nodes_df["node_id"] = nodes_df.index
    arcs_df = _read(trip_dir / f"{trip_id}.arcs", ["from_node_id", "to_node_id"])
    arcs_df["arc_id"] = arcs_df.index

    route_df = None
    route_path = trip_dir / f"{trip_id}.route"
    if include_route and route_path.exists():
        if is_evaluator_trip(trip_dir):
            raise PermissionError(
                f"refusing to read evaluator ground truth for trip {trip_id}; "
                "use load_trip(..., include_route=False)"
            )
        route_df = _read(route_path, ["arc_id"])

    return {
        "trip_id": trip_id,
        "track": track_df,
        "nodes": nodes_df,
        "arcs": arcs_df,
        "route": route_df,
    }


def list_trips(partition_dir: Path) -> list[Path]:
    """Sorted trip folders inside a partition directory (empty if it does not exist)."""
    partition_dir = Path(partition_dir)
    if not partition_dir.exists():
        return []
    return sorted(p for p in partition_dir.iterdir() if p.is_dir())
