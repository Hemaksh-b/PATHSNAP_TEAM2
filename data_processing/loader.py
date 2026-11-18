"""Utilities to load .track, .nodes, .arcs, and .route files for a trip."""
from pathlib import Path
from typing import List

import pandas as pd


def load_trip(trip_dir: Path, include_route: bool = True) -> dict:
    """Load a trip folder into a dict of DataFrames.

    Parameters
    ----------
    trip_dir:
        Path to the folder whose name is the trip_id.
    include_route:
        When *False* the reference route is never read from disk.
        Pass ``False`` in the inference pipeline to prevent any
        accidental label leakage into prediction.
    """
    trip_dir = Path(trip_dir)
    trip_id = trip_dir.name
    track_df = pd.read_csv(
        trip_dir / f"{trip_id}.track",
        sep=r"\s+",
        header=None,
        names=["lon", "lat", "timestamp"],
    )
    nodes_df = pd.read_csv(
        trip_dir / f"{trip_id}.nodes",
        sep=r"\s+",
        header=None,
        names=["lon", "lat"],
    )
    nodes_df["node_id"] = nodes_df.index

    arcs_df = pd.read_csv(
        trip_dir / f"{trip_id}.arcs",
        sep=r"\s+",
        header=None,
        names=["from_node_id", "to_node_id"],
    )
    arcs_df["arc_id"] = arcs_df.index

    route_df = None
    if include_route:
        route_path = trip_dir / f"{trip_id}.route"
        if route_path.exists():
            route_df = pd.read_csv(
                route_path,
                sep=r"\s+",
                header=None,
                names=["arc_id"],
            )

    return {
        "trip_id": trip_id,
        "track": track_df,
        "nodes": nodes_df,
        "arcs": arcs_df,
        "route": route_df,
    }


def list_trips(trips_dir: Path) -> List[Path]:
    """Return sorted list of trip sub-folder Paths under *trips_dir*.

    Each sub-folder whose name does not start with '.' is treated as a trip.
    Returns an empty list when *trips_dir* does not exist.
    """
    trips_dir = Path(trips_dir)
    if not trips_dir.is_dir():
        return []
    return sorted(p for p in trips_dir.iterdir() if p.is_dir() and not p.name.startswith("."))
