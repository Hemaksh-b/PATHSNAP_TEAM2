"""Utilities to load .track, .nodes, .arcs, and .route files for a trip."""
from pathlib import Path
import pandas as pd


def load_trip(trip_dir: Path) -> dict:
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

    route_path = trip_dir / f"{trip_id}.route"
    route_df = None
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
