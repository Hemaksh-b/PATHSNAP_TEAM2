"""Batch feature extraction and alignment pipeline for PathSnap partitions (Issue #3).

Builds candidate feature tables for training and student-testing partitions,
saves them under feature_engineering/results/, and generates visual alignment maps.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import folium
import numpy as np
import pandas as pd

from data_processing.config import RESULTS_DIR, STUDENT_TEST_DIR, TRAIN_DIR, VIS_DIR
from data_processing.loader import list_trips, load_trip
from data_processing.projection import project_trip
from feature_engineering.alignment import align_observations_to_route
from feature_engineering.features import extract_candidate_features

FEATURE_RESULTS_DIR = Path(__file__).resolve().parent / "results"


def process_partition(
    partition: str = "train",
    k: int = 5,
    radius_m: float = 100.0,
    out_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Extract candidate features and aligned labels across all trips in a partition.

    Saves results to <out_dir>/<partition>_features.csv.
    """
    if out_dir is None:
        out_dir = FEATURE_RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    folder = TRAIN_DIR if partition == "train" else STUDENT_TEST_DIR
    trip_paths = list_trips(folder)
    if not trip_paths:
        raise FileNotFoundError(f"No trip directories found in {folder}")

    frames: list[pd.DataFrame] = []
    total_obs = 0
    total_uncertain = 0

    print(f"[{partition}] Processing {len(trip_paths)} trips (k={k}, radius={radius_m}m)...")
    for idx, trip_path in enumerate(trip_paths):
        trip = load_trip(trip_path)
        ptrip = project_trip(trip)

        df = extract_candidate_features(
            ptrip,
            k=k,
            radius_m=radius_m,
            include_labels=(partition in {"train", "student_test"}),
        )
        frames.append(df)

        if "label_uncertain" in df.columns:
            obs_grouped = df.groupby("obs_idx")["label_uncertain"].first()
            total_obs += len(obs_grouped)
            total_uncertain += int(obs_grouped.sum())

        if (idx + 1) % 20 == 0 or (idx + 1) == len(trip_paths):
            print(f"  Processed {idx + 1}/{len(trip_paths)} trips...")

    combined = pd.concat(frames, ignore_index=True)
    out_csv = out_dir / f"{partition}_features.csv"
    combined.to_csv(out_csv, index=False)

    if total_obs > 0:
        unc_pct = (total_uncertain / total_obs) * 100.0
        print(f"[{partition}] Saved {len(combined)} candidate rows to {out_csv}")
        print(f"[{partition}] Observation alignment: {total_obs} obs, {total_uncertain} uncertain ({unc_pct:.2f}%)")
    else:
        print(f"[{partition}] Saved {len(combined)} candidate rows to {out_csv}")

    return combined


def generate_alignment_map(
    trip_dir: Path,
    out_html: Path,
    margin_deg: float = 0.003,
) -> None:
    """Generate interactive Folium map illustrating GPS observations, road network, and alignment uncertainty."""
    trip = load_trip(trip_dir)
    ptrip = project_trip(trip)
    alignment = align_observations_to_route(ptrip)

    track, nodes, arcs = trip["track"], trip["nodes"], trip["arcs"]
    min_lat, max_lat = track["lat"].min() - margin_deg, track["lat"].max() + margin_deg
    min_lon, max_lon = track["lon"].min() - margin_deg, track["lon"].max() + margin_deg

    inside = (nodes["lat"].between(min_lat, max_lat) & nodes["lon"].between(min_lon, max_lon)).to_numpy()
    u, v = arcs["from_node_id"].to_numpy(), arcs["to_node_id"].to_numpy()
    keep = inside[u] | inside[v]

    lat, lon = nodes["lat"].to_numpy(), nodes["lon"].to_numpy()
    segments = [[(lat[a], lon[a]), (lat[b], lon[b])] for a, b in zip(u[keep], v[keep])]

    fmap = folium.Map(control_scale=True)

    # Road network layer
    roads = folium.FeatureGroup(name="Road network", show=True)
    folium.PolyLine(segments, color="#808080", weight=2, opacity=0.5).add_to(roads)
    roads.add_to(fmap)

    # Ground truth route layer
    if ptrip.get("route") is not None:
        route_arc_ids = ptrip["route"]["arc_id"].to_numpy(dtype=int)
        r_u = arcs["from_node_id"].to_numpy()[route_arc_ids]
        r_v = arcs["to_node_id"].to_numpy()[route_arc_ids]
        route_segs = [[(lat[a], lon[a]), (lat[b], lon[b])] for a, b in zip(r_u, r_v)]
        route_layer = folium.FeatureGroup(name="Ground truth route (.route)", show=True)
        folium.PolyLine(route_segs, color="#00aa00", weight=4, opacity=0.8).add_to(route_layer)
        route_layer.add_to(fmap)

    # Observations layer
    gps_layer = folium.FeatureGroup(name="GPS observations", show=True)
    is_unc = alignment["is_uncertain"].to_numpy() if alignment is not None else np.zeros(len(track), dtype=bool)

    for i, (la, lo) in enumerate(zip(track["lat"], track["lon"])):
        unc = is_unc[i] if i < len(is_unc) else False
        color = "#e65100" if unc else "#1565c0"
        radius = 4 if unc else 2.5
        folium.CircleMarker(
            (la, lo),
            radius=radius,
            color=color,
            fill=True,
            fill_opacity=0.9,
            popup=f"Obs {i}: {'UNCERTAIN' if unc else 'Aligned'}",
        ).add_to(gps_layer)
    gps_layer.add_to(fmap)

    fmap.fit_bounds([[track["lat"].min(), track["lon"].min()], [track["lat"].max(), track["lon"].max()]])
    folium.LayerControl(collapsed=False).add_to(fmap)

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out_html))
    print(f"Alignment visualization saved to {out_html}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract candidate features and build feature matrices (Issue #3)")
    parser.add_argument("--partition", choices=["train", "student_test", "all"], default="all")
    parser.add_argument("--map-trip", default=None, help="Generate alignment map for specific trip ID")
    args = parser.parse_args()

    partitions = ["train", "student_test"] if args.partition == "all" else [args.partition]
    for part in partitions:
        process_partition(part)

    # Generate Folium alignment map
    map_trip_id = args.map_trip or "00000001"
    map_path = VIS_DIR / "maps" / f"alignment_{map_trip_id}.html"
    generate_alignment_map(TRAIN_DIR / map_trip_id, map_path)


if __name__ == "__main__":
    main()
