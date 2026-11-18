"""Exploratory data analysis for the permitted partitions (Issue #1).

Run from the repo root:
    python -m data_processing.eda

Writes (all generated from the real data, nothing hand-typed):
    supporting_results/eda/trip_summary.csv      one row per trip
    supporting_results/eda/partition_summary.csv aggregated per partition
    visualisations/plots/eda_*.png               distribution plots
    visualisations/maps/eda_<trip>.html          Folium map of one training trip

Evaluator-testing data is never read.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import folium
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data_processing.config import RESULTS_DIR, STUDENT_TEST_DIR, TRAIN_DIR, VIS_DIR  # noqa: E402
from data_processing.loader import list_trips, load_trip  # noqa: E402
from data_processing.projection import project_trip  # noqa: E402
from data_processing.validation import validate_trip  # noqa: E402

PARTITIONS = {"train": TRAIN_DIR, "student_test": STUDENT_TEST_DIR}


def network_stats(trip: dict) -> dict:
    arcs = trip["arcs"]
    graph = nx.DiGraph()
    graph.add_nodes_from(trip["nodes"]["node_id"])
    graph.add_edges_from(zip(arcs["from_node_id"], arcs["to_node_id"]))
    pairs = set(zip(arcs["from_node_id"], arcs["to_node_id"]))
    two_way = sum((v, u) in pairs for u, v in pairs) / max(len(pairs), 1)
    return {
        "weakly_connected_components": nx.number_weakly_connected_components(graph),
        "strongly_connected_components": nx.number_strongly_connected_components(graph),
        "isolated_nodes": nx.number_of_isolates(graph),
        "two_way_arc_share": two_way,
    }


def track_stats(ptrip: dict) -> dict:
    track = ptrip["track"]
    step = np.hypot(np.diff(track["x"]), np.diff(track["y"]))
    duration = float(track["timestamp"].iloc[-1] - track["timestamp"].iloc[0])
    length_km = float(step.sum() / 1000)
    return {
        "crs_epsg": ptrip["crs_epsg"],
        "mean_lon": float(track["lon"].mean()),
        "mean_lat": float(track["lat"].mean()),
        "extent_x_km": float((track["x"].max() - track["x"].min()) / 1000),
        "extent_y_km": float((track["y"].max() - track["y"].min()) / 1000),
        "duration_s": duration,
        "gps_path_km": length_km,
        "mean_speed_kmh": length_km / (duration / 3600) if duration > 0 else np.nan,
        "max_step_m": float(step.max()) if step.size else 0.0,
        "median_arc_length_m": float(ptrip["arcs"]["length_m"].median()),
    }


def summarise_partition(name: str, folder: Path) -> tuple[pd.DataFrame, np.ndarray]:
    rows, all_dt = [], []
    for trip_dir in list_trips(folder):
        trip = load_trip(trip_dir)
        ptrip = project_trip(trip)
        row = {"partition": name, **validate_trip(trip).as_dict(),
               **track_stats(ptrip), **network_stats(trip)}
        rows.append(row)
        all_dt.append(np.diff(trip["track"]["timestamp"].to_numpy(dtype=float)))
        print(f"[{name}] {trip['trip_id']}: {row['n_obs']} obs, {row['n_arcs']} arcs")
    dts = np.concatenate(all_dt) if all_dt else np.array([])
    return pd.DataFrame(rows), dts


def save_plots(summary: pd.DataFrame, dts: dict[str, np.ndarray], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("n_obs", "GPS observations per trip"),
        ("n_arcs", "Road arcs per trip"),
        ("max_gap_s", "Largest time gap per trip (s)"),
        ("mean_speed_kmh", "Mean GPS speed per trip (km/h)"),
    ]
    for col, title in specs:
        fig, ax = plt.subplots(figsize=(6, 4))
        for part, grp in summary.groupby("partition"):
            ax.hist(grp[col].dropna(), bins=20, alpha=0.6, label=part)
        ax.set_title(title)
        ax.set_xlabel(col)
        ax.set_ylabel("trips")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"eda_{col}.png", dpi=120)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for part, dt in dts.items():
        ax.hist(np.clip(dt, 0, 10), bins=np.arange(0, 11.5, 0.5), alpha=0.6, label=part)
    ax.set_yscale("log")
    ax.set_title("Sampling interval (clipped at 10 s)")
    ax.set_xlabel("seconds between GPS points")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "eda_sampling_interval.png", dpi=120)
    plt.close(fig)


def eda_map(trip_dir: Path, out_path: Path, margin_deg: float = 0.002) -> None:
    """GPS points (blue) + nearby road network (grey) for one training trip."""
    trip = load_trip(trip_dir, include_route=False)
    track, nodes, arcs = trip["track"], trip["nodes"], trip["arcs"]
    inside = (nodes["lat"].between(track["lat"].min() - margin_deg, track["lat"].max() + margin_deg)
              & nodes["lon"].between(track["lon"].min() - margin_deg,
                                     track["lon"].max() + margin_deg)).to_numpy()
    u, v = arcs["from_node_id"].to_numpy(), arcs["to_node_id"].to_numpy()
    keep = inside[u] | inside[v]
    lat, lon = nodes["lat"].to_numpy(), nodes["lon"].to_numpy()
    segments = [[(lat[a], lon[a]), (lat[b], lon[b])] for a, b in zip(u[keep], v[keep])]

    fmap = folium.Map(control_scale=True)
    roads = folium.FeatureGroup(name="Road network")
    folium.PolyLine(segments, color="#9e9e9e", weight=2, opacity=0.6).add_to(roads)
    roads.add_to(fmap)
    gps = folium.FeatureGroup(name="GPS observations")
    for la, lo in zip(track["lat"], track["lon"]):
        folium.CircleMarker((la, lo), radius=2, color="#1f4fd8", fill=True, weight=0,
                            fill_opacity=0.9).add_to(gps)
    gps.add_to(fmap)
    fmap.fit_bounds([[track["lat"].min(), track["lon"].min()],
                     [track["lat"].max(), track["lon"].max()]])
    folium.LayerControl(collapsed=False).add_to(fmap)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="PathSnap EDA on train + student_test.")
    parser.add_argument("--map-trip", default=None, help="training trip id for the Folium map")
    args = parser.parse_args()

    frames, dts = [], {}
    for name, folder in PARTITIONS.items():
        df, dt = summarise_partition(name, folder)
        frames.append(df)
        dts[name] = dt
    summary = pd.concat(frames, ignore_index=True)
    if summary.empty:
        raise SystemExit("no trips found - unzip the dataset into handouts/ first")

    out = RESULTS_DIR / "eda"
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "trip_summary.csv", index=False)
    numeric = summary.select_dtypes("number")
    agg = numeric.groupby(summary["partition"]).agg(["count", "mean", "min", "max"])
    agg.to_csv(out / "partition_summary.csv")
    save_plots(summary, dts, VIS_DIR / "plots")

    map_trip = args.map_trip or summary.loc[summary["partition"] == "train", "trip_id"].iloc[0]
    eda_map(TRAIN_DIR / map_trip, VIS_DIR / "maps" / f"eda_{map_trip}.html")
    print(f"wrote {out / 'trip_summary.csv'}, plots and eda_{map_trip}.html")


if __name__ == "__main__":
    main()
