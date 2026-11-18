
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import folium

from .pipeline import list_trips, load_trip, inspect_partition

OUT = Path("data_processing/eda_outputs")
PLOTS = OUT / "plots"
MAPS = OUT / "maps"
TABLES = OUT / "tables"


def main():
    for folder in (PLOTS, MAPS, TABLES):
        folder.mkdir(parents=True, exist_ok=True)

    train_report = inspect_partition("train")
    student_report = inspect_partition("student_test")

    train_report.to_csv(TABLES / "train_inspection.csv", index=False)
    student_report.to_csv(TABLES / "student_test_inspection.csv", index=False)

    trips = list_trips("train")
    observations = []
    intervals = []
    network_rows = []

    for folder in trips:
        trip = load_trip(folder, partition="train")
        observations.append(len(trip.track))
        intervals.extend(trip.track.t.diff().dropna().tolist())

        graph = nx.Graph()
        graph.add_nodes_from(trip.nodes.node_id)
        graph.add_edges_from(trip.arcs[["u", "v"]].itertuples(index=False, name=None))

        network_rows.append({
            "trip_id": trip.trip_id,
            "nodes": len(trip.nodes),
            "arcs": len(trip.arcs),
            "components": nx.number_connected_components(graph),
            "isolated_nodes": len(list(nx.isolates(graph)))
        })

    pd.DataFrame(network_rows).to_csv(
        TABLES / "network_statistics.csv", index=False
    )

    plt.figure()
    plt.hist(observations, bins=20)
    plt.title("GPS observations per training trip")
    plt.xlabel("Observations")
    plt.ylabel("Trips")
    plt.savefig(PLOTS / "observations_per_trip.png")
    plt.close()

    plt.figure()
    plt.hist(intervals, bins=50)
    plt.title("GPS sampling interval distribution")
    plt.xlabel("Seconds")
    plt.ylabel("Frequency")
    plt.savefig(PLOTS / "sampling_intervals.png")
    plt.close()

    if trips:
        trip = load_trip(trips[0], partition="train")
        fmap = folium.Map(
            location=[trip.track.lat.mean(), trip.track.lon.mean()],
            zoom_start=13
        )

        points = list(zip(trip.track.lat, trip.track.lon))
        folium.PolyLine(points, color="blue", tooltip="GPS track").add_to(fmap)

        nodes = trip.nodes.set_index("node_id")
        for _, arc in trip.arcs.iterrows():
            u = nodes.loc[arc.u]
            v = nodes.loc[arc.v]
            folium.PolyLine(
                [[u.lat, u.lon], [v.lat, v.lon]],
                color="gray",
                weight=1
            ).add_to(fmap)

        fmap.save(MAPS / "training_trip_map.html")

    (OUT / "DATA_QUALITY_LOG.md").write_text(
        "# PathSnap Data Quality Log\n\n"
        f"- Training trips inspected: {len(train_report)}\n"
        f"- Student-testing trips inspected: {len(student_report)}\n"
        "- Evaluator-testing reference routes were not used.\n"
        "- Input files are whitespace-separated and headerless.\n"
        "- Coordinates are interpreted as longitude, latitude.\n"
        "- Node and arc IDs retain their original zero-based row indices.\n"
        "- A local UTM CRS is used for metre-based calculations.\n"
        "- Duplicate GPS points, repeated timestamps and gaps over 5 seconds "
        "are recorded in the inspection CSV files.\n",
        encoding="utf-8"
    )

    print("EDA completed. Outputs saved in", OUT)


if __name__ == "__main__":
    main()
