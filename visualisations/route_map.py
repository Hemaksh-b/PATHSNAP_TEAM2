"""Interactive Folium maps for PathSnap map-matching (Issue #9).

Layers (each toggleable via LayerControl):
  * raw GPS observations ......... blue points
  * road network near the trip ... grey lines
  * reference route .............. green (training / student-test trips ONLY)
  * classifier-only route ........ orange dashed (optional)
  * predicted HMM route .......... red

CLI (repo root):
    python -m visualisations.route_map --trip 00000001 --partition train --reference
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import folium
import pandas as pd

PERMITTED_REFERENCE_PARTITIONS = frozenset({"train", "student_test"})
ROAD_MARGIN_DEG = 0.002  # ~200 m around the GPS extent

COLORS = {
    "gps": "#1f4fd8",
    "road": "#9e9e9e",
    "reference": "#1a9850",
    "classifier": "#f28e2b",
    "predicted": "#d62728",
}

LEGEND_HTML = """
<div style="position: fixed; bottom: 56px; left: 12px; z-index: 9999;
            background: white; padding: 8px 12px; border: 1px solid #888;
            border-radius: 6px; font: 13px sans-serif; line-height: 1.6;">
  <b>{title}</b><br>
  {rows}
</div>
"""


def _legend_row(color: str, label: str, dashed: bool = False) -> str:
    style = "dashed" if dashed else "solid"
    return (
        f'<span style="display:inline-block;width:22px;border-top:4px {style} {color};'
        f'vertical-align:middle;margin-right:6px"></span>{label}<br>'
    )


def _arc_segments(
    nodes: pd.DataFrame, arcs: pd.DataFrame, arc_ids: Sequence[int | None]
) -> list[list[tuple[float, float]]]:
    """(lat, lon) segment per arc_id in order; ``None`` entries are skipped."""
    ids = [int(a) for a in arc_ids if a is not None]
    if not ids:
        return []
    arc_rows = arcs.set_index("arc_id").loc[ids]
    lat, lon = nodes["lat"].to_numpy(), nodes["lon"].to_numpy()
    return [
        [(lat[u], lon[u]), (lat[v], lon[v])]
        for u, v in zip(arc_rows["from_node_id"], arc_rows["to_node_id"])
    ]


def _dedupe_consecutive(arc_ids: Sequence[int | None]) -> list[int]:
    out: list[int] = []
    for a in arc_ids:
        if a is not None and (not out or out[-1] != a):
            out.append(int(a))
    return out


def _nearby_road_segments(trip: dict, margin_deg: float) -> list[list[tuple[float, float]]]:
    track, nodes, arcs = trip["track"], trip["nodes"], trip["arcs"]
    lat_min, lat_max = track["lat"].min() - margin_deg, track["lat"].max() + margin_deg
    lon_min, lon_max = track["lon"].min() - margin_deg, track["lon"].max() + margin_deg
    inside = nodes["lat"].between(lat_min, lat_max) & nodes["lon"].between(lon_min, lon_max)
    keep = inside.to_numpy()
    mask = keep[arcs["from_node_id"].to_numpy()] | keep[arcs["to_node_id"].to_numpy()]
    return _arc_segments(nodes, arcs, arcs.loc[mask, "arc_id"].tolist())


def build_route_map(
    trip: dict,
    partition: str,
    predicted_arc_ids: Sequence[int | None] | None = None,
    reference_arc_ids: Sequence[int] | None = None,
    classifier_arc_ids: Sequence[int | None] | None = None,
    title: str | None = None,
    margin_deg: float = ROAD_MARGIN_DEG,
) -> folium.Map:
    """Build the comparison map for one trip (dict from ``data_processing.loader.load_trip``).

    Raises ``ValueError`` if a reference route is requested for a partition
    other than training / student-test (evaluator ground truth must never be shown).
    """
    if reference_arc_ids is not None and partition not in PERMITTED_REFERENCE_PARTITIONS:
        raise ValueError(
            f"reference route may only be drawn for {sorted(PERMITTED_REFERENCE_PARTITIONS)}, "
            f"not '{partition}'"
        )
    track = trip["track"]
    if track.empty:
        raise ValueError("trip has no GPS observations")
    title = title or f"Trip {trip['trip_id']} ({partition})"

    fmap = folium.Map(tiles="OpenStreetMap", control_scale=True)
    legend = []

    roads = folium.FeatureGroup(name="Road network", show=True)
    folium.PolyLine(_nearby_road_segments(trip, margin_deg), color=COLORS["road"],
                    weight=2, opacity=0.6).add_to(roads)
    roads.add_to(fmap)
    legend.append(_legend_row(COLORS["road"], "Road network"))

    route_layers = [
        ("reference", "Reference route", reference_arc_ids, 7, False),
        ("classifier", "Classifier-only route", classifier_arc_ids, 4, True),
        ("predicted", "Predicted route (HMM)", predicted_arc_ids, 4, False),
    ]
    for key, label, ids, weight, dashed in route_layers:
        if ids is None:
            continue
        group = folium.FeatureGroup(name=label, show=True)
        folium.PolyLine(
            _arc_segments(trip["nodes"], trip["arcs"], _dedupe_consecutive(ids)),
            color=COLORS[key], weight=weight, opacity=0.85,
            dash_array="8 6" if dashed else None, tooltip=label,
        ).add_to(group)
        group.add_to(fmap)
        legend.append(_legend_row(COLORS[key], label, dashed))

    gps = folium.FeatureGroup(name="GPS observations", show=True)
    for idx, (lat, lon, ts) in enumerate(track[["lat", "lon", "timestamp"]].itertuples(index=False)):
        folium.CircleMarker((lat, lon), radius=2, color=COLORS["gps"], fill=True,
                            fill_opacity=0.9, weight=0,
                            tooltip=f"obs {idx} · t={ts:g}s").add_to(gps)
    gps.add_to(fmap)
    legend.append(_legend_row(COLORS["gps"], "GPS observations"))

    fmap.fit_bounds([[track["lat"].min(), track["lon"].min()],
                     [track["lat"].max(), track["lon"].max()]])
    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.get_root().html.add_child(
        folium.Element(LEGEND_HTML.format(title=title, rows="".join(legend)))
    )
    return fmap


def _main() -> None:
    from data_processing.config import STUDENT_TEST_DIR, TRAIN_DIR, VIS_DIR
    from data_processing.loader import load_trip

    parser = argparse.ArgumentParser(description="Render a Folium map for one trip.")
    parser.add_argument("--trip", required=True, help="trip id, e.g. 00000001")
    parser.add_argument("--partition", choices=sorted(PERMITTED_REFERENCE_PARTITIONS),
                        default="train")
    parser.add_argument("--reference", action="store_true", help="draw the reference route")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    base = TRAIN_DIR if args.partition == "train" else STUDENT_TEST_DIR
    trip = load_trip(base / args.trip)
    reference = None
    if args.reference and trip["route"] is not None:
        reference = trip["route"]["arc_id"].tolist()
    fmap = build_route_map(trip, args.partition, reference_arc_ids=reference)
    out = args.out or VIS_DIR / "maps" / f"{args.trip}_{args.partition}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out))
    print(f"saved {out}")


if __name__ == "__main__":
    _main()
