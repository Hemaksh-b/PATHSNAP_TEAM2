"""Tests for the Folium route map builder (Issue #9).

Run from the repo root:  python -m pytest visualisations
"""
import pandas as pd
import pytest

from visualisations.route_map import build_route_map


def toy_trip():
    nodes = pd.DataFrame({"lon": [8.0, 8.001, 8.002, 8.5, 8.7],
                          "lat": [50.0, 50.0, 50.0, 50.5, 50.7]})
    nodes["node_id"] = nodes.index
    arcs = pd.DataFrame({"from_node_id": [0, 1, 2, 3], "to_node_id": [1, 2, 3, 4]})
    arcs["arc_id"] = arcs.index
    track = pd.DataFrame({"lon": [8.0002, 8.0012], "lat": [50.0001, 49.9999],
                          "timestamp": [0.0, 1.0]})
    return {"trip_id": "toy", "track": track, "nodes": nodes, "arcs": arcs,
            "route": pd.DataFrame({"arc_id": [0, 1]})}


def html(fmap):
    return fmap.get_root().render()


def test_map_has_all_requested_layers_and_legend():
    out = html(build_route_map(toy_trip(), "train", predicted_arc_ids=[0, None, 1],
                               reference_arc_ids=[0, 1], classifier_arc_ids=[0, 2]))
    for label in ["Road network", "Reference route", "Classifier-only route",
                  "Predicted route (HMM)", "GPS observations"]:
        assert label in out
    assert "#d62728" in out and "#1a9850" in out and "#1f4fd8" in out  # red, green, blue
    assert "Trip toy (train)" in out


def test_optional_layers_are_omitted():
    out = html(build_route_map(toy_trip(), "train"))
    assert "Reference route" not in out
    assert "Predicted route (HMM)" not in out
    assert "GPS observations" in out


@pytest.mark.parametrize("partition", ["evaluator_test", "eval", ""])
def test_reference_route_refused_outside_permitted_partitions(partition):
    with pytest.raises(ValueError, match="reference route"):
        build_route_map(toy_trip(), partition, reference_arc_ids=[0, 1])


def test_evaluator_trip_without_reference_is_allowed():
    out = html(build_route_map(toy_trip(), "evaluator_test", predicted_arc_ids=[0, 1]))
    assert "Predicted route (HMM)" in out
    assert "Reference route" not in out


def test_far_away_roads_are_clipped():
    out = html(build_route_map(toy_trip(), "train"))
    assert "50.5, 8.5" in out  # arc 2->3 leaves the area but starts near the trip: kept
    assert "50.7" not in out  # arc 3->4 lies entirely ~60 km away: dropped


def test_empty_track_rejected():
    trip = toy_trip()
    trip["track"] = trip["track"].iloc[0:0]
    with pytest.raises(ValueError):
        build_route_map(trip, "train")
