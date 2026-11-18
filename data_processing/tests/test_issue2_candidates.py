"""Tests for candidate generation (Issue #2) on synthetic projected networks."""
import numpy as np
import pandas as pd
import pytest

from data_processing.candidates import (
    CANDIDATE_COLUMNS,
    candidate_lists,
    generate_candidates,
    observations_without_candidates,
)
from data_processing.miss_rate import approx_reference_arc, sweep_trip


def projected_trip(track_xy, nodes_xy, arcs_uv, arc_ids=None, route=None):
    """Trip dict that is already 'projected' (x/y in metres)."""
    track = pd.DataFrame(track_xy, columns=["x", "y"])
    track["timestamp"] = np.arange(len(track), dtype=float)
    nodes = pd.DataFrame(nodes_xy, columns=["x", "y"])
    nodes["node_id"] = nodes.index
    arcs = pd.DataFrame(arcs_uv, columns=["from_node_id", "to_node_id"])
    arcs["arc_id"] = arc_ids if arc_ids is not None else arcs.index
    return {"trip_id": "syn", "track": track, "nodes": nodes, "arcs": arcs,
            "route": None if route is None else pd.DataFrame({"arc_id": route})}


# two parallel horizontal roads (y=0 and y=20) and a far road (y=500)
NODES = [[0, 0], [100, 0], [0, 20], [100, 20], [0, 500], [100, 500]]
ARCS = [[0, 1], [2, 3], [4, 5]]


def test_nearest_by_perpendicular_distance_and_rank():
    trip = projected_trip([[50, 5]], NODES, ARCS)
    c = generate_candidates(trip, k=5, radius_m=50)
    assert list(c.columns) == CANDIDATE_COLUMNS
    assert c["arc_id"].tolist() == [0, 1]  # far road excluded by radius
    assert c["cand_rank"].tolist() == [0, 1]
    assert c["dist_m"].tolist() == pytest.approx([5.0, 15.0])


def test_distance_is_to_segment_not_to_nodes():
    # point above the middle of a 100 m arc: node distance ~50 m, segment distance 3 m
    trip = projected_trip([[50, 3]], [[0, 0], [100, 0]], [[0, 1]])
    c = generate_candidates(trip, k=1, radius_m=10)
    assert c["dist_m"].iloc[0] == pytest.approx(3.0)


def test_projection_point_and_frac_along():
    trip = projected_trip([[25, 4]], [[0, 0], [100, 0]], [[0, 1]])
    c = generate_candidates(trip, k=1, radius_m=10).iloc[0]
    assert c["frac_along"] == pytest.approx(0.25)
    assert (c["proj_x"], c["proj_y"]) == pytest.approx((25.0, 0.0))


def test_frac_follows_arc_direction():
    trip = projected_trip([[25, 4]], [[0, 0], [100, 0]], [[1, 0]])  # arc runs right-to-left
    assert generate_candidates(trip, k=1, radius_m=10)["frac_along"].iloc[0] == pytest.approx(0.75)


def test_k_limits_candidates():
    trip = projected_trip([[50, 5]], NODES, ARCS)
    assert len(generate_candidates(trip, k=1, radius_m=50)) == 1


def test_observation_without_candidates_gets_no_rows():
    trip = projected_trip([[50, 5], [50, 250]], NODES, ARCS)
    c = generate_candidates(trip, k=3, radius_m=30)
    assert set(c["obs_idx"]) == {0}
    assert observations_without_candidates(c, 2).tolist() == [1]
    arcs, fracs, dists = candidate_lists(c, 2)
    assert arcs[1] == [] and fracs[1] == [] and dists[1].size == 0


def test_no_candidates_at_all_returns_empty_frame():
    trip = projected_trip([[50, 250]], NODES, ARCS)
    c = generate_candidates(trip, k=3, radius_m=10)
    assert c.empty and list(c.columns) == CANDIDATE_COLUMNS


def test_original_arc_ids_preserved():
    trip = projected_trip([[50, 5]], NODES, ARCS, arc_ids=[700, 31, 9])
    assert generate_candidates(trip, k=2, radius_m=50)["arc_id"].tolist() == [700, 31]


def test_zero_length_arc_does_not_produce_nan():
    trip = projected_trip([[3, 4]], [[0, 0], [0, 0]], [[0, 1]])
    c = generate_candidates(trip, k=1, radius_m=10).iloc[0]
    assert c["dist_m"] == pytest.approx(5.0)
    assert c["frac_along"] == 0.0


def test_invalid_parameters_rejected():
    trip = projected_trip([[0, 0]], NODES, ARCS)
    with pytest.raises(ValueError):
        generate_candidates(trip, k=0)
    with pytest.raises(ValueError):
        generate_candidates(trip, radius_m=0)


def test_raw_lon_lat_trip_is_projected_automatically():
    raw = {
        "trip_id": "raw",
        "track": pd.DataFrame({"lon": [8.6005], "lat": [50.1], "timestamp": [0.0]}),
        "nodes": pd.DataFrame({"lon": [8.600, 8.601], "lat": [50.1, 50.1], "node_id": [0, 1]}),
        "arcs": pd.DataFrame({"from_node_id": [0], "to_node_id": [1], "arc_id": [0]}),
        "route": None,
    }
    c = generate_candidates(raw, k=1, radius_m=10)
    assert c["dist_m"].iloc[0] < 1.0


def test_miss_rate_sweep_counts_misses():
    # obs 0 near reference arc 0; obs 1 near non-reference arc 1 only
    trip = projected_trip([[50, 2], [50, 19]], NODES, ARCS, route=[0])
    assert approx_reference_arc(trip).tolist() == [0, 0]
    rows = {(r["k"], r["radius_m"]): r for r in sweep_trip(trip, k_values=(1, 3), radii=(5.0, 30.0))}
    assert rows[(1, 5.0)]["misses"] == 1  # obs 1 only sees arc 1 within 5 m
    assert rows[(3, 30.0)]["misses"] == 0  # widening the radius recovers it
