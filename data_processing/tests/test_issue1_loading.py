"""Tests for loading, projection and validation (Issue #1).

Run from the repo root:  python -m pytest data_processing
"""
import numpy as np
import pandas as pd
import pytest

try:
    from pyproj import Transformer
except ImportError:  # pragma: no cover
    Transformer = None  # type: ignore

from data_processing import loader
from data_processing.loader import list_trips, load_trip
from data_processing.projection import project_trip, utm_epsg_for
from data_processing.validation import validate_trip


def write_trip(folder, trip_id="00000042", route=True):
    d = folder / trip_id
    d.mkdir(parents=True)
    (d / f"{trip_id}.track").write_text("8.60 50.10 100\n8.601 50.10 101\n8.602 50.10 102\n")
    (d / f"{trip_id}.nodes").write_text("8.60 50.10\n8.601 50.10\n8.602 50.10\n")
    (d / f"{trip_id}.arcs").write_text("0 1\n1 2\n")
    if route:
        (d / f"{trip_id}.route").write_text("0\n1\n")
    return d


def toy_trip(**overrides):
    trip = {
        "trip_id": "t",
        "track": pd.DataFrame({"lon": [8.6, 8.601, 8.601, 8.602],
                               "lat": [50.1, 50.1, 50.1, 50.1],
                               "timestamp": [0.0, 1.0, 1.0, 40.0]}),
        "nodes": pd.DataFrame({"lon": [8.6, 8.601, 8.602], "lat": [50.1, 50.1, 50.1],
                               "node_id": [0, 1, 2]}),
        "arcs": pd.DataFrame({"from_node_id": [0, 1], "to_node_id": [1, 2], "arc_id": [0, 1]}),
        "route": pd.DataFrame({"arc_id": [0, 1]}),
    }
    trip.update(overrides)
    return trip


# ---------------------------------------------------------------- loader

def test_load_trip_parses_columns_and_ids(tmp_path):
    trip = load_trip(write_trip(tmp_path))
    assert trip["trip_id"] == "00000042"
    assert list(trip["track"].columns) == ["lon", "lat", "timestamp"]
    assert trip["nodes"]["node_id"].tolist() == [0, 1, 2]
    assert trip["arcs"]["arc_id"].tolist() == [0, 1]
    assert trip["route"]["arc_id"].tolist() == [0, 1]


def test_include_route_false_never_reads_route(tmp_path):
    assert load_trip(write_trip(tmp_path), include_route=False)["route"] is None


def test_missing_route_file_gives_none(tmp_path):
    assert load_trip(write_trip(tmp_path, route=False))["route"] is None


def test_evaluator_route_is_refused(tmp_path, monkeypatch):
    eval_dir = tmp_path / "evaluator_test"
    monkeypatch.setattr(loader, "EVALUATOR_TEST_DIR", eval_dir)
    trip_dir = write_trip(eval_dir)
    with pytest.raises(PermissionError):
        load_trip(trip_dir)
    assert load_trip(trip_dir, include_route=False)["route"] is None


def test_list_trips_sorted_and_missing_dir(tmp_path):
    write_trip(tmp_path, "00000002")
    write_trip(tmp_path, "00000001")
    assert [p.name for p in list_trips(tmp_path)] == ["00000001", "00000002"]
    assert list_trips(tmp_path / "nope") == []


# ---------------------------------------------------------------- projection

@pytest.mark.parametrize("lon,lat,epsg", [
    (8.6, 50.1, 32632),     # Frankfurt
    (-86.6, 34.7, 32616),   # Alabama
    (37.2, 55.9, 32637),    # Moscow region
    (151.2, -33.9, 32756),  # Sydney (southern hemisphere)
    (180.0, 0.0, 32660),    # zone clamp at the antimeridian
])
def test_utm_zone_selection(lon, lat, epsg):
    assert utm_epsg_for(lon, lat) == epsg


def test_utm_rejects_polar():
    with pytest.raises(ValueError):
        utm_epsg_for(0.0, 89.0)


def test_project_trip_is_metric_and_does_not_mutate_input():
    trip = toy_trip()
    before = trip["track"].copy()
    p = project_trip(trip)
    pd.testing.assert_frame_equal(trip["track"], before)
    assert "x" not in trip["track"].columns
    assert p["crs_epsg"] == 32632
    # 0.001 deg of longitude at 50.1 N is ~71.4 m
    assert p["arcs"]["length_m"].tolist() == pytest.approx([71.4, 71.4], abs=0.5)


def test_projection_round_trip():
    p = project_trip(toy_trip())
    if Transformer is not None:
        inv = Transformer.from_crs(f"EPSG:{p['crs_epsg']}", "EPSG:4326", always_xy=True)
        lon, lat = inv.transform(p["nodes"]["x"].to_numpy(), p["nodes"]["y"].to_numpy())
    else:
        cos_lat = float(np.cos(np.radians(float(toy_trip()["track"]["lat"].mean()))))
        lon = p["nodes"]["x"].to_numpy() / (111_320.0 * cos_lat)
        lat = p["nodes"]["y"].to_numpy() / 111_132.0
    assert np.allclose(lon, p["nodes"]["lon"], atol=1e-9)
    assert np.allclose(lat, p["nodes"]["lat"], atol=1e-9)


# ---------------------------------------------------------------- validation

def test_validation_counts_known_problems():
    r = validate_trip(toy_trip())
    assert r.n_obs == 4
    assert r.n_duplicate_points == 1
    assert r.n_repeated_timestamps == 1
    assert r.n_backward_timestamps == 0
    assert r.n_large_gaps == 1 and r.max_gap_s == 39.0
    assert r.n_invalid_arc_endpoints == 0
    assert r.route_connected_share == 1.0


def test_validation_flags_bad_ids_and_coords():
    trip = toy_trip(
        arcs=pd.DataFrame({"from_node_id": [0, 1], "to_node_id": [9, 1], "arc_id": [0, 1]}),
        route=pd.DataFrame({"arc_id": [0, 7]}),
        track=pd.DataFrame({"lon": [200.0, 8.6], "lat": [50.1, 50.1], "timestamp": [2.0, 1.0]}),
    )
    r = validate_trip(trip)
    assert r.n_invalid_arc_endpoints == 1
    assert r.n_self_loop_arcs == 1
    assert r.n_invalid_route_arcs == 1
    assert r.n_bad_coords == 1
    assert r.n_backward_timestamps == 1


def test_validation_without_route():
    r = validate_trip(toy_trip(route=None))
    assert r.route_len is None and r.route_connected_share is None
