
import numpy as np
import pytest

from data_processing.pipeline import (
    load_trip,
    choose_utm_crs,
    project_lonlat,
    inverse_project_xy,
)


def make_trip(tmp_path, timestamp="0\n1\n2", arc="0 1\n"):
    folder = tmp_path / "train" / "trip001"
    folder.mkdir(parents=True)

    times = timestamp.splitlines()
    with open(folder / "trip001.track", "w") as f:
        for i, t in enumerate(times):
            f.write(f"77.59{i} 12.97{i} {t}\n")

    (folder / "trip001.nodes").write_text(
        "77.5946 12.9716\n77.5950 12.9720\n"
    )
    (folder / "trip001.arcs").write_text(arc)
    (folder / "trip001.route").write_text("0\n")
    return folder


def test_projection_round_trip():
    lon = np.array([77.5946, 77.5950])
    lat = np.array([12.9716, 12.9720])
    crs = choose_utm_crs(lon, lat)

    x, y = project_lonlat(lon, lat, crs)
    lon2, lat2 = inverse_project_xy(x, y, crs)

    assert np.allclose(lon, lon2, atol=1e-6)
    assert np.allclose(lat, lat2, atol=1e-6)


def test_loader_columns_and_indices(tmp_path):
    folder = make_trip(tmp_path)
    trip = load_trip(folder, partition="train")

    assert list(trip.track.columns) == [
        "obs_idx", "lon", "lat", "t", "x", "y"
    ]
    assert trip.nodes.node_id.tolist() == [0, 1]
    assert trip.arcs.arc_id.tolist() == [0]
    assert trip.route == [0]


def test_invalid_node_index(tmp_path):
    folder = make_trip(tmp_path, arc="0 5\n")
    with pytest.raises(ValueError):
        load_trip(folder, partition="train")


def test_out_of_order_timestamp(tmp_path):
    folder = make_trip(tmp_path, timestamp="0\n2\n1")
    with pytest.raises(ValueError):
        load_trip(folder, partition="train")


def test_evaluator_route_protection(tmp_path):
    folder = tmp_path / "evaluator_test" / "trip001"
    folder.mkdir(parents=True)

    (folder / "trip001.track").write_text(
        "77.5946 12.9716 0\n77.5950 12.9720 1\n"
    )
    (folder / "trip001.nodes").write_text(
        "77.5946 12.9716\n77.5950 12.9720\n"
    )
    (folder / "trip001.arcs").write_text("0 1\n")
    (folder / "trip001.route").write_text("0\n")

    with pytest.raises(PermissionError):
        load_trip(folder, partition="evaluator_test")
