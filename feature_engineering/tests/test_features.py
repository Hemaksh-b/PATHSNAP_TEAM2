"""Comprehensive unit tests for feature engineering, label alignment, and data leakage (Issue #3).

Run with:
    python -m pytest feature_engineering/tests -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from feature_engineering.alignment import AlignmentConfig, align_observations_to_route
from feature_engineering.features import FEATURE_COLUMNS, extract_candidate_features


def make_toy_trip(route: bool = True, offset_y: float = 0.0) -> dict:
    """Create a synthetic 3-segment straight line trip in UTM metres."""
    # Nodes along x-axis: (0, 0) -> (100, 0) -> (200, 0) -> (300, 0)
    nodes = pd.DataFrame(
        {
            "node_id": [0, 1, 2, 3],
            "x": [0.0, 100.0, 200.0, 300.0],
            "y": [0.0, 0.0, 0.0, 0.0],
            "lon": [8.00, 8.01, 8.02, 8.03],
            "lat": [50.0, 50.0, 50.0, 50.0],
        }
    )
    # Arcs: 0: 0->1, 1: 1->2, 2: 2->3
    arcs = pd.DataFrame(
        {
            "arc_id": [0, 1, 2],
            "from_node_id": [0, 1, 2],
            "to_node_id": [1, 2, 3],
            "length_m": [100.0, 100.0, 100.0],
        }
    )
    # GPS track: 6 points moving from x=10 to x=250 along y=offset_y
    track = pd.DataFrame(
        {
            "obs_idx": [0, 1, 2, 3, 4, 5],
            "x": [10.0, 50.0, 110.0, 150.0, 210.0, 250.0],
            "y": [offset_y] * 6,
            "timestamp": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "lon": [8.001, 8.005, 8.011, 8.015, 8.021, 8.025],
            "lat": [50.0, 50.0, 50.0, 50.0, 50.0, 50.0],
        }
    )
    route_df = pd.DataFrame({"arc_id": [0, 1, 2]}) if route else None

    return {
        "trip_id": "test_toy",
        "track": track,
        "nodes": nodes,
        "arcs": arcs,
        "route": route_df,
        "projection_crs": "EPSG:32632",
    }


def test_monotonic_route_alignment():
    trip = make_toy_trip(route=True, offset_y=2.0)
    alignment = align_observations_to_route(trip)
    assert alignment is not None
    assert len(alignment) == 6
    # Points 0, 1 are on arc 0; points 2, 3 are on arc 1; points 4, 5 are on arc 2
    assert alignment["aligned_arc_id"].tolist() == [0, 0, 1, 1, 2, 2]
    # Route positions must be non-decreasing (monotonic)
    positions = alignment["route_pos"].tolist()
    assert all(positions[i] <= positions[i + 1] for i in range(len(positions) - 1))
    # All distances should be approximately 2.0 metres
    assert np.allclose(alignment["aligned_dist_m"], 2.0, atol=1e-3)
    assert not alignment["is_uncertain"].any()


def test_alignment_uncertainty_threshold():
    # Observations offset by 45 metres (exceeds default 35m threshold)
    trip = make_toy_trip(route=True, offset_y=45.0)
    config = AlignmentConfig(max_dist_threshold_m=35.0)
    alignment = align_observations_to_route(trip, config=config)
    assert alignment is not None
    assert alignment["is_uncertain"].all()


def test_leakage_guard_identical_features_without_route():
    """Verify that inference feature extraction NEVER touches route and produces 100% identical features."""
    trip_with_route = make_toy_trip(route=True, offset_y=3.0)
    trip_without_route = make_toy_trip(route=False, offset_y=3.0)

    features_infer = extract_candidate_features(trip_without_route, k=3, include_labels=False)
    features_blind = extract_candidate_features(trip_with_route, k=3, include_labels=False)

    # Both must lack label columns
    assert "label" not in features_infer.columns
    assert "label_uncertain" not in features_infer.columns
    assert "label" not in features_blind.columns

    # Numeric feature values must match bit-for-bit
    pd.testing.assert_frame_equal(features_infer, features_blind)


def test_feature_columns_and_bounds():
    trip = make_toy_trip(route=True, offset_y=5.0)
    df = extract_candidate_features(trip, k=3, radius_m=100.0, include_labels=True)

    # Check presence of all required feature columns
    for col in FEATURE_COLUMNS:
        assert col in df.columns, f"Missing feature {col}"

    assert "label" in df.columns
    assert "label_uncertain" in df.columns
    assert not df[FEATURE_COLUMNS].isna().any().any()

    # Mathematical bounds
    assert (df["dist_m"] >= 0.0).all()
    assert (df["cand_rank"].isin([0, 1, 2])).all()
    assert (df["heading_diff_deg"].between(0.0, 180.0)).all()
    assert (df["cos_heading_diff"].between(-1.0, 1.0)).all()
    assert (df["sin_heading_diff"].between(0.0, 1.0)).all()
    assert (df["frac_along"].between(0.0, 1.0)).all()
    assert (df["gps_speed_mps"] >= 0.0).all()
    assert (df["dt_prev_s"] >= 0.0).all()
    assert (df["dt_next_s"] >= 0.0).all()
    assert (df["out_degree_v"] >= 0).all()
    assert (df["in_degree_u"] >= 0).all()


def test_schema_validator_compatibility():
    """Verify that output passes model_training.schema.validate_and_prepare_features without issues."""
    import importlib.util
    from pathlib import Path

    schema_path = Path(__file__).resolve().parent.parent.parent / "model_training" / "schema.py"
    spec = importlib.util.spec_from_file_location("model_training_schema", schema_path)
    schema_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(schema_mod)

    trip = make_toy_trip(route=True, offset_y=2.0)
    df = extract_candidate_features(trip, k=3, radius_m=100.0, include_labels=True)

    clean_df, feature_cols = schema_mod.validate_and_prepare_features(df)
    assert len(clean_df) == len(df)
    assert set(FEATURE_COLUMNS).issubset(set(feature_cols))
