"""Tests for the pointwise vs HMM comparison charts (Issue #9)."""
import pandas as pd
import pytest

from visualisations.comparison import bar_summary, paired, per_trip_scatter


@pytest.fixture()
def per_trip():
    rows = []
    for trip, pw, hmm in [("00000001", 0.5, 0.9), ("00000002", 0.7, 0.72), ("00000003", 0.6, 0.8)]:
        for method, acc in (("pointwise", pw), ("hmm", hmm)):
            rows.append({"trip_id": trip, "method": method, "emission_source": "gaussian_distance",
                         "observation_accuracy": acc, "route_continuity": acc, "route_f1": acc})
    return pd.DataFrame(rows)


def test_paired_sorts_by_hmm_gain(per_trip):
    wide = paired(per_trip, "observation_accuracy")
    assert list(wide.index) == ["00000002", "00000003", "00000001"]
    assert wide.loc["00000001", "delta"] == pytest.approx(0.4)


def test_plots_are_written(per_trip, tmp_path):
    assert bar_summary(per_trip, tmp_path).exists()
    assert per_trip_scatter(per_trip, "route_continuity", tmp_path).exists()
