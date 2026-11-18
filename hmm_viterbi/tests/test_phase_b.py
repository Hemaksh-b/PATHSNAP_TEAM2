"""Tests for real-trip decoding, route metrics and the mock-model guard (Issues #7 / #9)."""
import numpy as np
import pandas as pd
import pytest

from hmm_viterbi import run_experiments
from hmm_viterbi.decode_trip import (GAUSSIAN_SOURCE, Classifier, DecodeSettings, decode_prepared,
                                     load_settings, prepare_trip)
from hmm_viterbi.route_metrics import (all_metrics, observation_accuracy, route_continuity,
                                       route_set_scores)

ARCS = pd.DataFrame({"from_node_id": [0, 1, 2, 5], "to_node_id": [1, 2, 3, 6], "arc_id": [0, 1, 2, 3]})


def write_trip(root, trip_id="00000001", n=12):
    """East-west road of three ~71 m arcs plus a disconnected parallel road ~67 m north."""
    d = root / trip_id
    d.mkdir(parents=True)
    lons = [8.6000 + 0.00025 * i for i in range(n)]
    (d / f"{trip_id}.track").write_text("".join(f"{lo:.5f} 50.10003 {i}\n" for i, lo in enumerate(lons)))
    nodes = "8.6000 50.1\n8.6010 50.1\n8.6020 50.1\n8.6030 50.1\n8.6000 50.1006\n8.6000 50.1006\n8.6030 50.1006\n"
    (d / f"{trip_id}.nodes").write_text(nodes)
    (d / f"{trip_id}.arcs").write_text("0 1\n1 2\n2 3\n5 6\n")
    (d / f"{trip_id}.route").write_text("0\n1\n2\n")
    return d


# ---------------------------------------------------------------- metrics

def test_route_continuity():
    assert route_continuity([0, 0, 1, None, 2], ARCS) == 1.0
    assert route_continuity([0, 3], ARCS) == 0.0
    assert route_continuity([0, 0, None], ARCS) is None


def test_observation_accuracy_and_mask():
    aligned = np.array([0, 0, 1, 2])
    assert observation_accuracy([0, 1, 1, None], aligned) == 0.5
    assert observation_accuracy([0, 1, 1, None], aligned, np.array([1, 0, 1, 0])) == 1.0


def test_route_set_scores():
    s = route_set_scores([0, 1, 3], [0, 1, 2])
    assert s["route_precision"] == pytest.approx(2 / 3) and s["route_recall"] == pytest.approx(2 / 3)
    assert route_set_scores([], [0])["route_f1"] is None


def test_all_metrics_keys():
    m = all_metrics([0, 1], np.array([0, 1]), np.array([True, True]), [0, 1], ARCS)
    assert m["observation_accuracy"] == 1.0 and m["route_continuity"] == 1.0


# ---------------------------------------------------------------- decoding

def test_settings_round_trip(tmp_path):
    s = DecodeSettings(k=4)
    (tmp_path / "hmm_config.json").write_text(s.to_json())
    assert load_settings(tmp_path) == s
    assert load_settings(tmp_path / "missing") == DecodeSettings()


def test_gaussian_decoding_stays_on_connected_road(tmp_path):
    d = write_trip(tmp_path)
    s = DecodeSettings(k=4, radius_m=50)
    prep = prepare_trip(d, s)
    assert prep.trip["route"] is None  # inference never loads the reference route
    out = decode_prepared(prep, s)
    assert out.emission_source == GAUSSIAN_SOURCE
    assert len(out.hmm_arc_ids) == 12
    assert set(out.hmm_arc_ids) <= {0, 1, 2}  # disconnected parallel road never chosen
    assert route_continuity(out.hmm_arc_ids, ARCS) == 1.0


def test_classifier_scores_drive_pointwise_choice(tmp_path, monkeypatch):
    d = write_trip(tmp_path)
    s = DecodeSettings(k=4, radius_m=50)
    prep = prepare_trip(d, s)
    clf = Classifier(pipeline=None, feature_columns=["dist_m"], card={"model_key": "stub"})
    monkeypatch.setattr(Classifier, "probabilities",
                        lambda self, f: (f["arc_id"].to_numpy() + 1) / 10.0)
    out = decode_prepared(prep, s, clf)
    assert out.emission_source == "classifier:stub"
    expected = prep.features.groupby("obs_idx")["arc_id"].max()
    for obs, arc in expected.items():
        assert out.pointwise_arc_ids[obs] == arc


# ---------------------------------------------------------------- mock guard

MOCK = Classifier(pipeline=None, feature_columns=[], card={"model_key": "lr", "is_mock_data": True})
REAL = Classifier(pipeline=None, feature_columns=[], card={"model_key": "lr", "is_mock_data": False})


def test_mock_card_detected():
    assert MOCK.is_mock and not REAL.is_mock
    assert Classifier(None, [], {"data_source": "MOCK_SYNTHETIC_DO_NOT_REPORT"}).is_mock


def test_auto_mode_falls_back_to_gaussian_on_mock_model(monkeypatch, tmp_path):
    monkeypatch.setattr(run_experiments, "load_classifier", lambda _: MOCK)
    assert run_experiments.choose_classifier("auto", False, tmp_path) == (None, False)


def test_classifier_mode_refuses_mock_without_flag(monkeypatch, tmp_path):
    monkeypatch.setattr(run_experiments, "load_classifier", lambda _: MOCK)
    with pytest.raises(SystemExit):
        run_experiments.choose_classifier("classifier", False, tmp_path)
    assert run_experiments.choose_classifier("classifier", True, tmp_path) == (MOCK, True)


def test_real_model_is_used(monkeypatch, tmp_path):
    monkeypatch.setattr(run_experiments, "load_classifier", lambda _: REAL)
    assert run_experiments.choose_classifier("auto", False, tmp_path) == (REAL, False)
    assert run_experiments.choose_classifier("gaussian", False, tmp_path) == (None, False)
