"""Tests for model selection and the live request log (Issue #30)."""
import pytest
from fastapi.testclient import TestClient

from api import pipeline, request_log
from api.main import app
from api.tests.test_issue10_api import write_trip


@pytest.fixture()
def client(tmp_path, monkeypatch):
    write_trip(tmp_path)
    monkeypatch.setenv("PATHSNAP_TRIPS_DIR", str(tmp_path))
    monkeypatch.setenv("PATHSNAP_MOCK", "0")
    request_log.clear()
    return TestClient(app)


def test_models_lists_all_four_options(client):
    body = client.get("/models").json()
    assert [m["id"] for m in body["models"]] == ["hmm_classifier", "classifier_only",
                                                 "hmm_distance", "nearest_road"]
    assert body["default"] in {"hmm_classifier", "hmm_distance"}


@pytest.mark.parametrize("model", ["hmm_distance", "nearest_road"])
def test_predict_with_each_classifier_free_model(client, model):
    body = client.post("/predict", json={"trip_id": "00000001", "model": model}).json()
    assert body["model"] == model and body["n_obs"] == 10
    assert body["emission_source"] == "gaussian_distance"
    assert body["metrics"]["observation_accuracy"] is not None
    assert "Reference route" not in body["map_html"]


def test_hmm_and_pointwise_options_differ_in_segments_field(client):
    hmm = client.post("/predict", json={"trip_id": "00000001", "model": "hmm_distance"}).json()
    nearest = client.post("/predict", json={"trip_id": "00000001", "model": "nearest_road"}).json()
    assert nearest["n_segments"] == 1 and nearest["breaks"] == []
    assert hmm["model_label"] != nearest["model_label"]


@pytest.mark.parametrize("bad", ["Bad-Model", "../x"])
def test_malformed_model_is_422(client, bad):
    assert client.post("/predict", json={"trip_id": "00000001", "model": bad}).status_code == 422


def test_unknown_model_is_422(client):
    r = client.post("/predict", json={"trip_id": "00000001", "model": "svm"})
    assert r.status_code == 422 and "unknown model" in r.json()["detail"]


def test_classifier_options_unavailable_without_real_model(client, monkeypatch):
    monkeypatch.setattr(pipeline, "get_classifier", lambda key=None: None)
    body = client.get("/models").json()
    avail = {m["id"]: m["available"] for m in body["models"]}
    assert avail == {"hmm_classifier": False, "classifier_only": False,
                     "hmm_distance": True, "nearest_road": True}
    assert body["default"] == "hmm_distance"
    r = client.post("/predict", json={"trip_id": "00000001", "model": "classifier_only"})
    assert r.status_code == 422


def test_map_endpoint_accepts_model(client):
    r = client.get("/maps/00000001", params={"model": "nearest_road"})
    assert r.status_code == 200 and "Nearest road" in r.text


def test_requests_appear_in_live_log_but_polling_does_not(client):
    client.post("/predict", json={"trip_id": "00000001", "model": "hmm_distance"})
    client.get("/logs")
    client.get("/logs")
    data = client.get("/logs").json()
    paths = [e["path"] for e in data["entries"]]
    assert paths == ["/predict"]
    entry = data["entries"][0]
    assert entry["trip_id"] == "00000001" and entry["model"] == "hmm_distance"
    assert entry["status"] == 200 and "points matched" in entry["summary"]


def test_log_view_page_is_html(client):
    r = client.get("/logs/view")
    assert r.status_code == 200 and "live request log" in r.text


def test_classifier_selection(client):
    body = client.get("/models").json()
    ids = [c["id"] for c in body["classifiers"]]
    assert ids and sum(c["is_default"] for c in body["classifiers"]) == 1
    for cid in ids:
        r = client.post("/predict", json={"trip_id": "00000001", "model": "hmm_classifier",
                                          "classifier": cid}).json()
        assert r["classifier"] == cid and r["emission_source"].startswith(f"classifier:{cid}")


def test_unknown_classifier_is_422(client):
    r = client.post("/predict", json={"trip_id": "00000001", "model": "hmm_classifier",
                                      "classifier": "svm"})
    assert r.status_code == 422 and "unknown classifier" in r.json()["detail"]


def test_classifier_ignored_for_distance_models(client):
    r = client.post("/predict", json={"trip_id": "00000001", "model": "nearest_road",
                                      "classifier": "svm"}).json()
    assert r["classifier"] is None
