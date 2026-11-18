"""API tests (Issue #10) on a tiny synthetic trip folder."""
import pytest
from fastapi.testclient import TestClient

from api.main import app


def write_trip(root, trip_id="00000001"):
    d = root / trip_id
    d.mkdir(parents=True)
    lons = [8.6000 + 0.0001 * i for i in range(10)]
    (d / f"{trip_id}.track").write_text("".join(f"{lo:.4f} 50.10002 {i}\n" for i, lo in enumerate(lons)))
    (d / f"{trip_id}.nodes").write_text("8.6000 50.1\n8.6005 50.1\n8.6010 50.1\n")
    (d / f"{trip_id}.arcs").write_text("0 1\n1 2\n")
    (d / f"{trip_id}.route").write_text("0\n1\n")
    return d


@pytest.fixture()
def client(tmp_path, monkeypatch):
    write_trip(tmp_path)
    monkeypatch.setenv("PATHSNAP_TRIPS_DIR", str(tmp_path))
    monkeypatch.setenv("PATHSNAP_MOCK", "0")
    return TestClient(app)


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["n_trips"] == 1 and body["mock"] is False


def test_trips_lists_folder(client):
    assert client.get("/trips").json() == {"trips": ["00000001"]}


def test_predict_runs_full_pipeline_and_scores_server_side(client):
    r = client.post("/predict", json={"trip_id": "00000001"})
    assert r.status_code == 200
    body = r.json()
    assert body["mock"] is False and body["n_obs"] == 10
    assert body["n_matched"] == 10
    assert set(body["arc_ids"]) <= {0, 1}
    assert body["metrics"]["observation_accuracy"] >= 0.9
    assert body["metrics"]["route_continuity"] in (None, 1.0)
    assert "Predicted route (HMM)" in body["map_html"]
    assert "Reference route" not in body["map_html"]  # ground truth never sent to the UI
    assert "route" not in body  # no raw reference in the payload


def test_unknown_trip_404(client):
    assert client.post("/predict", json={"trip_id": "99999999"}).status_code == 404


@pytest.mark.parametrize("bad", ["../secret", "a/b", ""])
def test_invalid_trip_id_422(client, bad):
    assert client.post("/predict", json={"trip_id": bad}).status_code == 422


def test_mock_mode_is_flagged(client, monkeypatch):
    monkeypatch.setenv("PATHSNAP_MOCK", "1")
    body = client.post("/predict", json={"trip_id": "anything"}).json()
    assert body["mock"] is True and body["emission_source"] == "MOCK"
    assert body["metrics"] is None
    assert "MOCK DATA" in body["map_html"]
