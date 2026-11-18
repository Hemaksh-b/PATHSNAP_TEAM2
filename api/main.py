"""PathSnap FastAPI backend — runs on Laptop 1 (Issue #10).

Start (repo root):
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Every request is logged to the terminal and api/logs/requests.log
(time, client IP, method, path, status, latency) — evaluators check these logs.
"""
from __future__ import annotations

import logging
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from api import settings
from api.evaluation import evaluate
from api.pipeline import GAUSSIAN_SOURCE, mock_prediction, predict_trip
from api.schemas import HealthResponse, PredictRequest, PredictResponse, TripsResponse
from data_processing.loader import list_trips


def _make_logger() -> logging.Logger:
    logger = logging.getLogger("pathsnap.api")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(settings.LOG_DIR / "requests.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    return logger


log = _make_logger()
app = FastAPI(title="PathSnap map-matching API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    client = request.client.host if request.client else "?"
    try:
        response = await call_next(request)
    except Exception:
        log.exception("%s %s %s -> 500", client, request.method, request.url.path)
        raise
    log.info("%s %s %s -> %d (%.0f ms)", client, request.method, request.url.path,
             response.status_code, (time.perf_counter() - start) * 1000)
    return response


def _trip_ids() -> list[str]:
    return [p.name for p in list_trips(settings.trips_dir())]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", mock=settings.mock_mode(),
                          trips_dir=str(settings.trips_dir()), n_trips=len(_trip_ids()),
                          emission_source="MOCK" if settings.mock_mode() else GAUSSIAN_SOURCE)


@app.get("/trips", response_model=TripsResponse)
@app.get("/evaluation/trips", response_model=TripsResponse)
def trips() -> TripsResponse:
    return TripsResponse(trips=_trip_ids())


@app.get("/maps/{trip_id}", response_class=HTMLResponse)
def get_map(trip_id: str):
    if settings.mock_mode():
        return HTMLResponse(mock_prediction(trip_id).map_html)
    if trip_id not in _trip_ids():
        raise HTTPException(status_code=404, detail=f"unknown trip '{trip_id}'")
    try:
        pred = predict_trip(settings.trips_dir() / trip_id)
        return HTMLResponse(pred.map_html)
    except Exception as exc:
        log.exception("map generation failed for trip %s", trip_id)
        raise HTTPException(status_code=500, detail="map generation failed; see server log") from exc


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if settings.mock_mode():
        pred, metrics = mock_prediction(req.trip_id), None
    else:
        if req.trip_id not in _trip_ids():  # also blocks path traversal
            raise HTTPException(status_code=404, detail=f"unknown trip '{req.trip_id}'")
        trip_dir = settings.trips_dir() / req.trip_id
        try:
            pred = predict_trip(trip_dir)
        except Exception as exc:  # report safely, keep details in the server log
            log.exception("prediction failed for trip %s", req.trip_id)
            raise HTTPException(status_code=500, detail="prediction failed; see server log") from exc
        metrics = evaluate(trip_dir, pred.arc_ids, pred.arcs)  # server-side, after prediction

    n_matched = sum(a is not None for a in pred.arc_ids)
    log.info("predict trip=%s obs=%d matched=%d segments=%d latency=%.0f ms mock=%s",
             pred.trip_id, len(pred.arc_ids), n_matched, pred.n_segments, pred.latency_ms,
             settings.mock_mode())
    return PredictResponse(
        trip_id=pred.trip_id, mock=settings.mock_mode(), n_obs=len(pred.arc_ids),
        n_matched=n_matched, arc_ids=list(pred.arc_ids), n_segments=pred.n_segments,
        breaks=list(pred.breaks), emission_source=pred.emission_source,
        latency_ms=pred.latency_ms, metrics=metrics, map_html=pred.map_html,
    )
