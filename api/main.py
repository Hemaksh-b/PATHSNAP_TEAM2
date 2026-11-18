"""PathSnap FastAPI backend — runs on Laptop 1 (Issues #10, #30).

Start (repo root):
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Endpoints: GET /health, GET /trips (= /evaluation/trips), GET /models,
POST /predict {trip_id, model}, GET /maps/{trip_id}?model=..., GET /logs (JSON)
and GET /logs/view (live request log page for Laptop 1's screen).

Every request is logged to the terminal, api/logs/requests.log and the live
log page (time, client IP, method, path, status, latency, trip, model).
"""
from __future__ import annotations

import logging
import re
import time

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from api import pipeline, request_log, settings
from api.evaluation import evaluate
from api.schemas import (MODEL_ID_PATTERN, TRIP_ID_PATTERN, ClassifierOut, HealthResponse, ModelOptionOut,
                         ModelsResponse, PredictRequest, PredictResponse, TripsResponse)
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
app = FastAPI(title="PathSnap map-matching API", version="0.3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

QUIET_PATHS = ("/logs",)  # the live log page polls these; do not log the polling itself


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    client = request.client.host if request.client else "?"
    path = request.url.path
    try:
        response = await call_next(request)
        status = response.status_code
    except Exception:
        log.exception("%s %s %s -> 500", client, request.method, path)
        raise
    latency = (time.perf_counter() - start) * 1000
    if not path.startswith(QUIET_PATHS):
        extra = getattr(request.state, "log_extra", {})
        log.info("%s %s %s -> %d (%.0f ms) %s", client, request.method, path, status, latency,
                 " ".join(f"{k}={v}" for k, v in extra.items()))
        request_log.record(client=client, method=request.method, path=path, status=status,
                           latency_ms=round(latency), **extra)
    return response


def _trip_ids() -> list[str]:
    return [p.name for p in list_trips(settings.trips_dir())]


def _trip_dir(trip_id: str):
    if trip_id not in _trip_ids():  # also blocks path traversal
        raise HTTPException(status_code=404, detail=f"unknown trip '{trip_id}'")
    return settings.trips_dir() / trip_id


def _predict(trip_id: str, model: str | None, classifier: str | None = None):
    if settings.mock_mode():
        return pipeline.mock_prediction(trip_id, model), None
    trip_dir = _trip_dir(trip_id)
    try:
        pred = pipeline.predict_trip(trip_dir, model, classifier)
    except ValueError as exc:  # unknown / unavailable model option
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # report safely, keep details in the server log
        log.exception("prediction failed for trip %s model %s", trip_id, model)
        raise HTTPException(status_code=500, detail="prediction failed; see server log") from exc
    return pred, evaluate(trip_dir, pred.arc_ids, pred.arcs)  # server-side, after prediction


@app.on_event("startup")
def announce() -> None:
    log.info("PathSnap API ready - trips: %s (%d) - classifier: %s - default model: %s",
             settings.trips_dir(), len(_trip_ids()), pipeline.classifier_name() or "none",
             pipeline.default_model())
    log.info("Live request log: http://<this-laptop-IP>:8000/logs/view")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> str:
    return ('<h2>PathSnap API (Laptop 1)</h2><ul><li><a href="/logs/view">Live request log</a></li>'
            '<li><a href="/docs">API docs</a></li><li><a href="/health">Health</a></li></ul>')


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    mock = settings.mock_mode()
    name = pipeline.classifier_name()
    source = "MOCK" if mock else (f"classifier:{name}" if name else "gaussian_distance")
    return HealthResponse(status="ok", mock=mock, trips_dir=str(settings.trips_dir()),
                          n_trips=len(_trip_ids()), emission_source=source,
                          default_model=pipeline.default_model(), classifier=name)


@app.get("/trips", response_model=TripsResponse)
@app.get("/evaluation/trips", response_model=TripsResponse)
def trips() -> TripsResponse:
    return TripsResponse(trips=_trip_ids())


@app.get("/models", response_model=ModelsResponse)
def models() -> ModelsResponse:
    return ModelsResponse(
        default=pipeline.default_model(), classifier=pipeline.classifier_name(),
        models=[ModelOptionOut(id=o.id, label=o.label, description=o.description, uses_hmm=o.use_hmm,
                               uses_classifier=o.needs_classifier, available=pipeline.available(o))
                for o in pipeline.MODEL_OPTIONS.values()],
        classifiers=[ClassifierOut(**c) for c in pipeline.classifier_options()])


@app.get("/maps/{trip_id}", response_class=HTMLResponse)
def get_map(request: Request, trip_id: str,
            model: str | None = Query(default=None, pattern=MODEL_ID_PATTERN),
            classifier: str | None = Query(default=None, pattern=MODEL_ID_PATTERN)) -> HTMLResponse:
    if not re.fullmatch(TRIP_ID_PATTERN, trip_id):
        raise HTTPException(status_code=422, detail="invalid trip id")
    pred, _ = _predict(trip_id, model, classifier)
    request.state.log_extra = {"trip_id": trip_id, "model": pred.model, "classifier": pred.classifier}
    return HTMLResponse(pred.map_html)


@app.post("/predict", response_model=PredictResponse)
def predict(request: Request, req: PredictRequest) -> PredictResponse:
    pred, metrics = _predict(req.trip_id, req.model, req.classifier)
    n_matched = sum(a is not None for a in pred.arc_ids)
    acc = metrics.get("observation_accuracy") if metrics else None
    request.state.log_extra = {
        "trip_id": pred.trip_id, "model": pred.model, "classifier": pred.classifier,
        "summary": f"{n_matched}/{len(pred.arc_ids)} points matched"
                   + (f", accuracy {acc:.3f}" if acc is not None else ""),
    }
    return PredictResponse(
        trip_id=pred.trip_id, model=pred.model, model_label=pred.model_label,
        classifier=pred.classifier, mock=settings.mock_mode(),
        n_obs=len(pred.arc_ids), n_matched=n_matched, arc_ids=list(pred.arc_ids),
        n_segments=pred.n_segments, breaks=list(pred.breaks), emission_source=pred.emission_source,
        latency_ms=pred.latency_ms, metrics=metrics, map_html=pred.map_html,
    )


@app.get("/logs")
def logs(limit: int = Query(default=200, ge=1, le=request_log.MAX_ENTRIES)) -> dict:
    return request_log.recent(limit)


@app.get("/logs/view", response_class=HTMLResponse)
def logs_view() -> str:
    return request_log.LOG_PAGE
