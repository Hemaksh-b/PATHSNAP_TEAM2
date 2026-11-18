"""Request / response models for the PathSnap API (Issue #10 / #30)."""
from __future__ import annotations

from pydantic import BaseModel, Field

TRIP_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"
MODEL_ID_PATTERN = r"^[a-z_]{1,32}$"


class HealthResponse(BaseModel):
    status: str
    mock: bool
    trips_dir: str
    n_trips: int
    emission_source: str
    default_model: str
    classifier: str | None


class TripsResponse(BaseModel):
    trips: list[str]


class ModelOptionOut(BaseModel):
    id: str
    label: str
    description: str
    uses_hmm: bool
    uses_classifier: bool
    available: bool


class ClassifierOut(BaseModel):
    id: str
    label: str
    algorithm_family: str
    is_default: bool
    tuned: bool


class ModelsResponse(BaseModel):
    default: str
    classifier: str | None
    models: list[ModelOptionOut]
    classifiers: list[ClassifierOut]


class PredictRequest(BaseModel):
    trip_id: str = Field(pattern=TRIP_ID_PATTERN)
    model: str | None = Field(default=None, pattern=MODEL_ID_PATTERN,
                              description="decoding option from GET /models (default: final model)")
    classifier: str | None = Field(default=None, pattern=MODEL_ID_PATTERN,
                                   description="classifier id from GET /models (default: tuned final model)")


class Metrics(BaseModel):
    """Computed server-side against the reference route AFTER prediction."""

    observation_accuracy: float | None = Field(
        description="share of GPS points whose predicted arc equals their aligned reference arc")
    observation_accuracy_certain: float | None = Field(
        default=None, description="same, on points whose reference label is not flagged uncertain")
    route_precision: float | None = None
    route_recall: float | None = None
    route_f1: float | None = None
    route_continuity: float | None = Field(
        default=None,
        description="share of consecutive predicted arcs that are connected in the road graph")


class PredictResponse(BaseModel):
    trip_id: str
    model: str
    model_label: str
    classifier: str | None
    mock: bool
    n_obs: int
    n_matched: int
    arc_ids: list[int | None]
    n_segments: int
    breaks: list[tuple[int, str]]
    emission_source: str
    latency_ms: float
    metrics: Metrics | None
    map_html: str
