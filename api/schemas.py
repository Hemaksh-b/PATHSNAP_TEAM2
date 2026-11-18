"""Request / response models for the PathSnap API (Issue #10)."""
from __future__ import annotations

from pydantic import BaseModel, Field

TRIP_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"


class HealthResponse(BaseModel):
    status: str
    mock: bool
    trips_dir: str
    n_trips: int
    emission_source: str


class TripsResponse(BaseModel):
    trips: list[str]


class PredictRequest(BaseModel):
    trip_id: str = Field(pattern=TRIP_ID_PATTERN)


class Metrics(BaseModel):
    """Computed server-side against the reference route AFTER prediction."""

    observation_accuracy: float | None = Field(
        description="share of GPS points whose predicted arc equals the nearest reference arc")
    route_precision: float | None = None
    route_recall: float | None = None
    route_f1: float | None = None
    route_continuity: float | None = Field(
        default=None,
        description="share of consecutive predicted arcs that are connected in the road graph")


class PredictResponse(BaseModel):
    trip_id: str
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
