"""Inference pipeline run on Laptop 1 (Issue #10).

load trip (NO reference route) -> project -> candidates (Member 1)
-> emissions -> road-graph transitions + Viterbi (Member 3) -> Folium map.

Emissions: Member 2's classifier probabilities once #6 exports a model and
#3 provides inference features; until then a Gaussian on point-to-arc distance
is used and every response reports ``emission_source`` honestly.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from data_processing.candidates import candidate_lists, generate_candidates
from data_processing.loader import load_trip
from data_processing.projection import project_trip
from hmm_viterbi import ViterbiConfig, decode, gaussian_emission_logp
from hmm_viterbi.transitions import RoadGraph, TransitionConfig, make_transition_fn
from visualisations.route_map import build_route_map

GAUSSIAN_SOURCE = "gaussian_distance (classifier pending #3/#6)"


@dataclass(frozen=True)
class PipelineConfig:
    k: int = 5
    radius_m: float = 100.0
    sigma_m: float = 10.0
    viterbi: ViterbiConfig = field(default_factory=ViterbiConfig)
    transitions: TransitionConfig = field(default_factory=TransitionConfig)


@dataclass(frozen=True)
class Prediction:
    trip_id: str
    arc_ids: tuple
    n_segments: int
    breaks: tuple
    emission_source: str
    latency_ms: float
    map_html: str
    arcs: pd.DataFrame  # road arcs, kept server-side for metric computation


def predict_trip(trip_dir: Path, config: PipelineConfig = PipelineConfig()) -> Prediction:
    start = time.perf_counter()
    trip = load_trip(trip_dir, include_route=False)  # ground truth never enters inference
    ptrip = project_trip(trip)
    n_obs = len(ptrip["track"])

    cands = generate_candidates(ptrip, k=config.k, radius_m=config.radius_m)
    cand_arcs, cand_frac, cand_dist = candidate_lists(cands, n_obs)
    emissions = [gaussian_emission_logp(d, config.sigma_m) for d in cand_dist]

    nodes, arcs, track = ptrip["nodes"], ptrip["arcs"], ptrip["track"]
    road = RoadGraph(nodes[["x", "y"]].to_numpy(), arcs[["from_node_id", "to_node_id"]].to_numpy())
    timestamps = track["timestamp"].to_numpy(dtype=float)
    transition = make_transition_fn(road, cand_arcs, cand_frac, track[["x", "y"]].to_numpy(),
                                    timestamps, config.transitions)
    result = decode(cand_arcs, emissions, transition, timestamps, config.viterbi)

    fmap = build_route_map(trip, "evaluator_test", predicted_arc_ids=result.arc_ids,
                           title=f"Trip {trip['trip_id']} — predicted route")
    return Prediction(
        trip_id=str(trip["trip_id"]),
        arc_ids=result.arc_ids,
        n_segments=len(result.segments),
        breaks=result.breaks,
        emission_source=GAUSSIAN_SOURCE,
        latency_ms=(time.perf_counter() - start) * 1000,
        map_html=fmap.get_root().render(),
        arcs=trip["arcs"],
    )


def mock_prediction(trip_id: str) -> Prediction:
    """SYNTHETIC response for UI development without data (PATHSNAP_MOCK=1)."""
    n = 20
    lon = 8.6 + np.linspace(0, 0.01, n)
    trip = {
        "trip_id": trip_id,
        "track": pd.DataFrame({"lon": lon, "lat": 50.1 + 0 * lon, "timestamp": np.arange(n, dtype=float)}),
        "nodes": pd.DataFrame({"lon": [8.6, 8.61], "lat": [50.1, 50.1], "node_id": [0, 1]}),
        "arcs": pd.DataFrame({"from_node_id": [0], "to_node_id": [1], "arc_id": [0]}),
        "route": None,
    }
    fmap = build_route_map(trip, "evaluator_test", predicted_arc_ids=[0] * n,
                           title="MOCK DATA — not a real prediction")
    return Prediction(trip_id, tuple([0] * n), 1, (), "MOCK", 0.0,
                      fmap.get_root().render(), trip["arcs"])
