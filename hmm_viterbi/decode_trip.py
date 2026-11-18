"""Decode one real trip end-to-end (Issue #7, phase B).

trip folder (loaded WITHOUT .route) -> UTM -> candidates (#2) -> features (#3)
-> emissions -> road-graph transitions (#8) -> Viterbi.

Emissions:
* ``classifier`` — log of Member 2's exported model probability (#6);
* ``gaussian``   — Gaussian on point-to-arc distance (no classifier needed).
A classifier whose model card says ``is_mock_data`` is never used for
reportable results (see ``run_experiments.py``).

This is the single implementation shared by the experiments (#7/#9) and,
once Member 4 switches to it, the API (#10).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_processing.candidates import candidate_lists, generate_candidates
from data_processing.config import ARTIFACTS_DIR
from data_processing.loader import load_trip
from data_processing.projection import project_trip
from feature_engineering.features import extract_candidate_features
from hmm_viterbi.emissions import classifier_emission_logp, gaussian_emission_logp
from hmm_viterbi.transitions import RoadGraph, TransitionConfig, make_transition_fn
from hmm_viterbi.viterbi import DecodeResult, ViterbiConfig, decode

HMM_CONFIG_FILE = "hmm_config.json"
GAUSSIAN_SOURCE = "gaussian_distance"


@dataclass(frozen=True)
class DecodeSettings:
    k: int = 8
    radius_m: float = 100.0
    sigma_m: float = 10.0  # Gaussian emission only
    viterbi: ViterbiConfig = field(default_factory=ViterbiConfig)
    transitions: TransitionConfig = field(default_factory=TransitionConfig)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "DecodeSettings":
        return cls(k=d["k"], radius_m=d["radius_m"], sigma_m=d["sigma_m"],
                   viterbi=ViterbiConfig(**d["viterbi"]),
                   transitions=TransitionConfig(**d["transitions"]))


def load_settings(artifacts_dir: Path = ARTIFACTS_DIR) -> DecodeSettings:
    path = artifacts_dir / HMM_CONFIG_FILE
    if not path.exists():
        return DecodeSettings()
    return DecodeSettings.from_dict(json.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class Classifier:
    """Exported candidate classifier (Member 2, #6)."""

    pipeline: Any
    feature_columns: list[str]
    card: dict

    @property
    def name(self) -> str:
        return str(self.card.get("model_key", "unknown"))

    @property
    def is_mock(self) -> bool:
        return bool(self.card.get("is_mock_data")) or "MOCK" in str(self.card.get("data_source", ""))

    def probabilities(self, features: pd.DataFrame) -> np.ndarray:
        from hyperparameter_optimisation.inference import predict_candidate_probs

        scores = predict_candidate_probs(features, pipeline=self.pipeline,
                                         feature_columns=self.feature_columns)
        return scores["prob"].to_numpy(dtype=float)


def load_classifier(artifacts_dir: Path = ARTIFACTS_DIR) -> Classifier | None:
    from hyperparameter_optimisation.inference import load_exported_model

    try:
        pipeline, card = load_exported_model(artifacts_dir=artifacts_dir)
    except FileNotFoundError:
        return None
    return Classifier(pipeline, list(card["feature_columns"]), card)


@dataclass(frozen=True)
class PreparedTrip:
    trip: dict  # raw trip, route NOT loaded
    ptrip: dict  # projected
    features: pd.DataFrame  # one row per candidate: ids + the 16 model features
    road: RoadGraph


@dataclass(frozen=True)
class TripDecode:
    trip_id: str
    hmm_arc_ids: tuple[int | None, ...]
    pointwise_arc_ids: tuple[int | None, ...]  # best candidate per point, no sequence model
    result: DecodeResult
    emission_source: str


def prepare_trip(trip_dir: Path, settings: DecodeSettings) -> PreparedTrip:
    trip = load_trip(Path(trip_dir), include_route=False)  # ground truth never enters inference
    ptrip = project_trip(trip)
    cands = generate_candidates(ptrip, k=settings.k, radius_m=settings.radius_m)
    feats = extract_candidate_features(ptrip, cands, include_labels=False)
    feats = feats.sort_values(["obs_idx", "cand_rank"], kind="mergesort").reset_index(drop=True)
    nodes, arcs = ptrip["nodes"], ptrip["arcs"]
    road = RoadGraph(nodes[["x", "y"]].to_numpy(), arcs[["from_node_id", "to_node_id"]].to_numpy())
    return PreparedTrip(trip, ptrip, feats, road)


def _per_observation(feats: pd.DataFrame, n_obs: int, column: str) -> list[np.ndarray]:
    out = [np.empty(0) for _ in range(n_obs)]
    for obs, grp in feats.groupby("obs_idx", sort=True):
        out[int(obs)] = grp[column].to_numpy(dtype=float)
    return out


def decode_prepared(prep: PreparedTrip, settings: DecodeSettings,
                    classifier: Classifier | None = None) -> TripDecode:
    track = prep.ptrip["track"]
    n_obs = len(track)
    feats = prep.features
    if classifier is not None and len(feats):
        feats = feats.assign(score=classifier.probabilities(feats))
        emissions = [classifier_emission_logp(p) for p in _per_observation(feats, n_obs, "score")]
        source = f"classifier:{classifier.name}" + (" (MOCK MODEL)" if classifier.is_mock else "")
    else:
        feats = feats.assign(score=-feats["dist_m"].astype(float))  # nearer = better
        emissions = [gaussian_emission_logp(d, settings.sigma_m)
                     for d in _per_observation(feats, n_obs, "dist_m")]
        source = GAUSSIAN_SOURCE

    cand_arcs, cand_frac, _ = candidate_lists(feats, n_obs)
    timestamps = track["timestamp"].to_numpy(dtype=float)
    transition = make_transition_fn(prep.road, cand_arcs, cand_frac, track[["x", "y"]].to_numpy(),
                                    timestamps, settings.transitions)
    result = decode(cand_arcs, emissions, transition, timestamps, settings.viterbi)

    pointwise: list[int | None] = [None] * n_obs
    best = (feats.sort_values(["score", "cand_rank"], ascending=[False, True], kind="mergesort")
            .drop_duplicates("obs_idx"))
    for obs, arc in zip(best["obs_idx"], best["arc_id"]):
        pointwise[int(obs)] = int(arc)
    return TripDecode(str(prep.trip["trip_id"]), result.arc_ids, tuple(pointwise), result, source)


def decode_trip(trip_dir: Path, settings: DecodeSettings | None = None,
                classifier: Classifier | None = None) -> TripDecode:
    settings = settings or load_settings()
    return decode_prepared(prepare_trip(trip_dir, settings), settings, classifier)
