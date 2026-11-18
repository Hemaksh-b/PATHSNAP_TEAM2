"""Inference pipeline run on Laptop 1 (Issue #10 / #30).

Uses Member 3's shared ``hmm_viterbi.decode_trip`` (trip loaded WITHOUT .route
-> candidates -> features -> emissions -> road-graph HMM / Viterbi) and offers
selectable decoding options and classifiers so the evaluator can compare live:

    hmm_classifier   classifier probabilities + HMM/Viterbi (final model)
    classifier_only  classifier probabilities, best candidate per point, no HMM
    hmm_distance     Gaussian point-to-arc distance + HMM/Viterbi
    nearest_road     nearest candidate per point, no HMM

Classifiers: the tuned final model in ``artifacts/`` (default) plus any extra
model exported to ``artifacts/classifiers/<key>/`` (model_training.export_classifier).
Model cards marked ``is_mock_data`` are never served.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from data_processing.config import ARTIFACTS_DIR
from hmm_viterbi.decode_trip import (Classifier, PreparedTrip, decode_prepared, load_classifier,
                                     load_settings, prepare_trip)
from visualisations.route_map import build_route_map

CLASSIFIERS_DIR = ARTIFACTS_DIR / "classifiers"


@dataclass(frozen=True)
class ModelOption:
    id: str
    label: str
    description: str
    needs_classifier: bool
    use_hmm: bool


MODEL_OPTIONS: dict[str, ModelOption] = {o.id: o for o in (
    ModelOption("hmm_classifier", "Classifier + HMM (final model)",
                "Classifier probabilities as emissions, road-graph HMM + Viterbi.", True, True),
    ModelOption("classifier_only", "Classifier only",
                "Highest classifier probability per GPS point, no sequence model.", True, False),
    ModelOption("hmm_distance", "Distance + HMM",
                "Gaussian point-to-road distance as emissions, road-graph HMM + Viterbi.", False, True),
    ModelOption("nearest_road", "Nearest road",
                "Nearest candidate road per GPS point, no model, no HMM (baseline).", False, False),
)}
DEFAULT_MODEL = "hmm_classifier"
FALLBACK_MODEL = "hmm_distance"


@dataclass(frozen=True)
class Prediction:
    trip_id: str
    model: str
    model_label: str
    classifier: str | None
    arc_ids: tuple[int | None, ...]
    n_segments: int
    breaks: tuple[tuple[int, str], ...]
    emission_source: str
    latency_ms: float
    map_html: str
    arcs: pd.DataFrame  # road arcs, kept server-side for metric computation


def _load_real(directory: Path) -> Classifier | None:
    clf = load_classifier(directory)
    return None if clf is None or clf.is_mock else clf


@lru_cache(maxsize=1)
def _classifiers() -> dict[str, tuple[Classifier, bool]]:
    """key -> (classifier, is_tuned_default). The final model in artifacts/ comes first."""
    found: dict[str, tuple[Classifier, bool]] = {}
    main = _load_real(ARTIFACTS_DIR)
    if main is not None:
        found[main.name] = (main, True)
    if CLASSIFIERS_DIR.exists():
        for sub in sorted(p for p in CLASSIFIERS_DIR.iterdir() if p.is_dir()):
            clf = _load_real(sub)
            if clf is not None and clf.name not in found:
                found[clf.name] = (clf, False)
    return found


def get_classifier(key: str | None = None) -> Classifier | None:
    """Requested classifier (default: the tuned final model); None if unavailable."""
    found = _classifiers()
    if not found:
        return None
    if key is None:
        return next(iter(found.values()))[0]
    entry = found.get(key)
    return entry[0] if entry else None


def classifier_options() -> list[dict]:
    return [{"id": key, "label": clf.card.get("model_name", key), "is_default": tuned,
             "tuned": tuned, "algorithm_family": clf.card.get("algorithm_family", "")}
            for key, (clf, tuned) in _classifiers().items()]


def classifier_name() -> str | None:
    clf = get_classifier()
    return clf.name if clf else None


def available(option: ModelOption) -> bool:
    return not option.needs_classifier or get_classifier() is not None


def default_model() -> str:
    return DEFAULT_MODEL if available(MODEL_OPTIONS[DEFAULT_MODEL]) else FALLBACK_MODEL


def resolve_model(model: str | None) -> ModelOption:
    option = MODEL_OPTIONS.get(model or default_model())
    if option is None:
        raise ValueError(f"unknown model '{model}'; choose one of {sorted(MODEL_OPTIONS)}")
    if not available(option):
        raise ValueError(f"model '{option.id}' needs an exported (non-mock) classifier in artifacts/")
    return option


def resolve_classifier(option: ModelOption, key: str | None) -> Classifier | None:
    if not option.needs_classifier:
        return None
    clf = get_classifier(key)
    if clf is None:
        known = [c["id"] for c in classifier_options()]
        raise ValueError(f"unknown classifier '{key}'; available: {known}")
    return clf


@lru_cache(maxsize=12)
def _prepared(trip_dir: str) -> PreparedTrip:
    return prepare_trip(Path(trip_dir), load_settings())


@lru_cache(maxsize=64)
def predict_trip(trip_dir: Path, model: str | None = None, classifier: str | None = None) -> Prediction:
    """Decode one trip with the chosen option and classifier (cached per combination)."""
    option = resolve_model(model)
    clf = resolve_classifier(option, classifier)
    start = time.perf_counter()
    prep = _prepared(str(trip_dir))
    out = decode_prepared(prep, load_settings(), clf)
    arc_ids = out.hmm_arc_ids if option.use_hmm else out.pointwise_arc_ids
    n_segments = len(out.result.segments) if option.use_hmm else 1
    label = option.label + (f" · {clf.card.get('model_name', clf.name)}" if clf else "")
    fmap = build_route_map(prep.trip, "evaluator_test", predicted_arc_ids=arc_ids,
                           title=f"Trip {out.trip_id} — {label}")
    return Prediction(
        trip_id=out.trip_id, model=option.id, model_label=option.label,
        classifier=clf.name if clf else None, arc_ids=arc_ids, n_segments=n_segments,
        breaks=out.result.breaks if option.use_hmm else (), emission_source=out.emission_source,
        latency_ms=(time.perf_counter() - start) * 1000, map_html=fmap.get_root().render(),
        arcs=prep.trip["arcs"],
    )


def mock_prediction(trip_id: str, model: str | None = None) -> Prediction:
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
    option = MODEL_OPTIONS.get(model or DEFAULT_MODEL, MODEL_OPTIONS[DEFAULT_MODEL])
    return Prediction(trip_id, option.id, option.label, None, tuple([0] * n), 1, (), "MOCK", 0.0,
                      fmap.get_root().render(), trip["arcs"])
