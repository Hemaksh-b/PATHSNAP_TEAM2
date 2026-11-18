"""Server-side scoring against the reference route (Issue #10 / #30).

Runs only AFTER prediction, on Laptop 1. The reference route is read here and
nowhere in the inference pipeline; it is never returned to the UI. Metrics are
the same as in the report (``hmm_viterbi.route_metrics``) and observation
labels use Member 1's order-preserving alignment (#3).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from data_processing.loader import load_trip
from data_processing.projection import project_trip
from feature_engineering.alignment import align_observations_to_route
from hmm_viterbi.route_metrics import all_metrics


def read_reference_route(trip_dir: Path) -> list[int] | None:
    trip_dir = Path(trip_dir)
    path = trip_dir / f"{trip_dir.name}.route"
    if not path.exists():
        return None
    return pd.read_csv(path, sep=r"\s+", header=None, names=["arc_id"])["arc_id"].tolist()


@lru_cache(maxsize=12)
def _reference(trip_dir: str) -> tuple[tuple[int, ...], np.ndarray, np.ndarray] | None:
    reference = read_reference_route(Path(trip_dir))
    if reference is None:
        return None
    trip = {**load_trip(Path(trip_dir), include_route=False),
            "route": pd.DataFrame({"arc_id": reference})}
    align = align_observations_to_route(project_trip(trip)).sort_values("obs_idx")
    aligned = align["aligned_arc_id"].fillna(-1).to_numpy(dtype=int)
    certain = ~align["is_uncertain"].to_numpy(dtype=bool)
    return tuple(reference), aligned, certain


def evaluate(trip_dir: Path, arc_ids: Sequence[int | None], arcs: pd.DataFrame) -> dict | None:
    ref = _reference(str(trip_dir))
    if ref is None:
        return None
    reference, aligned, certain = ref
    return all_metrics(arc_ids, aligned, certain, list(reference), arcs)
