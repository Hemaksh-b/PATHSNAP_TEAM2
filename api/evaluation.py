"""Server-side scoring against the reference route (Issue #10).

Runs only AFTER prediction, on Laptop 1. The reference route is read here and
nowhere in the inference pipeline; it is never returned to the UI.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from data_processing.loader import load_trip
from data_processing.miss_rate import approx_reference_arc
from data_processing.projection import project_trip


def read_reference_route(trip_dir: Path) -> list[int] | None:
    trip_dir = Path(trip_dir)
    path = trip_dir / f"{trip_dir.name}.route"
    if not path.exists():
        return None
    return pd.read_csv(path, sep=r"\s+", header=None, names=["arc_id"])["arc_id"].tolist()


def route_continuity(arc_ids: Sequence, arcs: pd.DataFrame) -> float | None:
    """Share of consecutive (deduplicated) predicted arcs joined head-to-tail."""
    seq = [a for i, a in enumerate(arc_ids) if a is not None and (i == 0 or a != arc_ids[i - 1])]
    seq = [a for i, a in enumerate(seq) if i == 0 or a != seq[i - 1]]
    if len(seq) < 2:
        return None
    to_node = arcs["to_node_id"].to_numpy()
    from_node = arcs["from_node_id"].to_numpy()
    joined = [to_node[a] == from_node[b] for a, b in zip(seq, seq[1:])]
    return float(np.mean(joined))


def evaluate(trip_dir: Path, arc_ids: Sequence, arcs: pd.DataFrame) -> dict | None:
    reference = read_reference_route(trip_dir)
    if reference is None:
        return None
    trip = load_trip(trip_dir, include_route=False)
    trip = {**trip, "route": pd.DataFrame({"arc_id": reference})}
    ref_per_obs = approx_reference_arc(project_trip(trip))
    pred = np.array([-1 if a is None else a for a in arc_ids])
    obs_acc = float(np.mean(pred == ref_per_obs))

    pred_set = {a for a in arc_ids if a is not None}
    ref_set = set(reference)
    tp = len(pred_set & ref_set)
    precision = tp / len(pred_set) if pred_set else None
    recall = tp / len(ref_set) if ref_set else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall else None)
    return {
        "observation_accuracy": obs_acc,
        "route_precision": precision,
        "route_recall": recall,
        "route_f1": f1,
        "route_continuity": route_continuity(arc_ids, arcs),
    }
