"""Route-level evaluation metrics (Issues #7 / #9 / #10).

All functions take predictions plus reference information and are used only
AFTER decoding. Definitions (also in hmm_viterbi/README.md):

* observation accuracy: share of GPS points whose predicted arc equals the
  point's aligned reference arc (order-preserving alignment from #3);
* route continuity: share of consecutive distinct predicted arcs that are
  joined head-to-tail (to_node of one == from_node of the next);
* route precision / recall / F1: set overlap of predicted vs reference arcs.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def _distinct_sequence(arc_ids: Sequence[int | None]) -> list[int]:
    seq: list[int] = []
    for a in arc_ids:
        if a is not None and (not seq or seq[-1] != a):
            seq.append(int(a))
    return seq


def route_continuity(arc_ids: Sequence[int | None], arcs: pd.DataFrame) -> float | None:
    seq = _distinct_sequence(arc_ids)
    if len(seq) < 2:
        return None
    to_node = arcs["to_node_id"].to_numpy()
    from_node = arcs["from_node_id"].to_numpy()
    return float(np.mean([to_node[a] == from_node[b] for a, b in zip(seq, seq[1:])]))


def observation_accuracy(arc_ids: Sequence[int | None], aligned_arc_ids: np.ndarray,
                         mask: np.ndarray | None = None) -> float:
    pred = np.array([-1 if a is None else a for a in arc_ids])
    ok = pred == np.asarray(aligned_arc_ids)
    if mask is not None:
        ok = ok[np.asarray(mask, dtype=bool)]
    return float(ok.mean()) if ok.size else float("nan")


def route_set_scores(arc_ids: Sequence[int | None], reference: Sequence[int]) -> dict:
    pred, ref = set(_distinct_sequence(arc_ids)), set(int(a) for a in reference)
    tp = len(pred & ref)
    precision = tp / len(pred) if pred else None
    recall = tp / len(ref) if ref else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
    return {"route_precision": precision, "route_recall": recall, "route_f1": f1}


def all_metrics(arc_ids: Sequence[int | None], aligned_arc_ids: np.ndarray, certain: np.ndarray,
                reference: Sequence[int], arcs: pd.DataFrame) -> dict:
    return {
        "observation_accuracy": observation_accuracy(arc_ids, aligned_arc_ids),
        "observation_accuracy_certain": observation_accuracy(arc_ids, aligned_arc_ids, certain),
        "route_continuity": route_continuity(arc_ids, arcs),
        **route_set_scores(arc_ids, reference),
    }
