"""Evaluation metrics for candidate classification and route-matching quality (Issue #5).

Implements:
- Binary & Macro candidate classification metrics (Macro F1, Binary F1, Precision, Recall, Accuracy)
- Probability calibration quality (Log Loss, ROC-AUC)
- Observation-level Top-1 Accuracy & Candidate Miss Rate
- Route Continuity (fraction of consecutive Top-1 predicted arcs that are identical or connected in the road network)
- Confusion matrix counts (TN, FP, FN, TP)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from data_processing.config import STUDENT_TEST_DIR, TRAIN_DIR
from model_training.schema import LABEL_COLUMN
from model_training.train import evaluate_top1_observation_accuracy


def _load_trip_arcs_map(
    trip_id: str,
    search_dirs: Tuple[Path, ...] = (TRAIN_DIR, STUDENT_TEST_DIR),
) -> Optional[Dict[int, Tuple[int, int]]]:
    """Load `arc_id -> (from_node_id, to_node_id)` from raw `.arcs` file if available on disk."""
    for base_dir in search_dirs:
        arcs_path = base_dir / str(trip_id) / f"{trip_id}.arcs"
        if arcs_path.exists():
            arcs_df = pd.read_csv(
                arcs_path,
                sep=r"\s+",
                header=None,
                names=["u", "v"],
                dtype={"u": int, "v": int},
            )
            return {
                int(idx): (int(row.u), int(row.v))
                for idx, row in arcs_df.iterrows()
            }
    return None


def compute_route_continuity(
    df: pd.DataFrame,
    probs: np.ndarray,
    arcs_lookup_by_trip: Optional[Mapping[str, Mapping[int, Tuple[int, int]]]] = None,
) -> float:
    """Compute Route Continuity across trips for the Top-1 predicted candidate per observation.

    Definition (aligned with Issue #5 & Issue #9):
    For each trip, let `a_0, a_1, ..., a_{T-1}` be the chronological sequence of Top-1 predicted
    candidate road segments (`argmax prob` at each observation `obs_idx`).
    A consecutive pair `(a_{t-1}, a_t)` is **continuous** if:
      1. `a_{t-1} == a_t` (the vehicle remains on the same road arc), OR
      2. `to_node(a_{t-1}) == from_node(a_t)` (the second arc starts where the first arc ends), OR
      3. They share an endpoint node (for undirected/bidirectional junction transitions).
    Returns the overall fraction of continuous consecutive transitions in `[0.0, 1.0]`.
    """
    scored = df[["trip_id", "obs_idx", "arc_id"]].copy()
    scored["prob"] = np.asarray(probs, dtype=float)
    has_uv_cols = "u" in df.columns and "v" in df.columns
    if has_uv_cols:
        scored["u"] = df["u"].to_numpy()
        scored["v"] = df["v"].to_numpy()

    scored["_row_order"] = np.arange(len(scored))
    scored_sorted = scored.sort_values(
        by=["trip_id", "obs_idx", "prob", "_row_order"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    top1 = scored_sorted.drop_duplicates(subset=["trip_id", "obs_idx"], keep="first")
    top1 = top1.sort_values(by=["trip_id", "obs_idx"], kind="mergesort")

    total_transitions = 0
    continuous_transitions = 0

    for trip_id, group in top1.groupby("trip_id", sort=False):
        arc_seq = group["arc_id"].to_numpy(dtype=int)
        if len(arc_seq) < 2:
            continue

        trip_map: Optional[Mapping[int, Tuple[int, int]]] = None
        if arcs_lookup_by_trip is not None and str(trip_id) in arcs_lookup_by_trip:
            trip_map = arcs_lookup_by_trip[str(trip_id)]
        elif has_uv_cols:
            u_arr = group["u"].to_numpy(dtype=int)
            v_arr = group["v"].to_numpy(dtype=int)
            trip_map = {int(a): (int(u), int(v)) for a, u, v in zip(arc_seq, u_arr, v_arr)}
        else:
            trip_map = _load_trip_arcs_map(str(trip_id))

        for idx in range(len(arc_seq) - 1):
            a_prev = int(arc_seq[idx])
            a_curr = int(arc_seq[idx + 1])
            total_transitions += 1

            if a_prev == a_curr:
                continuous_transitions += 1
                continue

            if trip_map is not None and a_prev in trip_map and a_curr in trip_map:
                u1, v1 = trip_map[a_prev]
                u2, v2 = trip_map[a_curr]
                if v1 == u2 or v1 == v2 or u1 == u2:
                    continuous_transitions += 1

    if total_transitions == 0:
        return 0.0
    return float(continuous_transitions / total_transitions)


def compute_all_evaluation_metrics(
    df: pd.DataFrame,
    probs: np.ndarray,
    arcs_lookup_by_trip: Optional[Mapping[str, Mapping[int, Tuple[int, int]]]] = None,
) -> Dict[str, Any]:
    """Compute all candidate-level and observation-level evaluation metrics for Issue #5."""
    y_true = df[LABEL_COLUMN].to_numpy(dtype=int)
    p = np.asarray(probs, dtype=float)
    y_pred = (p >= 0.5).astype(int)
    p_clipped = np.clip(p, 1e-7, 1.0 - 1e-7)

    top1_stats = evaluate_top1_observation_accuracy(df, p)
    continuity = compute_route_continuity(df, p, arcs_lookup_by_trip=arcs_lookup_by_trip)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))

    return {
        "n_trips": int(df["trip_id"].nunique()),
        "n_rows": int(len(df)),
        "n_observations": int(top1_stats["n_observations"]),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "binary_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "log_loss": float(log_loss(y_true, p_clipped, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y_true, p)) if len(np.unique(y_true)) > 1 else float("nan"),
        "top1_accuracy": float(top1_stats["top1_accuracy"]),
        "top1_accuracy_given_hit": float(top1_stats["top1_accuracy_given_hit"]),
        "candidate_miss_rate": float(top1_stats["candidate_miss_rate"]),
        "route_continuity": float(continuity),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }
