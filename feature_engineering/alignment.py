"""Ground-truth observation to reference-route alignment (Issue #3).

Assigns each GPS observation in a trip to an arc in its reference .route sequence,
enforcing monotonic (order-preserving) forward progress along the route.
Observations with excessive distance (> 35m) or ambiguous match near junctions
are explicitly tagged with label_uncertain=True.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AlignmentConfig:
    max_dist_threshold_m: float = 35.0   # Beyond this, observation is marked uncertain
    search_window: int = 15              # Maximum forward search window along route sequence
    ambiguous_margin_m: float = 3.0      # Difference between top 2 route candidates under which label is ambiguous
    ambiguous_min_dist_m: float = 5.0    # Only flag ambiguity if distance is at least this far


def align_observations_to_route(
    ptrip: dict,
    config: Optional[AlignmentConfig] = None,
) -> Optional[pd.DataFrame]:
    """Align GPS observations to the reference route monotonically.

    Returns a DataFrame with columns:
        obs_idx, aligned_arc_id, aligned_dist_m, is_uncertain, route_pos
    or None if the trip has no reference route.
    """
    if config is None:
        config = AlignmentConfig()

    route = ptrip.get("route")
    if route is None or len(route) == 0:
        return None

    track = ptrip["track"]
    nodes = ptrip["nodes"]
    arcs = ptrip["arcs"]

    route_arc_ids = route["arc_id"].to_numpy(dtype=int)
    M = len(route_arc_ids)
    N = len(track)

    if M == 0 or N == 0:
        return None

    # Line segment endpoints for each route arc
    xy = nodes[["x", "y"]].to_numpy(dtype=float)
    u_idx = arcs["from_node_id"].to_numpy(dtype=int)[route_arc_ids]
    v_idx = arcs["to_node_id"].to_numpy(dtype=int)[route_arc_ids]
    u_xy = xy[u_idx]
    v_xy = xy[v_idx]
    vec = v_xy - u_xy
    lsq = np.sum(vec * vec, axis=1)

    obs_xy = track[["x", "y"]].to_numpy(dtype=float)

    aligned_arcs = np.zeros(N, dtype=int)
    aligned_dists = np.zeros(N, dtype=float)
    aligned_pos = np.zeros(N, dtype=int)
    is_uncertain = np.zeros(N, dtype=bool)

    curr_j = 0
    w = config.search_window

    for i in range(N):
        pt = obs_xy[i]
        j_min = curr_j
        j_max = min(curr_j + w + 1, M)

        sub_u = u_xy[j_min:j_max]
        sub_v = vec[j_min:j_max]
        sub_lsq = lsq[j_min:j_max]

        rel = pt - sub_u
        dot = rel[:, 0] * sub_v[:, 0] + rel[:, 1] * sub_v[:, 1]
        frac = np.where(sub_lsq > 0.0, np.clip(dot / sub_lsq, 0.0, 1.0), 0.0)
        proj = sub_u + frac[:, None] * sub_v
        dists = np.hypot(pt[0] - proj[:, 0], pt[1] - proj[:, 1])

        best_rel_j = int(np.argmin(dists))
        best_j = j_min + best_rel_j
        best_d = float(dists[best_rel_j])

        # Monotonically advance route pointer
        curr_j = best_j

        aligned_arcs[i] = route_arc_ids[best_j]
        aligned_dists[i] = best_d
        aligned_pos[i] = best_j

        # Uncertainty criteria:
        # 1. Distance exceeds plausible threshold (GPS dropout / detour / off-network)
        if best_d > config.max_dist_threshold_m:
            is_uncertain[i] = True
        # 2. Ambiguity: multiple route arcs in the local window have nearly identical distances
        elif len(dists) > 1:
            sorted_d = np.sort(dists)
            if (
                sorted_d[1] - sorted_d[0] < config.ambiguous_margin_m
                and sorted_d[0] > config.ambiguous_min_dist_m
            ):
                is_uncertain[i] = True

    return pd.DataFrame(
        {
            "obs_idx": np.arange(N, dtype=int),
            "aligned_arc_id": aligned_arcs,
            "aligned_dist_m": aligned_dists,
            "route_pos": aligned_pos,
            "is_uncertain": is_uncertain,
        }
    )


def assign_candidate_labels(
    candidates_df: pd.DataFrame,
    alignment_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Join ground-truth alignment with candidate table to assign binary label and label_uncertain.

    If alignment_df is None, returns candidates_df unchanged.
    """
    if alignment_df is None or candidates_df.empty:
        return candidates_df

    df = candidates_df.copy()
    align_dict = alignment_df.set_index("obs_idx")[["aligned_arc_id", "is_uncertain"]].to_dict("index")

    obs_indices = df["obs_idx"].to_numpy(dtype=int)
    arc_ids = df["arc_id"].to_numpy(dtype=int)

    labels = np.zeros(len(df), dtype=int)
    uncertains = np.zeros(len(df), dtype=bool)

    for idx, (o_idx, a_id) in enumerate(zip(obs_indices, arc_ids)):
        if o_idx in align_dict:
            true_arc = align_dict[o_idx]["aligned_arc_id"]
            unc = align_dict[o_idx]["is_uncertain"]
            labels[idx] = 1 if a_id == true_arc else 0
            uncertains[idx] = unc

    df["label"] = labels
    df["label_uncertain"] = uncertains
    return df
