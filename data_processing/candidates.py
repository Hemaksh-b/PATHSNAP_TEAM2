"""Candidate road-segment generation for each GPS observation.

Stub implementation — returns the K nearest arc midpoints using Euclidean
distance in projected (x, y) coordinates. Member 1's full implementation
(Issue #2/#3) replaces this once merged.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def generate_candidates(
    ptrip: dict,
    k: int = 5,
    radius_m: float = 100.0,
) -> List[List[dict]]:
    """Return per-observation candidate lists.

    Each candidate is a dict with keys:
        arc_id, frac (0-1 position along arc), dist_m (metres to GPS point).

    If fewer than ``k`` arcs are within ``radius_m``, all within-radius arcs
    are returned (possibly an empty list for a GPS observation with no nearby arc).
    """
    track = ptrip["track"]
    nodes = ptrip["nodes"]
    arcs = ptrip["arcs"]

    # Compute arc midpoints in projected space
    n_x = nodes["x"].to_numpy()
    n_y = nodes["y"].to_numpy()

    from_ids = arcs["from_node_id"].to_numpy()
    to_ids = arcs["to_node_id"].to_numpy()

    mid_x = (n_x[from_ids] + n_x[to_ids]) / 2
    mid_y = (n_y[from_ids] + n_y[to_ids]) / 2

    obs_x = track["x"].to_numpy()
    obs_y = track["y"].to_numpy()
    arc_ids = arcs["arc_id"].to_numpy()

    result = []
    for ox, oy in zip(obs_x, obs_y):
        dists = np.hypot(ox - mid_x, oy - mid_y)
        order = np.argsort(dists)
        cands = []
        for idx in order:
            if dists[idx] > radius_m:
                break
            cands.append({"arc_id": int(arc_ids[idx]), "frac": 0.5, "dist_m": float(dists[idx])})
            if len(cands) == k:
                break
        # Fallback: always include the nearest arc even if beyond radius
        if not cands:
            idx = int(order[0])
            cands.append({"arc_id": int(arc_ids[idx]), "frac": 0.5, "dist_m": float(dists[idx])})
        result.append(cands)
    return result


def candidate_lists(
    candidates: List[List[dict]],
    n_obs: int,
) -> Tuple[List[List[int]], List[List[float]], List[List[float]]]:
    """Split per-observation candidate dicts into three parallel lists.

    Returns
    -------
    cand_arcs  : [[arc_id, ...], ...]  — one inner list per observation
    cand_frac  : [[frac,   ...], ...]
    cand_dist  : [[dist_m, ...], ...]
    """
    cand_arcs = [[c["arc_id"] for c in obs] for obs in candidates]
    cand_frac = [[c["frac"]   for c in obs] for obs in candidates]
    cand_dist = [[c["dist_m"] for c in obs] for obs in candidates]
    return cand_arcs, cand_frac, cand_dist
