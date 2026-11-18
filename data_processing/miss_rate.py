"""Approximate reference-arc assignment per GPS observation.

For each GPS observation in the projected trip, finds the nearest arc in the
reference route and returns its arc_id. Used by the server-side evaluation
module (api/evaluation.py) to compute observation-level accuracy.

Stub implementation — will be replaced by Member 1's full spatial-join once
Issue #3 is merged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def approx_reference_arc(ptrip: dict) -> np.ndarray:
    """Return per-observation reference arc IDs (shape: [n_obs]).

    Parameters
    ----------
    ptrip:
        Projected trip dict (from data_processing.projection.project_trip).
        Must contain keys: 'track' (with 'x','y'), 'nodes' (with 'x','y'),
        'arcs' (with 'from_node_id', 'to_node_id', 'arc_id'), 'route'.

    Returns
    -------
    np.ndarray of int, shape (n_obs,)
        Each entry is the arc_id of the nearest reference-route arc to the
        corresponding GPS observation.  -1 when the reference route is empty.
    """
    track = ptrip["track"]
    nodes = ptrip["nodes"]
    arcs = ptrip["arcs"]
    route = ptrip.get("route")

    n_obs = len(track)

    if route is None or len(route) == 0:
        return np.full(n_obs, -1, dtype=int)

    ref_arc_ids = route["arc_id"].to_numpy()

    n_x = nodes["x"].to_numpy()
    n_y = nodes["y"].to_numpy()
    from_ids = arcs["from_node_id"].to_numpy()
    to_ids = arcs["to_node_id"].to_numpy()
    arc_id_col = arcs["arc_id"].to_numpy()

    # Build midpoints only for reference-route arcs
    ref_mask = np.isin(arc_id_col, ref_arc_ids)
    ref_arcs = arc_id_col[ref_mask]
    mid_x = (n_x[from_ids[ref_mask]] + n_x[to_ids[ref_mask]]) / 2
    mid_y = (n_y[from_ids[ref_mask]] + n_y[to_ids[ref_mask]]) / 2

    obs_x = track["x"].to_numpy()
    obs_y = track["y"].to_numpy()

    result = np.empty(n_obs, dtype=int)
    for i, (ox, oy) in enumerate(zip(obs_x, obs_y)):
        dists = np.hypot(ox - mid_x, oy - mid_y)
        result[i] = int(ref_arcs[np.argmin(dists)])
    return result
