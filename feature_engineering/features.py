"""Spatial, geometric, kinematic, and topological feature engineering for candidate road arcs (Issue #3).

Computes candidate-level features compatible with Member 2 and Member 4:
    dist_m, cand_rank, dist_gap_to_nearest_m, dist_ratio_to_nearest,
    frac_along, dist_to_start_m, dist_to_end_m, arc_length_m,
    gps_speed_mps, dt_prev_s, dt_next_s,
    heading_diff_deg, cos_heading_diff, sin_heading_diff,
    out_degree_v, in_degree_u

All features are strictly computable without ground truth labels.
When labels are requested for training/student_test, alignment.py derives
binary `label` and `label_uncertain`.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from data_processing.candidates import generate_candidates
from feature_engineering.alignment import (
    align_observations_to_route,
    assign_candidate_labels,
)

FEATURE_COLUMNS: List[str] = [
    "dist_m",
    "cand_rank",
    "dist_gap_to_nearest_m",
    "dist_ratio_to_nearest",
    "frac_along",
    "dist_to_start_m",
    "dist_to_end_m",
    "arc_length_m",
    "gps_speed_mps",
    "dt_prev_s",
    "dt_next_s",
    "heading_diff_deg",
    "cos_heading_diff",
    "sin_heading_diff",
    "out_degree_v",
    "in_degree_u",
]


def compute_kinematics_and_bearings(
    track: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute instantaneous speed (m/s), dt_prev (s), dt_next (s), and compass bearing (deg) per observation."""
    N = len(track)
    x = track["x"].to_numpy(dtype=float)
    y = track["y"].to_numpy(dtype=float)
    t = track["timestamp"].to_numpy(dtype=float)

    dx_gps = np.zeros(N, dtype=float)
    dy_gps = np.zeros(N, dtype=float)
    dt_gps = np.ones(N, dtype=float)

    dt_prev = np.zeros(N, dtype=float)
    dt_next = np.zeros(N, dtype=float)

    if N > 1:
        # Central difference for interior points
        dx_gps[1:-1] = x[2:] - x[:-2]
        dy_gps[1:-1] = y[2:] - y[:-2]
        dt_gps[1:-1] = np.maximum(t[2:] - t[:-2], 1e-3)

        # Boundary endpoints
        dx_gps[0] = x[1] - x[0]
        dy_gps[0] = y[1] - y[0]
        dt_gps[0] = max(t[1] - t[0], 1e-3)

        dx_gps[-1] = x[-1] - x[-2]
        dy_gps[-1] = y[-1] - y[-2]
        dt_gps[-1] = max(t[-1] - t[-2], 1e-3)

        # Immediate time intervals
        diff_t = t[1:] - t[:-1]
        dt_prev[1:] = diff_t
        dt_prev[0] = diff_t[0] if len(diff_t) else 1.0

        dt_next[:-1] = diff_t
        dt_next[-1] = diff_t[-1] if len(diff_t) else 1.0
    else:
        dt_prev[0] = 1.0
        dt_next[0] = 1.0

    step_dist = np.hypot(dx_gps, dy_gps)
    speed_mps = step_dist / np.maximum(dt_gps, 1e-3)

    # Compass bearing: 0=North, 90=East
    bearing_deg = np.degrees(np.arctan2(dx_gps, dy_gps)) % 360.0

    return speed_mps, dt_prev, dt_next, bearing_deg


def compute_node_degrees(
    nodes: pd.DataFrame,
    arcs: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute in-degree and out-degree vectors indexed by node_id."""
    node_count = len(nodes)
    u = arcs["from_node_id"].to_numpy(dtype=int)
    v = arcs["to_node_id"].to_numpy(dtype=int)

    in_deg = np.bincount(v, minlength=node_count)
    out_deg = np.bincount(u, minlength=node_count)
    return in_deg, out_deg


def compute_arc_bearings_and_lengths(
    nodes: pd.DataFrame,
    arcs: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute directional bearing (deg) and length (m) per road arc."""
    xy = nodes[["x", "y"]].to_numpy(dtype=float)
    u = arcs["from_node_id"].to_numpy(dtype=int)
    v = arcs["to_node_id"].to_numpy(dtype=int)

    u_xy = xy[u]
    v_xy = xy[v]
    dx = v_xy[:, 0] - u_xy[:, 0]
    dy = v_xy[:, 1] - u_xy[:, 1]

    bearing = np.degrees(np.arctan2(dx, dy)) % 360.0
    length = np.hypot(dx, dy)
    return bearing, length


def extract_candidate_features(
    ptrip: dict,
    candidates_df: Optional[pd.DataFrame] = None,
    k: int = 5,
    radius_m: float = 100.0,
    include_labels: bool = True,
) -> pd.DataFrame:
    """Extract candidate-level features for a projected trip.

    Parameters:
        ptrip: Projected trip dictionary (track, nodes, arcs, route, projection_crs).
        candidates_df: Precomputed candidate table. If None, generated automatically.
        k: Maximum candidate count per observation.
        radius_m: Search radius in metres.
        include_labels: Whether to perform ground-truth alignment and attach label columns.
                        If False or if ptrip has no route, label columns are NOT attached.

    Returns:
        DataFrame containing identification columns, candidate features, and optionally labels:
            trip_id, obs_idx, arc_id, <16 feature columns...>, [label, label_uncertain]
    """
    if candidates_df is None:
        candidates_df = generate_candidates(ptrip, k=k, radius_m=radius_m)

    if candidates_df.empty:
        cols = ["trip_id", "obs_idx", "arc_id"] + FEATURE_COLUMNS
        if include_labels and ptrip.get("route") is not None:
            cols += ["label", "label_uncertain"]
        return pd.DataFrame(columns=cols)

    track = ptrip["track"]
    nodes = ptrip["nodes"]
    arcs = ptrip["arcs"]

    # 1. Kinematics and GPS bearings
    speed_mps, dt_prev, dt_next, gps_bearing = compute_kinematics_and_bearings(track)

    # 2. Road network topological degrees
    in_degrees, out_degrees = compute_node_degrees(nodes, arcs)

    # 3. Arc geometric bearings and lengths
    arc_bearing, arc_length = compute_arc_bearings_and_lengths(nodes, arcs)

    # 4. Map candidate indices
    c_obs = candidates_df["obs_idx"].to_numpy(dtype=int)
    c_arc = candidates_df["arc_id"].to_numpy(dtype=int)

    c_speed = speed_mps[c_obs]
    c_dt_p = dt_prev[c_obs]
    c_dt_n = dt_next[c_obs]
    c_gps_b = gps_bearing[c_obs]

    c_arc_b = arc_bearing[c_arc]
    c_arc_l = arc_length[c_arc]

    # Heading difference: angular distance between GPS motion vector and arc bearing
    h_diff = np.abs(c_gps_b - c_arc_b) % 360.0
    h_diff = np.minimum(h_diff, 360.0 - h_diff)

    # When vehicle is stopped/crawling (< 0.5 m/s), GPS heading is noisy noise: neutral difference
    stationary = c_speed < 0.5
    h_diff[stationary] = 0.0

    cos_h = np.cos(np.radians(h_diff))
    sin_h = np.sin(np.radians(h_diff))

    u_nodes = arcs["from_node_id"].to_numpy(dtype=int)[c_arc]
    v_nodes = arcs["to_node_id"].to_numpy(dtype=int)[c_arc]
    c_in_deg = in_degrees[u_nodes]
    c_out_deg = out_degrees[v_nodes]

    frac = candidates_df["frac_along"].to_numpy(dtype=float)
    dist_m = candidates_df["dist_m"].to_numpy(dtype=float)
    dist_to_start = frac * c_arc_l
    dist_to_end = (1.0 - frac) * c_arc_l

    # Proximity gap and ratio to nearest candidate for the same observation
    min_dist_per_obs = candidates_df.groupby("obs_idx")["dist_m"].transform("min").to_numpy(dtype=float)
    dist_gap = dist_m - min_dist_per_obs
    dist_ratio = (dist_m + 1.0) / (min_dist_per_obs + 1.0)

    out_df = pd.DataFrame(
        {
            "trip_id": candidates_df["trip_id"].astype(str),
            "obs_idx": c_obs,
            "arc_id": c_arc,
            # Spatial Proximity Features
            "dist_m": dist_m,
            "cand_rank": candidates_df["cand_rank"].to_numpy(dtype=int),
            "dist_gap_to_nearest_m": dist_gap,
            "dist_ratio_to_nearest": dist_ratio,
            # Arc Geometry Features
            "frac_along": frac,
            "dist_to_start_m": dist_to_start,
            "dist_to_end_m": dist_to_end,
            "arc_length_m": c_arc_l,
            # Kinematic & Temporal Features
            "gps_speed_mps": c_speed,
            "dt_prev_s": c_dt_p,
            "dt_next_s": c_dt_n,
            # Heading Alignment Features
            "heading_diff_deg": h_diff,
            "cos_heading_diff": cos_h,
            "sin_heading_diff": sin_h,
            # Topological Network Features
            "out_degree_v": c_out_deg,
            "in_degree_u": c_in_deg,
        }
    )

    # 5. Attach ground truth labels if requested and route exists
    if include_labels and ptrip.get("route") is not None:
        alignment_df = align_observations_to_route(ptrip)
        out_df = assign_candidate_labels(out_df, alignment_df)

    return out_df
