"""Fast, vectorized real-dataset candidate & feature extraction on the 80 train and 10 student-test trips.

Extracts the 16 standard candidate features (`DEFAULT_FEATURE_COLUMNS`) and aligns reference `.route`
labels for the `train` (80 trips) and `student_test` (10 trips) partitions in `handouts/Student folder/`.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from data_processing.config import STUDENT_TEST_DIR, TRAIN_DIR
from data_processing.loader import list_trips, load_trip
from data_processing.projection import project_trip


def _vectorized_point_to_segments(
    pts_xy: np.ndarray,
    u_xy: np.ndarray,
    v_xy: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute exact perpendicular distance, frac_along, proj_x, proj_y for paired arrays (N, 2)."""
    seg_vec = v_xy - u_xy
    seg_len_sq = np.einsum("ij,ij->i", seg_vec, seg_vec)
    rel = pts_xy - u_xy
    dot = np.einsum("ij,ij->i", rel, seg_vec)

    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(seg_len_sq > 0.0, np.clip(dot / seg_len_sq, 0.0, 1.0), 0.0)

    proj_x = u_xy[:, 0] + frac * seg_vec[:, 0]
    proj_y = u_xy[:, 1] + frac * seg_vec[:, 1]
    dist_m = np.hypot(pts_xy[:, 0] - proj_x, pts_xy[:, 1] - proj_y)
    return dist_m, frac, proj_x, proj_y


def _build_arc_sample_kdtree(
    u_xy: np.ndarray,
    v_xy: np.ndarray,
    n_samples_per_arc: int = 5,
) -> Tuple[cKDTree, np.ndarray]:
    """Sample points along every road arc and index them in a cKDTree for fast candidate pruning."""
    alphas = np.linspace(0.0, 1.0, n_samples_per_arc)
    # Shape: (M, S, 2)
    pts = u_xy[:, None, :] * (1.0 - alphas[None, :, None]) + v_xy[:, None, :] * alphas[None, :, None]
    flat_pts = pts.reshape(-1, 2)
    arc_indices = np.repeat(np.arange(len(u_xy), dtype=int), n_samples_per_arc)
    return cKDTree(flat_pts), arc_indices


def extract_trip_candidate_features(
    trip_dir: Path,
    partition: str,
    k: int = 5,
    radius_m: float = 50.0,
    uncertain_dist_threshold_m: float = 35.0,
) -> pd.DataFrame:
    """Extract candidate-level feature matrix and aligned reference labels for one real trip."""
    raw_trip = load_trip(trip_dir, include_route=(partition in {"train", "student_test"}))
    ptrip = project_trip(raw_trip)

    track = ptrip["track"]
    nodes = ptrip["nodes"]
    arcs = ptrip["arcs"]
    route_df = ptrip.get("route")

    obs_xy = track[["x", "y"]].to_numpy(dtype=float)
    timestamps = track["timestamp"].to_numpy(dtype=float)
    n_obs = len(obs_xy)

    n_xy = nodes[["x", "y"]].to_numpy(dtype=float)
    u_ids = arcs["from_node_id"].to_numpy(dtype=int)
    v_ids = arcs["to_node_id"].to_numpy(dtype=int)
    arc_ids = arcs["arc_id"].to_numpy(dtype=int)
    arc_lengths = arcs["length_m"].to_numpy(dtype=float)

    u_xy = n_xy[u_ids]
    v_xy = n_xy[v_ids]

    # Arc bearings in radians [-pi, pi]
    arc_dx = v_xy[:, 0] - u_xy[:, 0]
    arc_dy = v_xy[:, 1] - u_xy[:, 1]
    arc_bearing = np.arctan2(arc_dy, arc_dx)

    # Node in-degree and out-degree in the road network
    max_node_id = max(int(nodes["node_id"].max()) + 1, int(max(u_ids.max(), v_ids.max())) + 1)
    out_deg_per_node = np.bincount(u_ids, minlength=max_node_id)
    in_deg_per_node = np.bincount(v_ids, minlength=max_node_id)
    arc_out_deg_v = out_deg_per_node[v_ids]
    arc_in_deg_u = in_deg_per_node[u_ids]

    # Observation-level kinematics (dt_prev, dt_next, speed, GPS movement heading)
    dt_prev = np.ones(n_obs, dtype=float)
    dt_next = np.ones(n_obs, dtype=float)
    if n_obs > 1:
        dt_prev[1:] = np.maximum(timestamps[1:] - timestamps[:-1], 0.0)
        dt_prev[0] = dt_prev[1]
        dt_next[:-1] = np.maximum(timestamps[1:] - timestamps[:-1], 0.0)
        dt_next[-1] = dt_next[-2]

    # Central difference displacement for smooth GPS heading & speed
    disp_x = np.zeros(n_obs, dtype=float)
    disp_y = np.zeros(n_obs, dtype=float)
    step_dt = np.ones(n_obs, dtype=float)
    if n_obs > 2:
        disp_x[1:-1] = obs_xy[2:, 0] - obs_xy[:-2, 0]
        disp_y[1:-1] = obs_xy[2:, 1] - obs_xy[:-2, 1]
        step_dt[1:-1] = np.maximum(timestamps[2:] - timestamps[:-2], 1e-3)
    if n_obs > 1:
        disp_x[0] = obs_xy[1, 0] - obs_xy[0, 0]
        disp_y[0] = obs_xy[1, 1] - obs_xy[0, 1]
        step_dt[0] = max(timestamps[1] - timestamps[0], 1e-3)
        disp_x[-1] = obs_xy[-1, 0] - obs_xy[-2, 0]
        disp_y[-1] = obs_xy[-1, 1] - obs_xy[-2, 1]
        step_dt[-1] = max(timestamps[-1] - timestamps[-2], 1e-3)

    step_dist = np.hypot(disp_x, disp_y)
    gps_speed = step_dist / np.maximum(step_dt, 1e-3)
    gps_heading = np.arctan2(disp_y, disp_x)

    # Query candidate arcs per observation using cKDTree over 5 sample points per arc
    # Query `k * 6` nearest sample points so at least `k` distinct arcs are retrieved
    tree, sample_to_arc = _build_arc_sample_kdtree(u_xy, v_xy, n_samples_per_arc=5)
    q_neighbors = min(len(sample_to_arc), max(k * 7, 35))
    _, nn_sample_idx = tree.query(obs_xy, k=q_neighbors)
    if nn_sample_idx.ndim == 1:
        nn_sample_idx = nn_sample_idx[:, None]
    cand_arc_matrix = sample_to_arc[nn_sample_idx]  # (n_obs, q_neighbors)

    # Build deduplicated (obs_idx, arc_pos) pairs
    obs_repeat = np.repeat(np.arange(n_obs, dtype=int), q_neighbors)
    arc_flat = cand_arc_matrix.ravel()
    pair_keys = obs_repeat.astype(np.int64) * np.int64(len(arc_ids) + 1) + arc_flat.astype(np.int64)
    _, unique_positions = np.unique(pair_keys, return_index=True)
    obs_pairs = obs_repeat[unique_positions]
    arc_pairs = arc_flat[unique_positions]

    # Exact perpendicular point-to-segment distance and fractional projection
    dist_m, frac_along, proj_x, proj_y = _vectorized_point_to_segments(
        obs_xy[obs_pairs],
        u_xy[arc_pairs],
        v_xy[arc_pairs],
    )

    # Filter to radius_m, but guarantee at least the nearest candidate per observation if all > radius_m
    cand_df = pd.DataFrame(
        {
            "obs_idx": obs_pairs,
            "arc_pos": arc_pairs,
            "arc_id": arc_ids[arc_pairs],
            "dist_m": dist_m,
            "frac_along": frac_along,
            "proj_x": proj_x,
            "proj_y": proj_y,
        }
    )
    cand_df = cand_df.sort_values(["obs_idx", "dist_m", "arc_id"], kind="mergesort")
    cand_df["cand_rank"] = cand_df.groupby("obs_idx", sort=False).cumcount()
    cand_df = cand_df[(cand_df["cand_rank"] < k) & ((cand_df["dist_m"] <= radius_m) | (cand_df["cand_rank"] == 0))].copy()
    cand_df["cand_rank"] = cand_df.groupby("obs_idx", sort=False).cumcount()
    cand_df = cand_df.reset_index(drop=True)

    # Compute reference route alignment (nearest arc in .route per observation)
    ref_arc_per_obs = np.full(n_obs, -1, dtype=int)
    ref_dist_per_obs = np.full(n_obs, np.inf, dtype=float)
    if route_df is not None and len(route_df) > 0:
        ref_route_arcs = np.unique(route_df["arc_id"].to_numpy(dtype=int))
        ref_tree, ref_sample_to_pos = _build_arc_sample_kdtree(
            u_xy[ref_route_arcs],
            v_xy[ref_route_arcs],
            n_samples_per_arc=5,
        )
        q_ref = min(len(ref_sample_to_pos), 15)
        _, ref_nn = ref_tree.query(obs_xy, k=q_ref)
        if ref_nn.ndim == 1:
            ref_nn = ref_nn[:, None]
        ref_cand_pos = ref_sample_to_pos[ref_nn]  # (n_obs, q_ref)
        obs_rep_r = np.repeat(np.arange(n_obs, dtype=int), q_ref)
        ref_arc_global = ref_route_arcs[ref_cand_pos.ravel()]
        r_dist, _, _, _ = _vectorized_point_to_segments(
            obs_xy[obs_rep_r],
            u_xy[ref_arc_global],
            v_xy[ref_arc_global],
        )
        r_dist_mat = r_dist.reshape(n_obs, q_ref)
        best_col = np.argmin(r_dist_mat, axis=1)
        row_arange = np.arange(n_obs)
        ref_arc_per_obs = ref_route_arcs[ref_cand_pos[row_arange, best_col]]
        ref_dist_per_obs = r_dist_mat[row_arange, best_col]

    # Assemble candidate features
    o_idx = cand_df["obs_idx"].to_numpy(dtype=int)
    a_pos = cand_df["arc_pos"].to_numpy(dtype=int)
    d_arr = cand_df["dist_m"].to_numpy(dtype=float)
    f_arr = cand_df["frac_along"].to_numpy(dtype=float)
    l_arr = arc_lengths[a_pos]

    nearest_per_obs = cand_df.groupby("obs_idx", sort=False)["dist_m"].transform("min").to_numpy(dtype=float)
    dist_gap = d_arr - nearest_per_obs
    dist_ratio = d_arr / np.maximum(nearest_per_obs, 0.25)

    # Angular difference between GPS movement heading and candidate arc bearing [0, 180]
    raw_angle_diff = gps_heading[o_idx] - arc_bearing[a_pos]
    wrapped_rad = np.arctan2(np.sin(raw_angle_diff), np.cos(raw_angle_diff))
    heading_diff_deg = np.abs(np.degrees(wrapped_rad))
    # For stationary GPS steps (displacement < 0.5 m), set neutral heading alignment
    stationary = step_dist[o_idx] < 0.5
    heading_diff_deg = np.where(stationary, 45.0, heading_diff_deg)
    cos_h = np.where(stationary, 0.0, np.cos(wrapped_rad))
    sin_h = np.where(stationary, 0.0, np.abs(np.sin(wrapped_rad)))

    labels = (cand_df["arc_id"].to_numpy(dtype=int) == ref_arc_per_obs[o_idx]).astype(int)
    uncertain = ref_dist_per_obs[o_idx] > uncertain_dist_threshold_m

    out = pd.DataFrame(
        {
            "trip_id": str(ptrip["trip_id"]),
            "obs_idx": o_idx,
            "arc_id": cand_df["arc_id"].to_numpy(dtype=int),
            "u": u_ids[a_pos],
            "v": v_ids[a_pos],
            "dist_m": np.round(d_arr, 4),
            "cand_rank": cand_df["cand_rank"].to_numpy(dtype=int),
            "dist_gap_to_nearest_m": np.round(dist_gap, 4),
            "dist_ratio_to_nearest": np.round(dist_ratio, 4),
            "frac_along": np.round(f_arr, 4),
            "dist_to_start_m": np.round(f_arr * l_arr, 4),
            "dist_to_end_m": np.round((1.0 - f_arr) * l_arr, 4),
            "arc_length_m": np.round(l_arr, 4),
            "gps_speed_mps": np.round(gps_speed[o_idx], 4),
            "dt_prev_s": np.round(dt_prev[o_idx], 4),
            "dt_next_s": np.round(dt_next[o_idx], 4),
            "heading_diff_deg": np.round(heading_diff_deg, 4),
            "cos_heading_diff": np.round(cos_h, 4),
            "sin_heading_diff": np.round(sin_h, 4),
            "out_degree_v": arc_out_deg_v[a_pos],
            "in_degree_u": arc_in_deg_u[a_pos],
            "label": labels,
            "label_uncertain": uncertain,
            "partition": partition,
            "is_mock": False,
        }
    )
    return out


def build_real_partition_features(
    partition: str,
    partition_dir: Optional[Path] = None,
    k: int = 5,
    radius_m: float = 50.0,
    max_obs_per_trip: Optional[int] = None,
) -> pd.DataFrame:
    """Extract candidate features across all trips in `train` (80 trips) or `student_test` (10 trips)."""
    if partition == "train":
        folder = partition_dir or TRAIN_DIR
    elif partition == "student_test":
        folder = partition_dir or STUDENT_TEST_DIR
    else:
        raise ValueError(f"Permitted partitions are 'train' and 'student_test', got '{partition}'.")

    trip_dirs = list_trips(folder)
    if not trip_dirs:
        raise FileNotFoundError(f"No trips found in {folder}")

    frames: list[pd.DataFrame] = []
    for tdir in trip_dirs:
        t_df = extract_trip_candidate_features(tdir, partition=partition, k=k, radius_m=radius_m)
        if max_obs_per_trip is not None:
            t_df = t_df[t_df["obs_idx"] < max_obs_per_trip].copy()
        frames.append(t_df)

    return pd.concat(frames, axis=0, ignore_index=True)
