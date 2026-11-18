"""K-nearest candidate road arcs per GPS observation (Issue #2).

Uses a Shapely 2 STRtree over arc LineStrings in projected metres (with an exact
vectorized NumPy point-to-segment fallback if Shapely is absent) and exact
point-to-segment perpendicular distances (not distance to nodes).

Output contract (shared with Members 2, 3, and 4):
    trip_id, obs_idx, t, cand_rank, arc_id, dist_m, proj_x, proj_y, frac_along
Observations with no arc within ``radius_m`` get no rows; use
``observations_without_candidates`` to list them.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    import shapely
    from shapely import STRtree

    _HAS_SHAPELY = True
except ImportError:  # pragma: no cover
    shapely = None  # type: ignore
    STRtree = Any  # type: ignore
    _HAS_SHAPELY = False

from data_processing.projection import project_trip

CANDIDATE_COLUMNS = [
    "trip_id",
    "obs_idx",
    "t",
    "cand_rank",
    "arc_id",
    "dist_m",
    "proj_x",
    "proj_y",
    "frac_along",
]


def arc_lines(ptrip: dict) -> np.ndarray:
    """LineString per arc (row order == arcs DataFrame order), metres."""
    nodes, arcs = ptrip["nodes"], ptrip["arcs"]
    xy = nodes[["x", "y"]].to_numpy(dtype=float)
    coords = np.stack(
        [
            xy[arcs["from_node_id"].to_numpy(dtype=int)],
            xy[arcs["to_node_id"].to_numpy(dtype=int)],
        ],
        axis=1,
    )  # (M, 2, 2)
    if _HAS_SHAPELY:
        return shapely.linestrings(coords)
    return coords


def _generate_candidates_numpy(
    ptrip: dict,
    k: int,
    radius_m: float,
) -> pd.DataFrame:
    """Exact vectorized NumPy point-to-segment projection when Shapely is unavailable."""
    track, nodes, arcs = ptrip["track"], ptrip["nodes"], ptrip["arcs"]
    obs_xy = track[["x", "y"]].to_numpy(dtype=float)
    t_arr = track["timestamp"].to_numpy(dtype=float)
    arc_ids = arcs["arc_id"].to_numpy(dtype=int)

    n_xy = nodes[["x", "y"]].to_numpy(dtype=float)
    u_xy = n_xy[arcs["from_node_id"].to_numpy(dtype=int)]  # (M, 2)
    v_xy = n_xy[arcs["to_node_id"].to_numpy(dtype=int)]    # (M, 2)
    seg_vec = v_xy - u_xy                                  # (M, 2)
    seg_len_sq = np.sum(seg_vec * seg_vec, axis=1)         # (M,)

    # Bounding-box pre-filter per segment for fast spatial pruning
    seg_min_x = np.minimum(u_xy[:, 0], v_xy[:, 0]) - radius_m
    seg_max_x = np.maximum(u_xy[:, 0], v_xy[:, 0]) + radius_m
    seg_min_y = np.minimum(u_xy[:, 1], v_xy[:, 1]) - radius_m
    seg_max_y = np.maximum(u_xy[:, 1], v_xy[:, 1]) + radius_m

    rows: list[dict] = []
    trip_id_str = str(ptrip["trip_id"])

    for obs_idx, (ox, oy) in enumerate(obs_xy):
        mask = (
            (ox >= seg_min_x)
            & (ox <= seg_max_x)
            & (oy >= seg_min_y)
            & (oy <= seg_max_y)
        )
        cand_pos = np.flatnonzero(mask)
        if len(cand_pos) == 0:
            continue

        u_sub = u_xy[cand_pos]
        vec_sub = seg_vec[cand_pos]
        len_sq_sub = seg_len_sq[cand_pos]

        rel_x = ox - u_sub[:, 0]
        rel_y = oy - u_sub[:, 1]
        dot = rel_x * vec_sub[:, 0] + rel_y * vec_sub[:, 1]

        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(len_sq_sub > 0.0, np.clip(dot / len_sq_sub, 0.0, 1.0), 0.0)

        px = u_sub[:, 0] + frac * vec_sub[:, 0]
        py = u_sub[:, 1] + frac * vec_sub[:, 1]
        dist = np.hypot(ox - px, oy - py)

        within = dist <= radius_m
        if not np.any(within):
            continue

        cand_pos = cand_pos[within]
        dist = dist[within]
        frac = frac[within]
        px = px[within]
        py = py[within]

        for a_pos, d_val, f_val, proj_x_val, proj_y_val in zip(cand_pos, dist, frac, px, py):
            rows.append(
                {
                    "trip_id": trip_id_str,
                    "obs_idx": int(obs_idx),
                    "t": float(t_arr[obs_idx]),
                    "arc_id": int(arc_ids[a_pos]),
                    "dist_m": float(d_val),
                    "proj_x": float(proj_x_val),
                    "proj_y": float(proj_y_val),
                    "frac_along": float(f_val),
                }
            )

    if not rows:
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)

    df = pd.DataFrame(rows)
    df = df.sort_values(["obs_idx", "dist_m", "arc_id"], kind="mergesort")
    df["cand_rank"] = df.groupby("obs_idx").cumcount()
    df = df[df["cand_rank"] < k]
    return df[CANDIDATE_COLUMNS].reset_index(drop=True)


def generate_candidates(
    trip: dict, k: int = 5, radius_m: float = 50.0, tree: Any | None = None
) -> pd.DataFrame:
    """Candidate arcs for every observation of ``trip``.

    ``trip`` may be raw (it is projected here) or already projected
    (``x``/``y`` columns present). Never reads the reference route.
    """
    if k < 1 or radius_m <= 0:
        raise ValueError("k must be >= 1 and radius_m > 0")
    ptrip = trip if "x" in trip["track"].columns else project_trip(trip)

    if not _HAS_SHAPELY:
        return _generate_candidates_numpy(ptrip, k=k, radius_m=radius_m)

    track, arcs = ptrip["track"], ptrip["arcs"]
    lines = arc_lines(ptrip)
    tree = tree if tree is not None else STRtree(lines)
    points = shapely.points(track[["x", "y"]].to_numpy())

    obs_pos, arc_pos = tree.query(points, predicate="dwithin", distance=radius_m)
    if len(obs_pos) == 0:
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)

    pts, segs = points[obs_pos], lines[arc_pos]
    dist = shapely.distance(pts, segs)
    frac = shapely.line_locate_point(segs, pts, normalized=True)
    frac = np.nan_to_num(frac, nan=0.0)  # zero-length arcs
    proj = shapely.line_interpolate_point(segs, frac, normalized=True)
    proj_xy = shapely.get_coordinates(proj)

    df = pd.DataFrame(
        {
            "trip_id": str(ptrip["trip_id"]),
            "obs_idx": obs_pos.astype(int),
            "t": track["timestamp"].to_numpy()[obs_pos],
            "arc_id": arcs["arc_id"].to_numpy()[arc_pos].astype(int),
            "dist_m": dist,
            "proj_x": proj_xy[:, 0],
            "proj_y": proj_xy[:, 1],
            "frac_along": frac,
        }
    )
    df = df.sort_values(["obs_idx", "dist_m", "arc_id"], kind="mergesort")
    df["cand_rank"] = df.groupby("obs_idx").cumcount()
    df = df[df["cand_rank"] < k]
    return df[CANDIDATE_COLUMNS].reset_index(drop=True)


def observations_without_candidates(candidates: pd.DataFrame, n_obs: int) -> np.ndarray:
    """Sorted obs indices that received no candidate."""
    return np.setdiff1d(np.arange(n_obs), candidates["obs_idx"].unique())


def candidate_lists(
    candidates: pd.DataFrame | list[list[dict]], n_obs: int
) -> tuple[list[list[int]], list[list[float]], list[np.ndarray]]:
    """Per-observation (arc_ids, frac_along, dist_m) in rank order, empty lists for gaps.

    Supports both Member 1's candidate DataFrame and legacy list-of-dicts format.
    """
    if isinstance(candidates, list):
        cand_arcs = [[int(c["arc_id"]) for c in obs] for obs in candidates]
        cand_frac = [[float(c.get("frac_along", c.get("frac", 0.5))) for c in obs] for obs in candidates]
        cand_dist = [np.asarray([float(c["dist_m"]) for c in obs], dtype=float) for obs in candidates]
        return cand_arcs, cand_frac, cand_dist

    arcs: list[list[int]] = [[] for _ in range(n_obs)]
    fracs: list[list[float]] = [[] for _ in range(n_obs)]
    dists: list[list[float]] = [[] for _ in range(n_obs)]
    for row in candidates.sort_values(["obs_idx", "cand_rank"]).itertuples(index=False):
        arcs[row.obs_idx].append(int(row.arc_id))
        fracs[row.obs_idx].append(float(row.frac_along))
        dists[row.obs_idx].append(float(row.dist_m))
    return arcs, fracs, [np.asarray(d, dtype=float) for d in dists]
