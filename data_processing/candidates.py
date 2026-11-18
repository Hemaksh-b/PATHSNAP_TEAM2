"""K-nearest candidate road arcs per GPS observation (Issue #2).

Uses a Shapely 2 STRtree over arc LineStrings in projected metres and exact
point-to-segment distances (not distance to nodes).

Output contract (shared with Members 2 and 3):
    trip_id, obs_idx, t, cand_rank, arc_id, dist_m, proj_x, proj_y, frac_along
Observations with no arc within ``radius_m`` get no rows; use
``observations_without_candidates`` to list them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shapely
from shapely import STRtree

from data_processing.projection import project_trip

CANDIDATE_COLUMNS = [
    "trip_id", "obs_idx", "t", "cand_rank", "arc_id",
    "dist_m", "proj_x", "proj_y", "frac_along",
]


def arc_lines(ptrip: dict) -> np.ndarray:
    """LineString per arc (row order == arcs DataFrame order), metres."""
    nodes, arcs = ptrip["nodes"], ptrip["arcs"]
    xy = nodes[["x", "y"]].to_numpy()
    coords = np.stack([xy[arcs["from_node_id"].to_numpy()],
                       xy[arcs["to_node_id"].to_numpy()]], axis=1)  # (M, 2, 2)
    return shapely.linestrings(coords)


def generate_candidates(
    trip: dict, k: int = 5, radius_m: float = 50.0, tree: STRtree | None = None
) -> pd.DataFrame:
    """Candidate arcs for every observation of ``trip``.

    ``trip`` may be raw (it is projected here) or already projected
    (``x``/``y`` columns present). Never reads the reference route.
    """
    if k < 1 or radius_m <= 0:
        raise ValueError("k must be >= 1 and radius_m > 0")
    ptrip = trip if "x" in trip["track"].columns else project_trip(trip)
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

    df = pd.DataFrame({
        "trip_id": str(ptrip["trip_id"]),
        "obs_idx": obs_pos.astype(int),
        "t": track["timestamp"].to_numpy()[obs_pos],
        "arc_id": arcs["arc_id"].to_numpy()[arc_pos].astype(int),
        "dist_m": dist,
        "proj_x": proj_xy[:, 0],
        "proj_y": proj_xy[:, 1],
        "frac_along": frac,
    })
    df = df.sort_values(["obs_idx", "dist_m", "arc_id"], kind="mergesort")
    df["cand_rank"] = df.groupby("obs_idx").cumcount()
    df = df[df["cand_rank"] < k]
    return df[CANDIDATE_COLUMNS].reset_index(drop=True)


def observations_without_candidates(candidates: pd.DataFrame, n_obs: int) -> np.ndarray:
    """Sorted obs indices that received no candidate."""
    return np.setdiff1d(np.arange(n_obs), candidates["obs_idx"].unique())


def candidate_lists(
    candidates: pd.DataFrame, n_obs: int
) -> tuple[list[list[int]], list[list[float]], list[np.ndarray]]:
    """Per-observation (arc_ids, frac_along, dist_m) in rank order, empty lists for gaps.

    This is the shape ``hmm_viterbi.decode`` / ``make_transition_fn`` expect.
    """
    arcs: list[list[int]] = [[] for _ in range(n_obs)]
    fracs: list[list[float]] = [[] for _ in range(n_obs)]
    dists: list[list[float]] = [[] for _ in range(n_obs)]
    for row in candidates.sort_values(["obs_idx", "cand_rank"]).itertuples(index=False):
        arcs[row.obs_idx].append(int(row.arc_id))
        fracs[row.obs_idx].append(float(row.frac_along))
        dists[row.obs_idx].append(float(row.dist_m))
    return arcs, fracs, [np.asarray(d, dtype=float) for d in dists]
