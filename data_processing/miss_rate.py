"""Candidate miss-rate sweep over K and search radius (Issue #2).

Run from the repo root:
    python -m data_processing.miss_rate

For evaluation only, each observation is given an approximate reference arc:
the geometrically nearest arc *among the trip's reference route arcs*. A miss
is an observation whose reference arc is not in its candidate set. (The
proper order-preserving label alignment is Issue #3.)

Writes supporting_results/candidates/miss_rate.csv and per_trip_timing.csv.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import shapely
from shapely import STRtree

from data_processing.candidates import arc_lines, generate_candidates
from data_processing.config import RESULTS_DIR, STUDENT_TEST_DIR, TRAIN_DIR
from data_processing.loader import list_trips, load_trip
from data_processing.projection import project_trip

K_VALUES = (1, 3, 5, 8, 10)
RADII_M = (25.0, 50.0, 100.0, 200.0)


def approx_reference_arc(ptrip: dict) -> np.ndarray:
    """Nearest reference-route arc per observation (evaluation helper, uses labels)."""
    route_ids = np.unique(ptrip["route"]["arc_id"].to_numpy())
    lines = arc_lines(ptrip)[route_ids]
    points = shapely.points(ptrip["track"][["x", "y"]].to_numpy())
    nearest = STRtree(lines).nearest(points)
    return route_ids[nearest]


def sweep_trip(ptrip: dict, k_values=K_VALUES, radii=RADII_M) -> list[dict]:
    cands = generate_candidates(ptrip, k=max(k_values), radius_m=max(radii))
    ref = approx_reference_arc(ptrip)
    n_obs = len(ptrip["track"])
    rows = []
    for radius in radii:
        within = cands[cands["dist_m"] <= radius]
        for k in k_values:
            top = within[within["cand_rank"] < k]  # rank is by distance, so this is top-k within R
            hit = top["arc_id"].to_numpy() == ref[top["obs_idx"].to_numpy()]
            hit_obs = np.unique(top["obs_idx"].to_numpy()[hit])
            with_cands = top["obs_idx"].nunique()
            rows.append({
                "trip_id": ptrip["trip_id"], "k": k, "radius_m": radius, "n_obs": n_obs,
                "obs_without_candidates": n_obs - with_cands,
                "misses": n_obs - len(hit_obs),
                "mean_candidates": len(top) / n_obs,
            })
    return rows


def main() -> None:
    rows, timing = [], []
    for partition, folder in {"train": TRAIN_DIR, "student_test": STUDENT_TEST_DIR}.items():
        for trip_dir in list_trips(folder):
            ptrip = project_trip(load_trip(trip_dir))
            if ptrip["route"] is None:
                continue
            start = time.perf_counter()
            generate_candidates(ptrip, k=5, radius_m=50.0)
            elapsed = time.perf_counter() - start
            timing.append({"partition": partition, "trip_id": ptrip["trip_id"],
                           "n_obs": len(ptrip["track"]), "n_arcs": len(ptrip["arcs"]),
                           "seconds_k5_r50": elapsed})
            for r in sweep_trip(ptrip):
                rows.append({"partition": partition, **r})
            print(f"[{partition}] {ptrip['trip_id']} done ({elapsed:.2f}s for K=5,R=50)")

    per_trip = pd.DataFrame(rows)
    if per_trip.empty:
        raise SystemExit("no trips found - unzip the dataset into handouts/ first")
    agg = (per_trip.groupby(["partition", "k", "radius_m"])
           [["n_obs", "misses", "obs_without_candidates"]].sum().reset_index())
    agg["miss_rate"] = agg["misses"] / agg["n_obs"]
    agg["no_candidate_rate"] = agg["obs_without_candidates"] / agg["n_obs"]

    out = RESULTS_DIR / "candidates"
    out.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out / "miss_rate.csv", index=False)
    per_trip.to_csv(out / "miss_rate_per_trip.csv", index=False)
    pd.DataFrame(timing).to_csv(out / "per_trip_timing.csv", index=False)
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
