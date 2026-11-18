"""Clearly-labelled synthetic candidate-feature generator for Issue #4 training harness development.

WARNING: This module is ONLY for developing and testing the training harness before Member 1's
real feature table (#3) is merged. Never report mock results as experimental findings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import DEFAULT_FEATURE_COLUMNS


def generate_mock_feature_table(
    n_trips: int = 80,
    obs_per_trip: int = 60,
    k_candidates: int = 5,
    candidate_miss_rate: float = 0.02,
    uncertain_rate: float = 0.03,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic candidate-level feature table matching Issue #2 and #3 contracts.

    Each observation has up to `k_candidates` candidate road segments, with at most one positive
    label (`label == 1`), reflecting the natural 1 : K class imbalance of candidate classification.
    All generated rows carry `is_mock = True` and `trip_id` prefixed with `mock_train_`.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for trip_idx in range(n_trips):
        trip_id = f"mock_train_{trip_idx:03d}"
        base_speed = rng.uniform(6.0, 16.0)  # m/s (~22–58 km/h)

        for obs_idx in range(obs_per_trip):
            dt_prev = 1.0 if rng.random() < 0.97 else float(rng.choice([2.0, 3.0, 5.0]))
            dt_next = 1.0 if rng.random() < 0.97 else float(rng.choice([2.0, 3.0, 5.0]))
            gps_speed = max(0.0, float(base_speed + rng.normal(0.0, 1.8)))
            is_miss = rng.random() < candidate_miss_rate
            is_uncertain = rng.random() < uncertain_rate

            # True candidate is usually close in perpendicular distance and aligned in heading,
            # but GPS noise causes ~22% of true arcs to not be rank-0 by raw distance alone.
            true_slot = -1 if is_miss else int(rng.choice(k_candidates, p=[0.74, 0.16, 0.06, 0.03, 0.01][:k_candidates] / np.sum([0.74, 0.16, 0.06, 0.03, 0.01][:k_candidates])))

            raw_dists = []
            cand_specs = []
            for slot in range(k_candidates):
                is_true = slot == true_slot
                if is_true:
                    dist_m = float(np.abs(rng.normal(5.5, 4.5)))
                    heading_diff = float(np.clip(np.abs(rng.normal(12.0, 14.0)), 0.0, 180.0))
                    out_deg = int(rng.choice([1, 2, 3, 4], p=[0.15, 0.50, 0.25, 0.10]))
                    in_deg = int(rng.choice([1, 2, 3, 4], p=[0.15, 0.50, 0.25, 0.10]))
                else:
                    dist_m = float(np.abs(rng.normal(14.0 + 5.0 * slot, 7.0)))
                    heading_diff = float(np.clip(rng.uniform(15.0, 165.0), 0.0, 180.0))
                    out_deg = int(rng.choice([0, 1, 2, 3], p=[0.10, 0.45, 0.35, 0.10]))
                    in_deg = int(rng.choice([0, 1, 2, 3], p=[0.10, 0.45, 0.35, 0.10]))

                arc_len = float(rng.uniform(25.0, 220.0))
                frac_along = float(rng.uniform(0.0, 1.0))
                raw_dists.append(dist_m)
                cand_specs.append(
                    {
                        "arc_id": int(trip_idx * 1000 + (obs_idx * 3 + slot * 17) % 450),
                        "dist_m": dist_m,
                        "arc_length_m": arc_len,
                        "frac_along": frac_along,
                        "heading_diff_deg": heading_diff,
                        "out_degree_v": out_deg,
                        "in_degree_u": in_deg,
                        "label": int(is_true),
                    }
                )

            # Sort candidates by perpendicular distance to assign cand_rank (0 = nearest)
            cand_specs.sort(key=lambda d: d["dist_m"])
            min_dist = max(cand_specs[0]["dist_m"], 1e-3)

            for rank, spec in enumerate(cand_specs):
                dist_m = spec["dist_m"]
                arc_len = spec["arc_length_m"]
                frac = spec["frac_along"]
                h_rad = np.deg2rad(spec["heading_diff_deg"])

                rows.append(
                    {
                        "trip_id": trip_id,
                        "obs_idx": obs_idx,
                        "arc_id": spec["arc_id"],
                        "dist_m": round(dist_m, 4),
                        "cand_rank": rank,
                        "dist_gap_to_nearest_m": round(dist_m - cand_specs[0]["dist_m"], 4),
                        "dist_ratio_to_nearest": round(dist_m / min_dist, 4),
                        "frac_along": round(frac, 4),
                        "dist_to_start_m": round(frac * arc_len, 4),
                        "dist_to_end_m": round((1.0 - frac) * arc_len, 4),
                        "arc_length_m": round(arc_len, 4),
                        "gps_speed_mps": round(gps_speed, 4),
                        "dt_prev_s": dt_prev,
                        "dt_next_s": dt_next,
                        "heading_diff_deg": round(spec["heading_diff_deg"], 4),
                        "cos_heading_diff": round(float(np.cos(h_rad)), 4),
                        "sin_heading_diff": round(float(np.sin(h_rad)), 4),
                        "out_degree_v": spec["out_degree_v"],
                        "in_degree_u": spec["in_degree_u"],
                        "label": spec["label"],
                        "label_uncertain": bool(is_uncertain),
                        "partition": "train",
                        "is_mock": True,
                    }
                )

    df = pd.DataFrame(rows)
    # Verify all standard feature columns exist
    missing = set(DEFAULT_FEATURE_COLUMNS) - set(df.columns)
    assert not missing, f"Mock table missing expected columns: {missing}"
    return df
