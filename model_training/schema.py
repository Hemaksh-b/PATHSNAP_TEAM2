"""Feature schema, validation, and data-leakage guards for candidate classification (Issue #4).

Integrates with Member 1's Issue #2 and Issue #3 output schema:
    trip_id, obs_idx, arc_id, <candidate features...>, label, label_uncertain
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from data_processing.config import EVALUATOR_TEST_DIR

ID_COLUMNS: List[str] = ["trip_id", "obs_idx", "arc_id"]
LABEL_COLUMN: str = "label"
UNCERTAIN_COLUMN: str = "label_uncertain"

# Non-feature metadata columns that must never be passed into model X matrices
NON_FEATURE_COLUMNS = {
    "trip_id",
    "obs_idx",
    "arc_id",
    "t",
    "timestamp",
    "lon",
    "lat",
    "x",
    "y",
    "proj_x",
    "proj_y",
    "u",
    "v",
    "from_node_id",
    "to_node_id",
    "label",
    "label_uncertain",
    "partition",
    "is_mock",
    "route",
    "ref_arc_id",
}

# Standard candidate features specified in Issue #2 and Issue #3
DEFAULT_FEATURE_COLUMNS: List[str] = [
    "dist_m",                  # Perpendicular point-to-segment distance (m)
    "cand_rank",               # Candidate rank for this observation (0 = nearest)
    "dist_gap_to_nearest_m",   # Distance minus nearest candidate distance (m)
    "dist_ratio_to_nearest",   # Ratio of distance to nearest candidate distance
    "frac_along",              # Fractional projection position along the arc [0, 1]
    "dist_to_start_m",         # Distance along arc from start node u (m)
    "dist_to_end_m",           # Distance along arc to end node v (m)
    "arc_length_m",            # Total arc length (m)
    "gps_speed_mps",           # Estimated instantaneous GPS speed (m/s)
    "dt_prev_s",               # Elapsed time since previous GPS observation (s)
    "dt_next_s",               # Elapsed time to next GPS observation (s)
    "heading_diff_deg",        # Absolute difference between GPS heading and arc bearing [0, 180]
    "cos_heading_diff",        # cos(heading_diff) in [-1, 1]
    "sin_heading_diff",        # sin(heading_diff) in [0, 1]
    "out_degree_v",            # Out-degree of candidate arc's target node v
    "in_degree_u",             # In-degree of candidate arc's source node u
]


def get_evaluator_trip_ids(evaluator_dir: Path = EVALUATOR_TEST_DIR) -> set[str]:
    """Return the set of evaluator-testing trip IDs to enforce strict leakage prevention."""
    if not evaluator_dir.exists():
        return set()
    trip_ids: set[str] = set()
    for item in evaluator_dir.iterdir():
        if item.is_dir():
            trip_ids.add(item.name)
        elif item.suffix in {".track", ".nodes", ".arcs", ".route"}:
            trip_ids.add(item.stem)
    return trip_ids


def assert_no_evaluator_leakage(
    df: pd.DataFrame,
    source_path: Optional[Path] = None,
    evaluator_dir: Path = EVALUATOR_TEST_DIR,
) -> None:
    """Ensure no evaluator-testing trips or files are used during training or CV."""
    if source_path is not None:
        resolved = str(source_path.resolve()).lower()
        if "evaluator_test" in resolved or "evaluator-test" in resolved:
            raise ValueError(
                f"Data leakage violation: refused to load evaluator-testing file '{source_path}'."
            )

    if "partition" in df.columns:
        bad_parts = set(df["partition"].astype(str).str.lower().unique()) & {
            "evaluator",
            "evaluator_test",
            "evaluator-test",
            "eval_test",
        }
        if bad_parts:
            raise ValueError(
                f"Data leakage violation: dataframe contains forbidden partition(s): {bad_parts}."
            )

    eval_trips = get_evaluator_trip_ids(evaluator_dir)
    if eval_trips and "trip_id" in df.columns:
        present = set(df["trip_id"].astype(str).unique()) & eval_trips
        if present:
            raise ValueError(
                f"Data leakage violation: evaluator trip IDs found in training table: {sorted(present)}"
            )


def resolve_feature_columns(
    df: pd.DataFrame,
    explicit_features: Optional[Iterable[str]] = None,
) -> List[str]:
    """Determine the numeric candidate feature columns to use from a feature table."""
    if explicit_features is not None:
        features = list(explicit_features)
        missing = [c for c in features if c not in df.columns]
        if missing:
            raise KeyError(f"Requested feature columns missing from dataframe: {missing}")
        return features

    # Prefer the standard Issue #3 feature list if all are present; otherwise auto-detect numeric columns
    present_defaults = [c for c in DEFAULT_FEATURE_COLUMNS if c in df.columns]
    extra_numeric = [
        c
        for c in df.select_dtypes(include=[np.number, "bool"]).columns
        if c not in NON_FEATURE_COLUMNS and c not in present_defaults
    ]
    feature_cols = present_defaults + extra_numeric
    if not feature_cols:
        raise ValueError(
            "No valid numeric feature columns found in dataframe. "
            f"Columns present: {list(df.columns)}"
        )
    return feature_cols


def validate_and_prepare_features(
    df: pd.DataFrame,
    feature_columns: Optional[Iterable[str]] = None,
    drop_uncertain_labels: bool = True,
    source_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Validate required columns, check leakage guards, optionally filter uncertain labels, and return (clean_df, feature_cols)."""
    missing_ids = [col for col in ID_COLUMNS + [LABEL_COLUMN] if col not in df.columns]
    if missing_ids:
        raise KeyError(
            f"Candidate feature table is missing required columns {missing_ids}. "
            f"Expected at least {ID_COLUMNS + [LABEL_COLUMN]}."
        )

    assert_no_evaluator_leakage(df, source_path=source_path)

    clean_df = df.copy()
    if drop_uncertain_labels and UNCERTAIN_COLUMN in clean_df.columns:
        uncertain_mask = clean_df[UNCERTAIN_COLUMN].fillna(False).astype(bool)
        clean_df = clean_df.loc[~uncertain_mask].reset_index(drop=True)

    clean_df[LABEL_COLUMN] = clean_df[LABEL_COLUMN].astype(int)
    unique_labels = set(clean_df[LABEL_COLUMN].unique())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(f"Binary candidate label column must contain only {{0, 1}}, got {unique_labels}.")

    resolved_cols = resolve_feature_columns(clean_df, explicit_features=feature_columns)
    return clean_df, resolved_cols


def load_feature_table(path: Path) -> pd.DataFrame:
    """Load Member 1's candidate feature matrix from CSV or Parquet with leakage checks."""
    assert_no_evaluator_leakage(pd.DataFrame(), source_path=path)
    if not path.exists():
        raise FileNotFoundError(f"Feature table not found at: {path}")

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix in {".csv", ".gz"}:
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported feature file extension '{suffix}'. Use .csv or .parquet.")

    assert_no_evaluator_leakage(df, source_path=path)
    return df
