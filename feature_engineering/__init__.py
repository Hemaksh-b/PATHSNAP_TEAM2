"""Feature engineering and reference-route label alignment module for PathSnap (Issue #3)."""

from __future__ import annotations

from feature_engineering.alignment import (
    AlignmentConfig,
    align_observations_to_route,
    assign_candidate_labels,
)
from feature_engineering.features import (
    FEATURE_COLUMNS,
    extract_candidate_features,
)
from feature_engineering.pipeline import (
    generate_alignment_map,
    process_partition,
)

__all__ = [
    "AlignmentConfig",
    "align_observations_to_route",
    "assign_candidate_labels",
    "FEATURE_COLUMNS",
    "extract_candidate_features",
    "process_partition",
    "generate_alignment_map",
]
