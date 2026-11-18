"""PathSnap Model Evaluation, Comparison, Plotting, and HMM Score Export Package (Issue #5)."""

from .hmm_bridge import scores_df_to_hmm_inputs, verify_hmm_decoder_compatibility
from .metrics import (
    compute_all_evaluation_metrics,
    compute_route_continuity,
)

__all__ = [
    "compute_all_evaluation_metrics",
    "compute_route_continuity",
    "scores_df_to_hmm_inputs",
    "verify_hmm_decoder_compatibility",
]
