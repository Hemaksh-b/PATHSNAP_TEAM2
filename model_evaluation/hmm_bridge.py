"""Bridge between Member 2's candidate classifier probabilities (#5) and Member 3's HMM decoder (#7/#8/#9).

Guarantees:
1. Exported candidate score tables strictly follow the schema:
       `trip_id, obs_idx, arc_id, prob`
2. Exact `(trip_id, obs_idx, arc_id)` alignment is preserved (never relies on implicit row position).
3. Verifies end-to-end compatibility with `hmm_viterbi.emissions.classifier_emission_logp`
   and `hmm_viterbi.viterbi.decode`.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from hmm_viterbi.emissions import classifier_emission_logp
from hmm_viterbi.viterbi import DecodeResult, ViterbiConfig, decode

SCORE_COLUMNS: List[str] = ["trip_id", "obs_idx", "arc_id", "prob"]


def build_candidate_scores_table(
    df: pd.DataFrame,
    probs: Sequence[float] | np.ndarray,
) -> pd.DataFrame:
    """Create the standardized `trip_id, obs_idx, arc_id, prob` dataframe for Member 3."""
    missing = [c for c in ("trip_id", "obs_idx", "arc_id") if c not in df.columns]
    if missing:
        raise KeyError(f"DataFrame missing required identifier columns: {missing}")

    p = np.asarray(probs, dtype=float)
    if len(p) != len(df):
        raise ValueError(f"Length mismatch: {len(df)} candidate rows vs {len(p)} probabilities.")
    if np.isnan(p).any():
        raise ValueError("Candidate probabilities contain NaN.")
    if (p < 0.0).any() or (p > 1.0).any():
        raise ValueError("Candidate probabilities must lie in [0.0, 1.0].")

    out = pd.DataFrame(
        {
            "trip_id": df["trip_id"].to_numpy(),
            "obs_idx": df["obs_idx"].to_numpy(dtype=int),
            "arc_id": df["arc_id"].to_numpy(dtype=int),
            "prob": np.round(p, 6),
        }
    )
    return out


def scores_df_to_hmm_inputs(
    scores_df: pd.DataFrame,
    trip_id: str,
    total_observations: Optional[int] = None,
) -> Tuple[List[List[int]], List[np.ndarray]]:
    """Convert a `trip_id, obs_idx, arc_id, prob` score table into `hmm_viterbi.decode` inputs.

    Returns:
        candidates: list of candidate `arc_id` lists per observation `0 .. T-1`
        emission_logp: list of log-probability arrays `log P(o_t | c)` per observation `0 .. T-1`
    """
    missing = [c for c in SCORE_COLUMNS if c not in scores_df.columns]
    if missing:
        raise KeyError(f"Score table missing required columns: {missing}")

    sub = scores_df.loc[scores_df["trip_id"].astype(str) == str(trip_id)]
    if sub.empty:
        raise ValueError(f"No candidate scores found for trip_id='{trip_id}'.")

    max_obs = int(sub["obs_idx"].max())
    n_obs = max(max_obs + 1, int(total_observations or 0))

    candidates: List[List[int]] = [[] for _ in range(n_obs)]
    emission_logp: List[np.ndarray] = [np.empty(0, dtype=float) for _ in range(n_obs)]

    for obs_idx, group in sub.groupby("obs_idx", sort=True):
        t = int(obs_idx)
        cands = [int(a) for a in group["arc_id"].to_numpy()]
        probs = group["prob"].to_numpy(dtype=float)
        candidates[t] = cands
        emission_logp[t] = classifier_emission_logp(probs)

    return candidates, emission_logp


def verify_hmm_decoder_compatibility(
    scores_df: pd.DataFrame,
    trip_id: Optional[str] = None,
) -> DecodeResult:
    """Smoke-test that `scores_df` loads cleanly into Member 3's `hmm_viterbi.decode`."""
    target_trip = str(trip_id) if trip_id is not None else str(scores_df["trip_id"].iloc[0])
    candidates, emission_logp = scores_df_to_hmm_inputs(scores_df, trip_id=target_trip)

    def _same_arc_bonus_transition(t_prev: int, t_next: int) -> np.ndarray:
        prev_arcs = np.asarray(candidates[t_prev], dtype=int)[:, None]
        next_arcs = np.asarray(candidates[t_next], dtype=int)[None, :]
        return np.where(prev_arcs == next_arcs, -0.1, -1.5)

    return decode(
        candidates=candidates,
        emission_logp=emission_logp,
        transition_logp=_same_arc_bonus_transition,
        config=ViterbiConfig(),
    )
