"""Log-space Viterbi decoder for HMM map-matching (Issue #7).

Hidden states at step t are the candidate road arcs for GPS observation t.
The decoder is independent of how emissions/transitions are computed:

* emissions are passed in as one log-probability array per observation;
* transitions come from a callable ``transition_logp(t_prev, t_next)`` that
  returns a ``(K_prev, K_next)`` log-probability matrix (see Issue #8).

Recurrence (per segment)::

    delta_0(j) = logB_0(j)
    delta_t(j) = max_i [delta_{t-1}(i) + logA_t(i, j)] + logB_t(j)
    psi_t(j)   = argmax_i [delta_{t-1}(i) + logA_t(i, j)]

The best path is recovered by backtracking ``psi`` from ``argmax_j delta_T(j)``.

Robustness rules:

* observations with no candidates are skipped and decoded as ``None``;
* a time gap larger than ``gap_break_s`` starts a new, independent segment;
* a step where every transition is at or below ``min_logp`` (no valid move)
  starts a new segment instead of forcing an impossible path;
* all log-probabilities are floored at ``min_logp`` so ``-inf`` never
  propagates; NaN inputs raise ``ValueError``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

TransitionFn = Callable[[int, int], np.ndarray]

GAP_BREAK = "time_gap"
NO_TRANSITION_BREAK = "no_valid_transition"


@dataclass(frozen=True)
class ViterbiConfig:
    """Decoder settings. Tune only on training / student-test trips."""

    min_logp: float = -1e4
    gap_break_s: float | None = 30.0


@dataclass(frozen=True)
class DecodeResult:
    """Output of :func:`decode`.

    arc_ids:   predicted arc_id per observation (``None`` = no candidates).
    segments:  (first_obs, last_obs) of each independently decoded segment.
    breaks:    (obs_index, reason) for every segment start after the first.
    log_prob:  sum of the best-path log-probabilities of all segments.
    """

    arc_ids: tuple[int | None, ...]
    segments: tuple[tuple[int, int], ...]
    breaks: tuple[tuple[int, str], ...]
    log_prob: float


def _validate(
    candidates: Sequence[Sequence[int]],
    emission_logp: Sequence[np.ndarray],
    timestamps: np.ndarray | None,
) -> None:
    if len(candidates) != len(emission_logp):
        raise ValueError(
            f"candidates ({len(candidates)}) and emission_logp "
            f"({len(emission_logp)}) must have the same length"
        )
    if timestamps is not None and len(timestamps) != len(candidates):
        raise ValueError("timestamps must have one entry per observation")
    for t, (cands, emis) in enumerate(zip(candidates, emission_logp)):
        if len(cands) != len(emis):
            raise ValueError(
                f"observation {t}: {len(cands)} candidates but {len(emis)} emissions"
            )
        if np.isnan(np.asarray(emis, dtype=float)).any():
            raise ValueError(f"observation {t}: emission_logp contains NaN")


def _backtrack(
    steps: list[int],
    backptrs: list[np.ndarray],
    delta: np.ndarray,
    candidates: Sequence[Sequence[int]],
    out: list[int | None],
) -> float:
    """Fill ``out`` for one segment and return its best-path log-prob."""
    state = int(np.argmax(delta))
    best = float(delta[state])
    for k in range(len(steps) - 1, -1, -1):
        out[steps[k]] = int(candidates[steps[k]][state])
        if k > 0:
            state = int(backptrs[k - 1][state])
    return best


def _step_transition(
    transition_logp: TransitionFn,
    prev: int,
    nxt: int,
    shape: tuple[int, int],
    min_logp: float,
) -> np.ndarray:
    log_a = np.asarray(transition_logp(prev, nxt), dtype=float)
    if log_a.shape != shape:
        raise ValueError(
            f"transition_logp({prev}, {nxt}) returned shape {log_a.shape}, "
            f"expected {shape}"
        )
    if np.isnan(log_a).any():
        raise ValueError(f"transition_logp({prev}, {nxt}) contains NaN")
    return np.maximum(log_a, min_logp)


def decode(
    candidates: Sequence[Sequence[int]],
    emission_logp: Sequence[np.ndarray],
    transition_logp: TransitionFn,
    timestamps: np.ndarray | None = None,
    config: ViterbiConfig = ViterbiConfig(),
) -> DecodeResult:
    """Decode the most likely candidate arc for every observation.

    Args:
        candidates: arc_ids per observation (inner lists may be empty).
        emission_logp: log P(obs_t | candidate) arrays, same shapes as candidates.
        transition_logp: ``f(t_prev, t_next) -> (K_prev, K_next)`` log-probs.
            ``t_prev`` is the last observation that had candidates, so empty
            observations are bridged automatically.
        timestamps: seconds per observation; enables gap-based segment breaks.
        config: decoder settings.
    """
    _validate(candidates, emission_logp, timestamps)
    min_logp = config.min_logp
    out: list[int | None] = [None] * len(candidates)
    segments: list[tuple[int, int]] = []
    breaks: list[tuple[int, str]] = []
    total = 0.0

    steps: list[int] = []
    backptrs: list[np.ndarray] = []
    delta: np.ndarray | None = None

    for t, cands in enumerate(candidates):
        if len(cands) == 0:
            continue
        emis = np.maximum(np.asarray(emission_logp[t], dtype=float), min_logp)

        reason = None
        log_a = None
        if delta is not None:
            prev = steps[-1]
            gap = config.gap_break_s
            if timestamps is not None and gap is not None and timestamps[t] - timestamps[prev] > gap:
                reason = GAP_BREAK
            else:
                log_a = _step_transition(
                    transition_logp, prev, t, (len(candidates[prev]), len(cands)), min_logp
                )
                if log_a.max() <= min_logp:
                    reason = NO_TRANSITION_BREAK

        if delta is None or reason is not None:
            if delta is not None:
                total += _backtrack(steps, backptrs, delta, candidates, out)
                segments.append((steps[0], steps[-1]))
                breaks.append((t, reason))
            steps, backptrs, delta = [t], [], emis
            continue

        scores = delta[:, None] + log_a  # (K_prev, K_next)
        bp = np.argmax(scores, axis=0)  # first max -> lowest index on ties
        delta = scores[bp, np.arange(len(cands))] + emis
        backptrs.append(bp)
        steps.append(t)

    if delta is not None:
        total += _backtrack(steps, backptrs, delta, candidates, out)
        segments.append((steps[0], steps[-1]))

    return DecodeResult(tuple(out), tuple(segments), tuple(breaks), total)
