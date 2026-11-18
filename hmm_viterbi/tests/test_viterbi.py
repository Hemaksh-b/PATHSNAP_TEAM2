"""Synthetic tests with known answers for the Viterbi decoder (Issue #7).

Run from the repo root:  python -m pytest hmm_viterbi
"""
import itertools

import numpy as np
import pytest

from hmm_viterbi import ViterbiConfig, classifier_emission_logp, decode, gaussian_emission_logp
from hmm_viterbi.viterbi import GAP_BREAK, NO_TRANSITION_BREAK

LOG = np.log


def const_transition(matrix):
    """Transition fn returning the same log matrix for every step."""
    m = LOG(np.asarray(matrix, dtype=float))
    return lambda _prev, _next: m


def uniform_transition(candidates):
    return lambda p, n: np.zeros((len(candidates[p]), len(candidates[n])))


# ---------------------------------------------------------------- basics

def test_single_observation_picks_max_emission():
    cands = [[7, 3, 9]]
    res = decode(cands, [LOG([0.2, 0.7, 0.1])], uniform_transition(cands))
    assert res.arc_ids == (3,)
    assert res.segments == ((0, 0),)
    assert res.breaks == ()
    assert res.log_prob == pytest.approx(LOG(0.7))


def test_two_steps_hand_computed():
    cands = [[10, 20], [10, 20]]
    emis = [LOG([0.5, 0.5]), LOG([0.5, 0.5])]
    trans = const_transition([[0.9, 0.1], [0.2, 0.8]])
    res = decode(cands, emis, trans)
    # paths: AA=.5*.9*.5=.225, AB=.025, BA=.05, BB=.2  -> A,A
    assert res.arc_ids == (10, 10)
    assert res.log_prob == pytest.approx(LOG(0.225))


def test_viterbi_differs_from_greedy_argmax():
    cands = [[10, 20], [10, 20]]
    emis = [LOG([0.6, 0.4]), LOG([0.3, 0.7])]
    trans = const_transition([[0.5, 0.01], [0.01, 0.99]])
    greedy = tuple(int(c[np.argmax(e)]) for c, e in zip(cands, emis))
    res = decode(cands, emis, trans)
    assert greedy == (10, 20)  # per-point argmax jumps to an unconnected arc
    assert res.arc_ids == (20, 20)  # BB = .4*.99*.7 = .277 is the global best


@pytest.mark.parametrize("seed", range(25))
def test_matches_brute_force_on_random_problems(seed):
    rng = np.random.default_rng(seed)
    t_len = int(rng.integers(1, 6))
    sizes = rng.integers(1, 4, size=t_len)
    cands = [list(rng.choice(1000, size=k, replace=False)) for k in sizes]
    emis = [LOG(rng.uniform(0.01, 1.0, size=k)) for k in sizes]
    mats = {
        (i, i + 1): LOG(rng.uniform(0.01, 1.0, size=(sizes[i], sizes[i + 1])))
        for i in range(t_len - 1)
    }
    res = decode(cands, emis, lambda p, n: mats[(p, n)])

    best_score, best_path = -np.inf, None
    for path in itertools.product(*[range(k) for k in sizes]):
        s = sum(emis[t][j] for t, j in enumerate(path))
        s += sum(mats[(t, t + 1)][path[t], path[t + 1]] for t in range(t_len - 1))
        if s > best_score:
            best_score, best_path = s, path
    assert res.log_prob == pytest.approx(best_score)
    assert res.arc_ids == tuple(int(cands[t][j]) for t, j in enumerate(best_path))


# ---------------------------------------------------------------- robustness

def test_empty_candidate_observation_is_skipped_and_bridged():
    cands = [[1, 2], [], [1, 2]]
    emis = [LOG([0.9, 0.1]), np.array([]), LOG([0.1, 0.9])]
    calls = []

    def trans(p, n):
        calls.append((p, n))
        return LOG([[0.01, 0.99], [0.5, 0.5]])

    res = decode(cands, emis, trans)
    assert calls == [(0, 2)]  # bridged over the empty observation
    assert res.arc_ids == (1, None, 2)
    assert res.segments == ((0, 2),)


def test_all_empty_returns_none():
    res = decode([[], []], [np.array([]), np.array([])], lambda p, n: None)
    assert res.arc_ids == (None, None)
    assert res.segments == ()
    assert res.log_prob == 0.0


def test_impossible_transition_step_breaks_segment():
    cands = [[1, 2], [3, 4], [3, 4]]
    emis = [LOG([0.8, 0.2]), LOG([0.3, 0.7]), LOG([0.4, 0.6])]

    def trans(p, n):
        if (p, n) == (0, 1):
            return np.full((2, 2), -np.inf)  # disconnected road components
        return LOG([[0.9, 0.1], [0.1, 0.9]])

    res = decode(cands, emis, trans)
    assert res.breaks == ((1, NO_TRANSITION_BREAK),)
    assert res.segments == ((0, 0), (1, 2))
    assert res.arc_ids == (1, 4, 4)
    assert np.isfinite(res.log_prob)


def test_large_time_gap_splits_sequence_without_calling_transition():
    cands = [[1, 2], [1, 2], [1, 2]]
    emis = [LOG([0.9, 0.1]), LOG([0.9, 0.1]), LOG([0.1, 0.9])]
    calls = []

    def trans(p, n):
        calls.append((p, n))
        return LOG([[0.99, 0.01], [0.01, 0.99]])

    ts = np.array([0.0, 1.0, 500.0])
    res = decode(cands, emis, trans, timestamps=ts, config=ViterbiConfig(gap_break_s=30.0))
    assert calls == [(0, 1)]
    assert res.breaks == ((2, GAP_BREAK),)
    assert res.arc_ids == (1, 1, 2)


def test_gap_break_disabled():
    cands = [[1], [1]]
    emis = [np.zeros(1), np.zeros(1)]
    res = decode(cands, emis, uniform_transition(cands), timestamps=np.array([0.0, 1e6]),
                 config=ViterbiConfig(gap_break_s=None))
    assert res.breaks == ()


def test_long_sequence_is_numerically_stable():
    rng = np.random.default_rng(0)
    t_len, k = 10_000, 5
    cands = [list(range(k))] * t_len
    emis = [LOG(rng.uniform(1e-12, 1e-6, size=k)) for _ in range(t_len)]
    mat = LOG(rng.uniform(1e-9, 1e-3, size=(k, k)))
    res = decode(cands, emis, lambda p, n: mat)
    assert all(a is not None for a in res.arc_ids)
    assert np.isfinite(res.log_prob)


def test_arc_ids_are_preserved_not_reindexed():
    cands = [[905, 17, 44012], [3, 905]]
    emis = [LOG([0.1, 0.1, 0.8]), LOG([0.2, 0.8])]
    res = decode(cands, emis, uniform_transition(cands))
    assert res.arc_ids == (44012, 905)


def test_ties_break_to_lowest_candidate_index():
    cands = [[5, 6], [7, 8]]
    emis = [np.zeros(2), np.zeros(2)]
    res = decode(cands, emis, uniform_transition(cands))
    assert res.arc_ids == (5, 7)


# ---------------------------------------------------------------- validation

def test_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        decode([[1, 2]], [np.zeros(3)], lambda p, n: None)
    with pytest.raises(ValueError):
        decode([[1], [2]], [np.zeros(1)], lambda p, n: None)


def test_rejects_wrong_transition_shape():
    cands = [[1, 2], [3]]
    with pytest.raises(ValueError, match="shape"):
        decode(cands, [np.zeros(2), np.zeros(1)], lambda p, n: np.zeros((2, 2)))


def test_rejects_nan():
    with pytest.raises(ValueError, match="NaN"):
        decode([[1]], [np.array([np.nan])], lambda p, n: None)
    with pytest.raises(ValueError, match="NaN"):
        decode([[1], [2]], [np.zeros(1), np.zeros(1)], lambda p, n: np.array([[np.nan]]))


# ---------------------------------------------------------------- emissions

def test_classifier_emission_clips_zero_probability():
    out = classifier_emission_logp(np.array([0.0, 0.5, 1.0]))
    assert np.all(np.isfinite(out))
    assert out[1] == pytest.approx(LOG(0.5))
    assert out[2] == pytest.approx(0.0)


def test_gaussian_emission_prefers_closer_arc():
    out = gaussian_emission_logp(np.array([1.0, 5.0, 20.0]), sigma_m=5.0)
    assert out[0] > out[1] > out[2]
    with pytest.raises(ValueError):
        gaussian_emission_logp(np.array([1.0]), sigma_m=0.0)
