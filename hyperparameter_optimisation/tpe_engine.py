"""Bayesian TPE (Tree-structured Parzen Estimator) optimization engine with SQLite & CSV persistence.

Uses `optuna` (`optuna.samplers.TPESampler(seed=seed)`) when installed in the Python environment,
and includes a self-contained Tree-structured Parzen Estimator (`NativeTPEStudy`) with identical
`suggest_float` / `suggest_int` / `suggest_categorical` semantics and SQLite trial storage so
Bayesian optimization works reproducibly even in offline environments.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class TrialRecord:
    """Completed Bayesian optimization trial record."""

    number: int
    value: float
    duration_s: float
    params: Dict[str, Any]
    user_attrs: Dict[str, Any] = field(default_factory=dict)
    state: str = "COMPLETE"


class NativeTPETrial:
    """Trial object implementing Optuna's `suggest_*` and `set_user_attr` interface using TPE."""

    def __init__(
        self,
        number: int,
        rng: np.random.Generator,
        completed_trials: Sequence[TrialRecord],
        n_startup_trials: int = 6,
        gamma: float = 0.25,
        n_ei_candidates: int = 24,
    ) -> None:
        self.number = number
        self._rng = rng
        self._history = list(completed_trials)
        self._n_startup = n_startup_trials
        self._gamma = gamma
        self._n_ei_candidates = n_ei_candidates
        self.params: Dict[str, Any] = {}
        self.user_attrs: Dict[str, Any] = {}

    def set_user_attr(self, key: str, value: Any) -> None:
        self.user_attrs[key] = value

    def _split_good_bad(self, param_name: str) -> tuple[List[Any], List[Any]]:
        valid = [t for t in self._history if param_name in t.params and np.isfinite(t.value)]
        if len(valid) < self._n_startup:
            return [], []
        valid.sort(key=lambda t: t.value, reverse=True)
        n_good = max(2, int(np.ceil(self._gamma * len(valid))))
        good_vals = [t.params[param_name] for t in valid[:n_good]]
        bad_vals = [t.params[param_name] for t in valid[n_good:]]
        return good_vals, bad_vals

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        log: bool = False,
    ) -> float:
        good, bad = self._split_good_bad(name)
        lo = np.log(low) if log else float(low)
        hi = np.log(high) if log else float(high)

        if not good or not bad:
            raw = float(self._rng.uniform(lo, hi))
            val = float(np.exp(raw)) if log else raw
            self.params[name] = val
            return val

        g_arr = np.log(np.asarray(good, dtype=float)) if log else np.asarray(good, dtype=float)
        b_arr = np.log(np.asarray(bad, dtype=float)) if log else np.asarray(bad, dtype=float)

        sigma_g = max((hi - lo) / max(len(g_arr), 3), 1e-3)
        sigma_b = max((hi - lo) / max(len(b_arr), 3), 1e-3)

        # Sample candidate points from the good Parzen KDE l(x) and evaluate EI ratio l(x)/g(x)
        comp_idx = self._rng.integers(0, len(g_arr), size=self._n_ei_candidates)
        cands = np.clip(
            g_arr[comp_idx] + self._rng.normal(0.0, sigma_g, size=self._n_ei_candidates),
            lo,
            hi,
        )

        # Evaluate log l(x) - log g(x)
        diff_g = (cands[:, None] - g_arr[None, :]) / sigma_g
        l_density = np.mean(np.exp(-0.5 * diff_g**2), axis=1) + 1e-12

        diff_b = (cands[:, None] - b_arr[None, :]) / sigma_b
        g_density = np.mean(np.exp(-0.5 * diff_b**2), axis=1) + 1e-12

        best_cand = float(cands[int(np.argmax(l_density / g_density))])
        val = float(np.exp(best_cand)) if log else best_cand
        self.params[name] = val
        return val

    def suggest_int(
        self,
        name: str,
        low: int,
        high: int,
        *,
        step: int = 1,
    ) -> int:
        grid = np.arange(int(low), int(high) + 1, int(step), dtype=int)
        good, bad = self._split_good_bad(name)
        if not good or not bad:
            val = int(self._rng.choice(grid))
            self.params[name] = val
            return val

        g_arr = np.asarray(good, dtype=float)
        b_arr = np.asarray(bad, dtype=float)
        span = max(float(high - low), 1.0)
        sigma_g = max(span / max(len(g_arr), 3), float(step))
        sigma_b = max(span / max(len(b_arr), 3), float(step))

        diff_g = (grid[:, None] - g_arr[None, :]) / sigma_g
        l_prob = np.mean(np.exp(-0.5 * diff_g**2), axis=1) + 1e-9

        diff_b = (grid[:, None] - b_arr[None, :]) / sigma_b
        g_prob = np.mean(np.exp(-0.5 * diff_b**2), axis=1) + 1e-9

        scores = l_prob / g_prob
        probs = scores / np.sum(scores)
        # Pick from top EI candidates with stochastic tie-breaking
        val = int(self._rng.choice(grid, p=probs))
        self.params[name] = val
        return val

    def suggest_categorical(self, name: str, choices: Sequence[Any]) -> Any:
        choices_list = list(choices)
        good, bad = self._split_good_bad(name)
        if not good or not bad:
            idx = int(self._rng.integers(0, len(choices_list)))
            val = choices_list[idx]
            self.params[name] = val
            return val

        k = len(choices_list)
        l_counts = np.array([sum(1 for g in good if g == c) + 1.0 for c in choices_list])
        b_counts = np.array([sum(1 for b in bad if b == c) + 1.0 for c in choices_list])
        l_prob = l_counts / (len(good) + k)
        g_prob = b_counts / (len(bad) + k)

        ei_weights = l_prob / g_prob
        ei_probs = ei_weights / np.sum(ei_weights)
        idx = int(self._rng.choice(np.arange(k), p=ei_probs))
        val = choices_list[idx]
        self.params[name] = val
        return val


def _persist_trials_to_sqlite(trials: Sequence[TrialRecord], sqlite_path: Path) -> None:
    """Persist all Bayesian optimization trials to a SQLite database file."""
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    if sqlite_path.exists():
        sqlite_path.unlink()

    conn = sqlite3.connect(str(sqlite_path))
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE trials (
                trial_number INTEGER PRIMARY KEY,
                state TEXT NOT NULL,
                objective_value REAL NOT NULL,
                duration_s REAL NOT NULL,
                params_json TEXT NOT NULL,
                user_attrs_json TEXT NOT NULL
            )
            """
        )
        for tr in trials:
            cur.execute(
                """
                INSERT INTO trials (
                    trial_number, state, objective_value, duration_s, params_json, user_attrs_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    int(tr.number),
                    tr.state,
                    float(tr.value),
                    float(tr.duration_s),
                    json.dumps(tr.params, sort_keys=True),
                    json.dumps(tr.user_attrs, sort_keys=True),
                ),
            )
        conn.commit()
    finally:
        conn.close()


def execute_tpe_study(
    objective_fn: Callable[[Any], float],
    n_trials: int = 25,
    seed: int = 42,
    sqlite_path: Optional[Path] = None,
) -> tuple[List[TrialRecord], TrialRecord, str]:
    """Run a seeded TPE Bayesian optimization study using Optuna (if installed) or NativeTPE.

    Returns:
        trials: list of all completed TrialRecords
        best_trial: the TrialRecord with the highest objective value
        backend_used: string describing the TPE sampler backend used
    """
    trials: List[TrialRecord] = []
    backend_used = "TPEBayesianSampler (Tree-structured Parzen Estimator)"

    try:
        import optuna  # type: ignore

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        def _wrapped(trial: Any) -> float:
            t0 = time.perf_counter()
            val = float(objective_fn(trial))
            dur = time.perf_counter() - t0
            trial.set_user_attr("duration_s", round(dur, 5))
            return val

        study.optimize(_wrapped, n_trials=n_trials)
        for tr in study.trials:
            trials.append(
                TrialRecord(
                    number=int(tr.number),
                    value=float(tr.value),
                    duration_s=float(tr.user_attrs.get("duration_s", 0.0)),
                    params=dict(tr.params),
                    user_attrs={k: v for k, v in tr.user_attrs.items() if k != "duration_s"},
                    state="COMPLETE",
                )
            )
        backend_used = f"Optuna TPESampler (optuna {optuna.__version__}, seed={seed})"
    except ImportError:
        rng = np.random.default_rng(seed)
        for idx in range(n_trials):
            trial = NativeTPETrial(
                number=idx,
                rng=rng,
                completed_trials=trials,
                n_startup_trials=max(5, min(8, n_trials // 3)),
            )
            t0 = time.perf_counter()
            val = float(objective_fn(trial))
            dur = time.perf_counter() - t0
            trials.append(
                TrialRecord(
                    number=idx,
                    value=val,
                    duration_s=round(dur, 5),
                    params=dict(trial.params),
                    user_attrs=dict(trial.user_attrs),
                    state="COMPLETE",
                )
            )
        backend_used = f"TPEBayesianSampler (Parzen KDE l(x)/g(x), gamma=0.25, seed={seed})"

    if sqlite_path is not None:
        _persist_trials_to_sqlite(trials, sqlite_path=sqlite_path)

    best_trial = max(trials, key=lambda t: t.value)
    return trials, best_trial, backend_used


def trials_to_dataframe(trials: Sequence[TrialRecord]) -> pd.DataFrame:
    """Convert a list of `TrialRecord` objects into a detailed trial history DataFrame."""
    rows: List[Dict[str, Any]] = []
    best_so_far = -float("inf")

    for tr in trials:
        best_so_far = max(best_so_far, tr.value)
        row: Dict[str, Any] = {
            "trial_number": tr.number,
            "state": tr.state,
            "objective_value": round(tr.value, 6),
            "best_value_so_far": round(best_so_far, 6),
            "duration_s": round(tr.duration_s, 5),
        }
        for k, v in tr.user_attrs.items():
            row[k] = round(v, 5) if isinstance(v, float) else v
        for pk, pv in tr.params.items():
            row[f"param_{pk}"] = pv
        rows.append(row)

    return pd.DataFrame(rows)
