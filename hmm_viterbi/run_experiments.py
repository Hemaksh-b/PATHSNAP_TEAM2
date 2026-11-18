"""Decoder tuning and pointwise vs HMM experiments on real trips (Issues #7, #9).

Run from the repo root:
    python -m hmm_viterbi.run_experiments --tune --tune-trips 20   # tune on TRAINING trips -> artifacts/hmm_config.json
    python -m hmm_viterbi.run_experiments                          # compare on STUDENT-TEST trips

--emissions auto (default): use Member 2's exported classifier if its model card
is real; if the card says is_mock_data, fall back to Gaussian distance emissions
and say so. --emissions classifier with a mock model is refused unless
--allow-mock, and then everything is written under .../mock/ only.

Writes (emission source recorded in every CSV):
    hmm_viterbi/results/decoder_tuning.csv              grid on training trips
    artifacts/hmm_config.json                           chosen decoder settings
    visualisations/results/hmm_comparison_per_trip.csv  per trip, both methods
    visualisations/results/hmm_comparison_summary.csv   pooled over trips
    visualisations/results/decoded/<trip>.csv           obs_idx, aligned, pointwise, hmm arc
Evaluator data is never read.
"""
from __future__ import annotations

import argparse
import itertools
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from data_processing.config import ARTIFACTS_DIR, STUDENT_TEST_DIR, TRAIN_DIR, VIS_DIR
from data_processing.loader import list_trips, load_trip
from data_processing.projection import project_trip
from feature_engineering.alignment import align_observations_to_route
from hmm_viterbi.decode_trip import (HMM_CONFIG_FILE, Classifier, DecodeSettings, decode_prepared,
                                     load_classifier, load_settings, prepare_trip)
from hmm_viterbi.route_metrics import all_metrics

TUNING_DIR = Path(__file__).resolve().parent / "results"
COMPARE_DIR = VIS_DIR / "results"
BETAS = (0.02, 0.05, 0.1, 0.2)
GAP_BREAKS_S = (15.0, 30.0, 60.0)
METHODS = ("pointwise", "hmm")


def choose_classifier(mode: str, allow_mock: bool, artifacts_dir: Path) -> tuple[Classifier | None, bool]:
    """Return (classifier or None, is_mock_run)."""
    if mode == "gaussian":
        return None, False
    clf = load_classifier(artifacts_dir)
    if clf is None:
        if mode == "classifier":
            raise SystemExit(f"no exported model in {artifacts_dir}")
        print("no exported model -> Gaussian distance emissions")
        return None, False
    if clf.is_mock:
        if mode == "classifier" and allow_mock:
            print("WARNING: exported model is MOCK - results go to .../mock/ and must not be reported")
            return clf, True
        if mode == "classifier":
            raise SystemExit("exported model card says is_mock_data; refusing (use --allow-mock for a smoke test)")
        print("exported model card says is_mock_data -> using Gaussian distance emissions instead")
        return None, False
    return clf, False


def reference_info(trip_dir: Path) -> tuple[np.ndarray, np.ndarray, list[int], pd.DataFrame]:
    """Aligned arc and certainty per observation, route arcs, arcs table — scoring only."""
    trip = load_trip(trip_dir)  # permitted partitions only (train / student_test)
    ptrip = project_trip(trip)
    align = align_observations_to_route(ptrip).sort_values("obs_idx")
    aligned = align["aligned_arc_id"].fillna(-1).to_numpy(dtype=int)
    certain = ~align["is_uncertain"].to_numpy(dtype=bool)
    return aligned, certain, trip["route"]["arc_id"].tolist(), trip["arcs"]


def evaluate_trip(trip_dir: Path, settings: DecodeSettings, clf: Classifier | None,
                  prep=None, ref=None) -> tuple[list[dict], pd.DataFrame]:
    prep = prep or prepare_trip(trip_dir, settings)
    aligned, certain, route, arcs = ref or reference_info(trip_dir)
    out = decode_prepared(prep, settings, clf)
    rows = []
    for method, ids in zip(METHODS, (out.pointwise_arc_ids, out.hmm_arc_ids)):
        rows.append({"trip_id": out.trip_id, "method": method, "emission_source": out.emission_source,
                     "n_obs": len(ids), "n_segments": len(out.result.segments) if method == "hmm" else 1,
                     **all_metrics(ids, aligned, certain, route, arcs)})
    decoded = pd.DataFrame({"obs_idx": np.arange(len(aligned)), "aligned_arc": aligned,
                            "pointwise_arc": out.pointwise_arc_ids, "hmm_arc": out.hmm_arc_ids})
    return rows, decoded


def _require_trips(trip_dirs: list[Path], where: Path) -> list[Path]:
    if not trip_dirs:
        raise SystemExit(f"no trips found in {where} - put the dataset under handouts/ (see README)")
    return trip_dirs


def tune(trip_dirs: list[Path], base: DecodeSettings, clf: Classifier | None) -> tuple[pd.DataFrame, DecodeSettings]:
    cache = {d: (prepare_trip(d, base), reference_info(d)) for d in trip_dirs}
    rows = []
    for beta, gap in itertools.product(BETAS, GAP_BREAKS_S):
        s = replace(base, transitions=replace(base.transitions, beta=beta),
                    viterbi=replace(base.viterbi, gap_break_s=gap))
        per_trip = [evaluate_trip(d, s, clf, prep, ref)[0][1] for d, (prep, ref) in cache.items()]
        n = sum(r["n_obs"] for r in per_trip)
        acc = sum(r["observation_accuracy"] * r["n_obs"] for r in per_trip) / n
        rows.append({"beta": beta, "gap_break_s": gap, "pooled_observation_accuracy": acc,
                     "n_trips": len(per_trip), "emission_source": per_trip[0]["emission_source"]})
        print(f"beta={beta:<5} gap={gap:<5} pooled accuracy={acc:.4f}")
    grid = pd.DataFrame(rows)
    best = grid.loc[grid["pooled_observation_accuracy"].idxmax()]
    chosen = replace(base, transitions=replace(base.transitions, beta=float(best["beta"])),
                     viterbi=replace(base.viterbi, gap_break_s=float(best["gap_break_s"])))
    return grid, chosen


def summarise(per_trip: pd.DataFrame) -> pd.DataFrame:
    g = per_trip.assign(correct=per_trip["observation_accuracy"] * per_trip["n_obs"])
    out = g.groupby(["method", "emission_source"]).agg(
        n_trips=("trip_id", "nunique"), n_obs=("n_obs", "sum"), correct=("correct", "sum"),
        mean_observation_accuracy_certain=("observation_accuracy_certain", "mean"),
        mean_route_continuity=("route_continuity", "mean"), mean_route_f1=("route_f1", "mean"),
        mean_segments=("n_segments", "mean"))
    out.insert(2, "pooled_observation_accuracy", out.pop("correct") / out["n_obs"])
    return out.reset_index()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HMM decoder tuning and comparison on real trips.")
    parser.add_argument("--tune", action="store_true", help="grid-search beta / gap on training trips")
    parser.add_argument("--tune-trips", type=int, default=20)
    parser.add_argument("--emissions", choices=["auto", "classifier", "gaussian"], default="auto")
    parser.add_argument("--allow-mock", action="store_true")
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    args = parser.parse_args(argv)

    clf, mock_run = choose_classifier(args.emissions, args.allow_mock, args.artifacts_dir)
    tuning_dir = TUNING_DIR / "mock" if mock_run else TUNING_DIR
    compare_dir = COMPARE_DIR / "mock" if mock_run else COMPARE_DIR
    config_dir = tuning_dir if mock_run else args.artifacts_dir
    settings = load_settings(args.artifacts_dir)

    if args.tune:
        grid, settings = tune(_require_trips(list_trips(TRAIN_DIR), TRAIN_DIR)[: args.tune_trips], settings, clf)
        tuning_dir.mkdir(parents=True, exist_ok=True)
        grid.to_csv(tuning_dir / "decoder_tuning.csv", index=False)
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / HMM_CONFIG_FILE).write_text(settings.to_json(), encoding="utf-8")
        print(f"wrote {tuning_dir / 'decoder_tuning.csv'} and {config_dir / HMM_CONFIG_FILE}")
        return 0

    (compare_dir / "decoded").mkdir(parents=True, exist_ok=True)
    rows = []
    for d in _require_trips(list_trips(STUDENT_TEST_DIR), STUDENT_TEST_DIR):
        trip_rows, decoded = evaluate_trip(d, settings, clf)
        rows.extend(trip_rows)
        decoded.to_csv(compare_dir / "decoded" / f"{d.name}.csv", index=False)
        print(f"{d.name}: pointwise {trip_rows[0]['observation_accuracy']:.3f} -> "
              f"HMM {trip_rows[1]['observation_accuracy']:.3f}")
    per_trip = pd.DataFrame(rows)
    per_trip.to_csv(compare_dir / "hmm_comparison_per_trip.csv", index=False)
    summary = summarise(per_trip)
    summary.to_csv(compare_dir / "hmm_comparison_summary.csv", index=False)
    print(summary.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
