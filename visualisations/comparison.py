"""Pointwise vs HMM comparison charts and best/worst-case maps (Issue #9).

Run after ``python -m hmm_viterbi.run_experiments``:
    python -m visualisations.comparison

Reads  visualisations/results/hmm_comparison_per_trip.csv, decoded/<trip>.csv
Writes visualisations/plots/hmm_*.png, visualisations/results/hmm_improvement_per_trip.csv
and visualisations/maps/compare_<trip>.html for the trips where the HMM helped most
and least (report §5: "discuss cases where HMM decoding did not help").
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from data_processing.config import STUDENT_TEST_DIR, VIS_DIR  # noqa: E402
from data_processing.loader import load_trip  # noqa: E402
from visualisations.route_map import build_route_map  # noqa: E402

RESULTS = VIS_DIR / "results"
PLOTS = VIS_DIR / "plots"
MAPS = VIS_DIR / "maps"
METHODS = ("pointwise", "hmm")
LABELS = {"pointwise": "Pointwise (no sequence model)", "hmm": "HMM + Viterbi"}
COLORS = {"pointwise": "#f28e2b", "hmm": "#d62728"}
METRICS = {"observation_accuracy": "Point accuracy", "route_continuity": "Route continuity",
           "route_f1": "Route F1"}


def paired(per_trip: pd.DataFrame, metric: str) -> pd.DataFrame:
    """One row per trip: pointwise, hmm and delta (hmm - pointwise), sorted by delta."""
    wide = per_trip.pivot(index="trip_id", columns="method", values=metric)
    return wide.assign(delta=wide["hmm"] - wide["pointwise"]).sort_values("delta")


def bar_summary(per_trip: pd.DataFrame, out_dir: Path) -> Path:
    means = per_trip.groupby("method")[list(METRICS)].mean()
    source = per_trip["emission_source"].iloc[0]
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.38
    for i, method in enumerate(METHODS):
        xs = [m + (i - 0.5) * width for m in range(len(METRICS))]
        bars = ax.bar(xs, means.loc[method], width, label=LABELS[method], color=COLORS[method])
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
    ax.set_xticks(range(len(METRICS)), list(METRICS.values()))
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Student-test trips, mean over trips (emissions: {source})", fontsize=10)
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = out_dir / "hmm_vs_pointwise_summary.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def per_trip_scatter(per_trip: pd.DataFrame, metric: str, out_dir: Path) -> Path:
    wide = paired(per_trip, metric)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(wide["pointwise"], wide["hmm"], color="#1f4fd8")
    for trip, row in wide.iterrows():
        ax.annotate(str(trip)[-3:], (row["pointwise"], row["hmm"]), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    lo = float(min(wide[["pointwise", "hmm"]].min().min(), 1.0)) - 0.02
    ax.plot([lo, 1], [lo, 1], "--", color="#888", label="no change")
    ax.set_xlim(lo, 1.01)
    ax.set_ylim(lo, 1.01)
    ax.set_xlabel(f"{METRICS[metric]} — pointwise")
    ax.set_ylabel(f"{METRICS[metric]} — HMM + Viterbi")
    ax.set_title(f"Per trip {METRICS[metric].lower()} (above line = HMM helped)", fontsize=10)
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = out_dir / f"hmm_per_trip_{metric}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _ids(series: pd.Series) -> list[int | None]:
    return [None if pd.isna(v) else int(v) for v in series]


def comparison_map(trip_id: str, tag: str) -> Path:
    decoded = pd.read_csv(RESULTS / "decoded" / f"{trip_id}.csv")
    trip = load_trip(STUDENT_TEST_DIR / trip_id)
    fmap = build_route_map(
        trip, "student_test",
        predicted_arc_ids=_ids(decoded["hmm_arc"]),
        classifier_arc_ids=_ids(decoded["pointwise_arc"]),
        reference_arc_ids=trip["route"]["arc_id"].tolist(),
        title=f"Trip {trip_id} — {tag} · orange = pointwise, red = HMM, green = reference",
    )
    path = MAPS / f"compare_{trip_id}.html"
    fmap.save(str(path))
    return path


def main() -> int:
    per_trip = pd.read_csv(RESULTS / "hmm_comparison_per_trip.csv", dtype={"trip_id": str})
    PLOTS.mkdir(parents=True, exist_ok=True)
    MAPS.mkdir(parents=True, exist_ok=True)
    bar_summary(per_trip, PLOTS)
    for metric in ("observation_accuracy", "route_continuity"):
        per_trip_scatter(per_trip, metric, PLOTS)
    delta = paired(per_trip, "observation_accuracy")
    delta.to_csv(RESULTS / "hmm_improvement_per_trip.csv")
    worst, best = delta.index[0], delta.index[-1]
    comparison_map(best, f"HMM helped most (+{delta.loc[best, 'delta']:.3f} point accuracy)")
    comparison_map(worst, f"HMM helped least (+{delta.loc[worst, 'delta']:.3f} point accuracy)")
    print(f"plots in {PLOTS}; maps compare_{best}.html (most gain), compare_{worst}.html (least gain)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
