"""Export an additional, selectable candidate classifier for the API (Issue #30).

The tuned final model (#6/#28) stays in ``artifacts/``. This script fits one more
registry model on the REAL training feature table with its default (untuned)
configuration and exports it to ``artifacts/classifiers/<model_key>/`` using the
same model-card format, so the evaluator UI can switch classifiers live.

    python -m feature_engineering.pipeline --partition train        # builds the table
    python -m model_training.export_classifier --model logistic_regression \
        --train-features feature_engineering/results/train_features.csv
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from data_processing.config import ARTIFACTS_DIR
from hyperparameter_optimisation.inference import export_final_model_artifacts
from model_training.models import build_model_registry, get_model_pipeline
from model_training.schema import LABEL_COLUMN, load_feature_table, validate_and_prepare_features

CLASSIFIERS_DIR = ARTIFACTS_DIR / "classifiers"


def export_classifier(model_key: str, train_features: Path, seed: int = 42,
                      out_root: Path = CLASSIFIERS_DIR) -> Path:
    raw = load_feature_table(train_features)
    if "is_mock" in raw.columns and raw["is_mock"].any():
        raise SystemExit("refusing to export a classifier trained on mock rows")
    df, feats = validate_and_prepare_features(raw, source_path=train_features)
    spec, pipeline = get_model_pipeline(model_key, seed=seed)
    start = time.perf_counter()
    pipeline.fit(df[feats], df[LABEL_COLUMN])
    fit_s = time.perf_counter() - start
    out_dir = out_root / model_key
    export_final_model_artifacts(
        pipeline, model_key=model_key, model_name=spec.display_name,
        algorithm_family=spec.algorithm_family, feature_columns=feats,
        best_params=dict(spec.default_params),
        tuned_vs_untuned_summary={"note": "untuned registry defaults; the tuned final model is in artifacts/"},
        computational_cost={"fit_seconds": round(fit_s, 2), "n_training_rows": int(len(df)),
                            "n_training_trips": int(df["trip_id"].nunique())},
        seed=seed, is_mock=False, artifacts_dir=out_dir)
    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export an extra selectable classifier.")
    parser.add_argument("--model", required=True, choices=sorted(build_model_registry()))
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    out = export_classifier(args.model, args.train_features, args.seed)
    print(f"exported {args.model} to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
