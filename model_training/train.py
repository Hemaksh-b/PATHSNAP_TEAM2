"""Group-aware cross-validation training harness for candidate-classification models (Issue #4).

Usage examples:
    # 1. Run all models on the synthetic harness (before Member 1's #3 real features are merged):
    python -m model_training.train --all --mock

    # 2. Run a single model on the synthetic harness:
    python -m model_training.train --model random_forest --mock

    # 3. Run all models on Member 1's real candidate-feature matrix (#3):
    python -m model_training.train --all --features-path feature_engineering/results/train_features.csv
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

from .mock_data import generate_mock_feature_table
from .models import ModelSpec, build_model_registry
from .schema import (
    LABEL_COLUMN,
    load_feature_table,
    validate_and_prepare_features,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def evaluate_top1_observation_accuracy(
    eval_df: pd.DataFrame,
    probs: np.ndarray,
) -> Dict[str, float]:
    """Compute observation-level Top-1 accuracy and Candidate Miss Rate.

    For each `(trip_id, obs_idx)` observation, selects the candidate arc with the highest
    predicted probability `argmax(prob)` and checks whether its ground-truth `label == 1`.
    If an observation has no candidate with `label == 1` in the candidate set, it is counted
    as a candidate miss (and therefore also fails Top-1 route matching).
    """
    scored = eval_df[["trip_id", "obs_idx", "arc_id", LABEL_COLUMN]].copy()
    scored["prob"] = np.asarray(probs, dtype=float)

    # Sort deterministically so ties break by original row order (cand_rank)
    scored["_row_order"] = np.arange(len(scored))
    scored_sorted = scored.sort_values(
        by=["trip_id", "obs_idx", "prob", "_row_order"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )

    top1_per_obs = scored_sorted.drop_duplicates(subset=["trip_id", "obs_idx"], keep="first")
    obs_has_true = scored.groupby(["trip_id", "obs_idx"], sort=False)[LABEL_COLUMN].max()

    n_obs = int(len(top1_per_obs))
    if n_obs == 0:
        return {
            "n_observations": 0,
            "top1_accuracy": 0.0,
            "top1_accuracy_given_hit": 0.0,
            "candidate_miss_rate": 0.0,
        }

    n_hits_in_cands = int((obs_has_true > 0).sum())
    miss_rate = float(1.0 - (n_hits_in_cands / n_obs))
    top1_correct = int((top1_per_obs[LABEL_COLUMN] == 1).sum())
    top1_acc = float(top1_correct / n_obs)
    top1_given_hit = float(top1_correct / n_hits_in_cands) if n_hits_in_cands > 0 else 0.0

    return {
        "n_observations": n_obs,
        "top1_accuracy": top1_acc,
        "top1_accuracy_given_hit": top1_given_hit,
        "candidate_miss_rate": miss_rate,
    }


def verify_group_kfold_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create GroupKFold splits grouped by `trip_id` and assert zero trip overlap per fold."""
    unique_trips = df["trip_id"].unique()
    if len(unique_trips) < n_splits:
        raise ValueError(
            f"GroupKFold requires at least n_splits={n_splits} distinct trips, "
            f"found {len(unique_trips)}."
        )

    gkf = GroupKFold(n_splits=n_splits)
    groups = df["trip_id"].to_numpy()
    splits: List[Tuple[np.ndarray, np.ndarray]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(df, df[LABEL_COLUMN], groups=groups)):
        train_trips = set(groups[train_idx])
        val_trips = set(groups[val_idx])
        overlap = train_trips & val_trips
        assert not overlap, (
            f"GroupKFold leakage detected in fold {fold_idx}: "
            f"overlapping trip_ids = {sorted(overlap)}"
        )
        assert len(train_idx) > 0 and len(val_idx) > 0, f"Empty split in fold {fold_idx}"
        splits.append((train_idx, val_idx))

    return splits


def _extract_positive_probabilities(pipeline: Any, X_val: pd.DataFrame) -> np.ndarray:
    """Extract calibrated or estimated positive-class probabilities P(label=1 | x)."""
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X_val)
        return np.asarray(proba[:, 1], dtype=float)
    if hasattr(pipeline, "decision_function"):
        scores = np.asarray(pipeline.decision_function(X_val), dtype=float)
        return 1.0 / (1.0 + np.exp(-np.clip(scores, -50.0, 50.0)))
    return np.asarray(pipeline.predict(X_val), dtype=float)


def run_single_model_cv(
    spec: ModelSpec,
    df: pd.DataFrame,
    feature_cols: List[str],
    splits: List[Tuple[np.ndarray, np.ndarray]],
    seed: int = 42,
    is_mock: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], pd.DataFrame]:
    """Train and cross-validate a single model across all GroupKFold splits.

    Returns:
        fold_records: per-fold metric dictionaries
        summary_record: mean/std aggregated CV record for this model
        oof_predictions: out-of-fold candidate probabilities (`trip_id, obs_idx, arc_id, label, prob, fold`)
    """
    fold_records: List[Dict[str, Any]] = []
    oof_chunks: List[pd.DataFrame] = []
    data_source_tag = "MOCK_SYNTHETIC_DO_NOT_REPORT" if is_mock else "REAL_CANDIDATE_FEATURES"

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        train_sub = df.iloc[train_idx]
        val_sub = df.iloc[val_idx]

        # Re-assert strict trip-level isolation inside training loop
        train_trips = set(train_sub["trip_id"].unique())
        val_trips = set(val_sub["trip_id"].unique())
        assert train_trips.isdisjoint(val_trips), "Trip leakage between train and validation fold!"

        X_train = train_sub[feature_cols]
        y_train = train_sub[LABEL_COLUMN].to_numpy()
        X_val = val_sub[feature_cols]
        y_val = val_sub[LABEL_COLUMN].to_numpy()

        pipeline = spec.builder(seed)

        t0 = time.perf_counter()
        pipeline.fit(X_train, y_train)
        fit_time_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        val_probs = _extract_positive_probabilities(pipeline, X_val)
        val_preds = (val_probs >= 0.5).astype(int)
        infer_time_s = time.perf_counter() - t1

        clipped_probs = np.clip(val_probs, 1e-7, 1.0 - 1e-7)
        top1_stats = evaluate_top1_observation_accuracy(val_sub, val_probs)

        n_val_trips = max(1, len(val_trips))
        n_val_obs = max(1, top1_stats["n_observations"])

        fold_metrics: Dict[str, Any] = {
            "model_key": spec.key,
            "model_name": spec.display_name,
            "algorithm_family": spec.algorithm_family,
            "fold": fold_idx,
            "seed": seed,
            "data_source": data_source_tag,
            "n_train_trips": len(train_trips),
            "n_val_trips": len(val_trips),
            "n_train_rows": len(train_sub),
            "n_val_rows": len(val_sub),
            "n_val_observations": top1_stats["n_observations"],
            "macro_f1": float(f1_score(y_val, val_preds, average="macro", zero_division=0)),
            "binary_f1": float(f1_score(y_val, val_preds, pos_label=1, zero_division=0)),
            "accuracy": float(accuracy_score(y_val, val_preds)),
            "precision": float(precision_score(y_val, val_preds, pos_label=1, zero_division=0)),
            "recall": float(recall_score(y_val, val_preds, pos_label=1, zero_division=0)),
            "log_loss": float(log_loss(y_val, clipped_probs, labels=[0, 1])),
            "roc_auc": float(roc_auc_score(y_val, val_probs)) if len(np.unique(y_val)) > 1 else float("nan"),
            "top1_accuracy": top1_stats["top1_accuracy"],
            "top1_accuracy_given_hit": top1_stats["top1_accuracy_given_hit"],
            "candidate_miss_rate": top1_stats["candidate_miss_rate"],
            "fit_time_s": round(fit_time_s, 5),
            "infer_time_s": round(infer_time_s, 5),
            "infer_ms_per_trip": round((infer_time_s * 1000.0) / n_val_trips, 4),
            "infer_us_per_obs": round((infer_time_s * 1e6) / n_val_obs, 4),
        }
        fold_records.append(fold_metrics)

        oof_df = val_sub[["trip_id", "obs_idx", "arc_id", LABEL_COLUMN]].copy()
        oof_df["prob"] = val_probs
        oof_df["pred"] = val_preds
        oof_df["fold"] = fold_idx
        oof_chunks.append(oof_df)

    folds_df = pd.DataFrame(fold_records)
    oof_all = pd.concat(oof_chunks, axis=0, ignore_index=True)

    summary_record: Dict[str, Any] = {
        "model_key": spec.key,
        "model_name": spec.display_name,
        "algorithm_family": spec.algorithm_family,
        "requires_scaling": spec.requires_scaling,
        "imbalance_handling": spec.imbalance_handling,
        "seed": seed,
        "n_splits": len(splits),
        "data_source": data_source_tag,
        "cv_macro_f1_mean": round(float(folds_df["macro_f1"].mean()), 5),
        "cv_macro_f1_std": round(float(folds_df["macro_f1"].std(ddof=1)), 5),
        "cv_binary_f1_mean": round(float(folds_df["binary_f1"].mean()), 5),
        "cv_binary_f1_std": round(float(folds_df["binary_f1"].std(ddof=1)), 5),
        "cv_top1_accuracy_mean": round(float(folds_df["top1_accuracy"].mean()), 5),
        "cv_top1_accuracy_std": round(float(folds_df["top1_accuracy"].std(ddof=1)), 5),
        "cv_top1_given_hit_mean": round(float(folds_df["top1_accuracy_given_hit"].mean()), 5),
        "cv_accuracy_mean": round(float(folds_df["accuracy"].mean()), 5),
        "cv_precision_mean": round(float(folds_df["precision"].mean()), 5),
        "cv_recall_mean": round(float(folds_df["recall"].mean()), 5),
        "cv_log_loss_mean": round(float(folds_df["log_loss"].mean()), 5),
        "cv_log_loss_std": round(float(folds_df["log_loss"].std(ddof=1)), 5),
        "cv_roc_auc_mean": round(float(folds_df["roc_auc"].mean()), 5),
        "candidate_miss_rate_mean": round(float(folds_df["candidate_miss_rate"].mean()), 5),
        "fit_time_s_mean": round(float(folds_df["fit_time_s"].mean()), 5),
        "fit_time_s_total": round(float(folds_df["fit_time_s"].sum()), 5),
        "infer_time_s_mean": round(float(folds_df["infer_time_s"].mean()), 5),
        "infer_ms_per_trip_mean": round(float(folds_df["infer_ms_per_trip"].mean()), 4),
        "infer_us_per_obs_mean": round(float(folds_df["infer_us_per_obs"].mean()), 4),
        "default_params": json.dumps(spec.default_params, sort_keys=True),
    }

    return fold_records, summary_record, oof_all


def run_group_cv_experiment(
    df: pd.DataFrame,
    model_keys: Optional[Iterable[str]] = None,
    feature_columns: Optional[Iterable[str]] = None,
    n_splits: int = 5,
    seed: int = 42,
    drop_uncertain_labels: bool = True,
    output_dir: Optional[Path] = RESULTS_DIR,
    source_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run GroupKFold cross-validation across the selected models and persist experiment logs."""
    is_mock = bool(df["is_mock"].any()) if "is_mock" in df.columns else False
    clean_df, resolved_features = validate_and_prepare_features(
        df,
        feature_columns=feature_columns,
        drop_uncertain_labels=drop_uncertain_labels,
        source_path=source_path,
    )

    splits = verify_group_kfold_splits(clean_df, n_splits=n_splits)
    registry = build_model_registry(seed=seed)

    if model_keys is None:
        selected_keys = list(registry.keys())
    else:
        selected_keys = list(model_keys)
        unknown = [k for k in selected_keys if k not in registry]
        if unknown:
            raise KeyError(f"Unknown model key(s): {unknown}. Valid: {sorted(registry.keys())}")

    all_fold_records: List[Dict[str, Any]] = []
    summary_records: List[Dict[str, Any]] = []

    for key in selected_keys:
        spec = registry[key]
        fold_recs, summary_rec, _ = run_single_model_cv(
            spec=spec,
            df=clean_df,
            feature_cols=resolved_features,
            splits=splits,
            seed=seed,
            is_mock=is_mock,
        )
        all_fold_records.extend(fold_recs)
        summary_records.append(summary_rec)

    folds_df = pd.DataFrame(all_fold_records)
    summary_df = pd.DataFrame(summary_records).sort_values(
        by=["cv_top1_accuracy_mean", "cv_macro_f1_mean"],
        ascending=[False, False],
    ).reset_index(drop=True)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        prefix = "mock_harness" if is_mock else "cv"
        folds_path = output_dir / f"{prefix}_per_fold_metrics.csv"
        summary_path = output_dir / f"{prefix}_summary_metrics.csv"
        meta_path = output_dir / f"{prefix}_experiment_manifest.json"

        folds_df.to_csv(folds_path, index=False)
        summary_df.to_csv(summary_path, index=False)

        manifest = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "is_mock_data": is_mock,
            "warning": (
                "SYNTHETIC MOCK DATA FOR HARNESS VERIFICATION ONLY — DO NOT REPORT IN DELIVERABLES."
                if is_mock
                else "Trained on real PathSnap candidate features."
            ),
            "source_path": str(source_path) if source_path else "mock_generator",
            "seed": seed,
            "n_splits": n_splits,
            "n_trips": int(clean_df["trip_id"].nunique()),
            "n_rows": int(len(clean_df)),
            "positive_class_rate": round(float(clean_df[LABEL_COLUMN].mean()), 5),
            "feature_columns": resolved_features,
            "models_trained": [
                {
                    "model_key": k,
                    "display_name": registry[k].display_name,
                    "algorithm_family": registry[k].algorithm_family,
                    "requires_scaling": registry[k].requires_scaling,
                    "imbalance_handling": registry[k].imbalance_handling,
                    "default_params": registry[k].default_params,
                }
                for k in selected_keys
            ],
        }
        meta_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return summary_df, folds_df


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and cross-validate PathSnap candidate-classification models (Issue #4)."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--all",
        action="store_true",
        help="Train and evaluate all registered models (>= 5 distinct algorithm families).",
    )
    group.add_argument(
        "--model",
        type=str,
        nargs="+",
        help="One or more specific model keys to train (e.g. logistic_regression random_forest).",
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--features-path",
        type=Path,
        help="Path to Member 1's candidate feature CSV/Parquet file (#3).",
    )
    source.add_argument(
        "--mock",
        action="store_true",
        help="Run the training harness on clearly flagged synthetic mock data.",
    )

    parser.add_argument("--n-splits", type=int, default=5, help="Number of GroupKFold splits (default: 5).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42).")
    parser.add_argument(
        "--mock-trips",
        type=int,
        default=80,
        help="Number of synthetic trips when running with --mock (default: 80).",
    )
    parser.add_argument(
        "--mock-obs-per-trip",
        type=int,
        default=50,
        help="Observations per synthetic trip when running with --mock (default: 50).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory to write per-fold and summary CSV/JSON logs.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.mock:
        print(
            "[WARNING] Running in --mock mode with synthetic data for harness verification only. "
            "Never report mock numbers in the technical report."
        )
        df = generate_mock_feature_table(
            n_trips=args.mock_trips,
            obs_per_trip=args.mock_obs_per_trip,
            seed=args.seed,
        )
        source_path = None
    else:
        source_path = args.features_path
        df = load_feature_table(source_path)

    model_keys = None if args.all else args.model
    summary_df, _ = run_group_cv_experiment(
        df=df,
        model_keys=model_keys,
        n_splits=args.n_splits,
        seed=args.seed,
        output_dir=args.output_dir,
        source_path=source_path,
    )

    display_cols = [
        "model_key",
        "algorithm_family",
        "cv_top1_accuracy_mean",
        "cv_macro_f1_mean",
        "cv_binary_f1_mean",
        "cv_log_loss_mean",
        "fit_time_s_mean",
        "infer_ms_per_trip_mean",
        "data_source",
    ]
    print("\n=== GroupKFold Cross-Validation Summary ===")
    print(summary_df[display_cols].to_string(index=False))
    print(f"\nSaved experiment results to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
