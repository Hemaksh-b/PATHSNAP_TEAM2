"""End-to-end model evaluation, comparison, selection, plotting, and HMM score export (Issue #5).

Usage examples:
    # 1. Run full evaluation (5-fold CV on 80 train trips + evaluation on 10 student-test trips) in mock harness mode:
    python -m model_evaluation.evaluate --mock

    # 2. Run full evaluation on Member 1's real feature tables (#3):
    python -m model_evaluation.evaluate \
        --train-features feature_engineering/results/train_features.csv \
        --student-test-features feature_engineering/results/student_test_features.csv
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

from model_training.mock_data import generate_mock_feature_table
from model_training.models import ModelSpec, build_model_registry
from model_training.schema import (
    LABEL_COLUMN,
    load_feature_table,
    validate_and_prepare_features,
)
from model_training.train import (
    _extract_positive_probabilities,
    run_single_model_cv,
    verify_group_kfold_splits,
)

from .hmm_bridge import (
    build_candidate_scores_table,
    verify_hmm_decoder_compatibility,
)
from .metrics import compute_all_evaluation_metrics, compute_route_continuity
from .plots import FIGURES_DIR, generate_all_comparison_plots

EVAL_RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _attach_synthetic_topology(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Attach realistic `u` and `v` node IDs to synthetic candidate rows so Route Continuity is testable.

    True candidates (`label == 1`) along consecutive observations form a connected chain of road
    nodes `n_0 -> n_1 -> n_2 ...` (or stay on the same arc for several observations), while false
    candidates (`label == 0`) lie on side/parallel streets that are usually disconnected.
    """
    if "u" in df.columns and "v" in df.columns:
        return df

    rng = np.random.default_rng(seed)
    out = df.copy()
    u_list = np.zeros(len(out), dtype=int)
    v_list = np.zeros(len(out), dtype=int)
    arc_ids = out["arc_id"].to_numpy(dtype=int).copy()

    for trip_idx, (_, trip_group) in enumerate(out.groupby("trip_id", sort=False)):
        base_node = (trip_idx + 1) * 10000
        base_arc = (trip_idx + 1) * 50000
        curr_seg = 0
        curr_u = base_node
        curr_v = base_node + 1
        curr_true_arc = base_arc

        for obs_idx, obs_group in trip_group.groupby("obs_idx", sort=True):
            # Advance to next connected road segment every ~4 observations
            if int(obs_idx) > 0 and int(obs_idx) % 4 == 0:
                curr_seg += 1
                curr_u = base_node + curr_seg
                curr_v = base_node + curr_seg + 1
                curr_true_arc = base_arc + curr_seg

            for row_pos, row in obs_group.iterrows():
                if int(row[LABEL_COLUMN]) == 1:
                    u_list[row_pos] = curr_u
                    v_list[row_pos] = curr_v
                    arc_ids[row_pos] = curr_true_arc
                else:
                    # Side or parallel road segment (only occasionally sharing an intersection node)
                    if rng.random() < 0.15:
                        u_list[row_pos] = curr_v
                        v_list[row_pos] = int(rng.integers(base_node + 500, base_node + 9000))
                    else:
                        u_list[row_pos] = int(rng.integers(base_node + 500, base_node + 9000))
                        v_list[row_pos] = int(rng.integers(base_node + 500, base_node + 9000))

    out["arc_id"] = arc_ids
    out["u"] = u_list
    out["v"] = v_list
    return out


def generate_mock_train_and_student_test(
    n_train_trips: int = 80,
    n_student_test_trips: int = 10,
    obs_per_trip: int = 40,
    k_candidates: int = 5,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic 80-trip training and 10-trip student-test tables for harness testing."""
    train_df = generate_mock_feature_table(
        n_trips=n_train_trips,
        obs_per_trip=obs_per_trip,
        k_candidates=k_candidates,
        seed=seed,
    )
    train_df = _attach_synthetic_topology(train_df, seed=seed)

    st_df = generate_mock_feature_table(
        n_trips=n_student_test_trips,
        obs_per_trip=obs_per_trip,
        k_candidates=k_candidates,
        seed=seed + 999,
    )
    # Rename trip_ids to match student_test partition naming
    st_trip_map = {
        f"mock_train_{i:03d}": f"mock_student_test_{i:03d}"
        for i in range(n_student_test_trips)
    }
    st_df["trip_id"] = st_df["trip_id"].map(st_trip_map)
    st_df["partition"] = "student_test"
    st_df = _attach_synthetic_topology(st_df, seed=seed + 999)
    return train_df, st_df


def _format_key_configuration(spec: ModelSpec) -> str:
    """Format concise human-readable key configuration string for Report §3.1 table."""
    p = spec.default_params
    if spec.key == "logistic_regression":
        return f"L2 penalty, C={p['C']}, solver={p['solver']}, class_weight=balanced, StandardScaler"
    if spec.key == "knn_distance":
        return f"k={p['n_neighbors']}, weights={p['weights']}, Euclidean (p=2), StandardScaler"
    if spec.key == "decision_tree":
        return f"criterion={p['criterion']}, max_depth={p['max_depth']}, min_samples_leaf={p['min_samples_leaf']}, balanced"
    if spec.key == "random_forest":
        return (
            f"n_estimators={p['n_estimators']}, max_depth={p['max_depth']}, "
            f"min_samples_leaf={p['min_samples_leaf']}, balanced_subsample"
        )
    if spec.key == "mlp_neural_net":
        return (
            f"hidden_layers={p['hidden_layer_sizes']}, activation={p['activation']}, "
            f"alpha={p['alpha']}, early_stopping=True, StandardScaler"
        )
    return json.dumps(p)


def _compute_selection_score(row: Mapping[str, Any]) -> float:
    """Composite evidence score used to rank candidate classifiers for HMM route reconstruction.

    Weighting rationale (documented in SELECTION_RATIONALE.md):
    - 40% Student-Test Observation-level Top-1 Accuracy (direct route-matching objective on unseen trips)
    - 25% 5-Fold GroupKFold CV Macro F1-score (generalization across all 80 training trips under 1:K imbalance)
    - 20% Student-Test Macro F1-score (candidate classification quality on unseen trips)
    - 15% Probability Calibration term `1 / (1 + student_test_log_loss)` (critical for HMM emission log-probs)
    """
    st_top1 = float(row["student_test_top1_accuracy"])
    cv_f1 = float(row["cv_macro_f1_mean"])
    st_f1 = float(row["student_test_macro_f1"])
    calib = 1.0 / (1.0 + float(row["student_test_log_loss"]))
    return 0.40 * st_top1 + 0.25 * cv_f1 + 0.20 * st_f1 + 0.15 * calib


def evaluate_all_models(
    train_df: pd.DataFrame,
    student_test_df: pd.DataFrame,
    model_keys: Optional[Iterable[str]] = None,
    feature_columns: Optional[Iterable[str]] = None,
    n_splits: int = 5,
    seed: int = 42,
    output_dir: Path = EVAL_RESULTS_DIR,
    figures_dir: Path = FIGURES_DIR,
    train_source_path: Optional[Path] = None,
    student_test_source_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run GroupKFold CV on training trips + full evaluation on student-test trips for all 5 models."""
    is_mock = (
        (bool(train_df["is_mock"].any()) if "is_mock" in train_df.columns else False)
        or (bool(student_test_df["is_mock"].any()) if "is_mock" in student_test_df.columns else False)
    )
    data_source_tag = "MOCK_SYNTHETIC_DO_NOT_REPORT" if is_mock else "REAL_CANDIDATE_FEATURES"

    # For training, drop uncertain labels; for evaluation on student-test, keep all observations
    clean_train, resolved_features = validate_and_prepare_features(
        train_df,
        feature_columns=feature_columns,
        drop_uncertain_labels=True,
        source_path=train_source_path,
    )
    clean_st, _ = validate_and_prepare_features(
        student_test_df,
        feature_columns=resolved_features,
        drop_uncertain_labels=False,
        source_path=student_test_source_path,
    )

    # Assert zero trip overlap between training and student-testing partitions
    train_trips = set(clean_train["trip_id"].astype(str).unique())
    st_trips = set(clean_st["trip_id"].astype(str).unique())
    overlap_partitions = train_trips & st_trips
    if overlap_partitions:
        raise ValueError(
            f"Data leakage violation: trip_ids shared between train and student_test: {sorted(overlap_partitions)}"
        )

    splits = verify_group_kfold_splits(clean_train, n_splits=n_splits)
    registry = build_model_registry(seed=seed)
    selected_keys = list(model_keys) if model_keys is not None else list(registry.keys())

    cv_fold_rows: List[Dict[str, Any]] = []
    comparison_rows: List[Dict[str, Any]] = []
    st_per_trip_rows: List[Dict[str, Any]] = []
    confusion_records: Dict[str, Dict[str, int]] = {}
    oof_probs_by_model: Dict[str, pd.DataFrame] = {}
    st_probs_by_model: Dict[str, np.ndarray] = {}
    mlp_history: Optional[Dict[str, Any]] = None

    X_train_full = clean_train[resolved_features]
    y_train_full = clean_train[LABEL_COLUMN].to_numpy(dtype=int)
    X_st = clean_st[resolved_features]

    for key in selected_keys:
        spec = registry[key]
        # 1. 5-fold GroupKFold on the 80 training trips
        fold_recs, cv_summary, oof_df = run_single_model_cv(
            spec=spec,
            df=clean_train,
            feature_cols=resolved_features,
            splits=splits,
            seed=seed,
            is_mock=is_mock,
        )
        cv_fold_rows.extend(fold_recs)
        oof_probs_by_model[key] = oof_df

        # Compute OOF Route Continuity on training trips
        oof_aligned = clean_train.merge(
            oof_df[["trip_id", "obs_idx", "arc_id", "prob"]],
            on=["trip_id", "obs_idx", "arc_id"],
            how="left",
        )
        cv_continuity = compute_route_continuity(oof_aligned, oof_aligned["prob"].to_numpy())

        # 2. Fit on all 80 training trips and evaluate on the 10 student-testing trips
        pipeline = spec.builder(seed)
        t0 = time.perf_counter()
        pipeline.fit(X_train_full, y_train_full)
        full_fit_time_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        st_probs = _extract_positive_probabilities(pipeline, X_st)
        st_infer_time_s = time.perf_counter() - t1
        st_probs_by_model[key] = st_probs

        # Capture genuine epoch-wise training history for MLP only (Deliverables.pdf §3.3)
        clf_step = pipeline.named_steps.get("clf")
        if key == "mlp_neural_net" and hasattr(clf_step, "loss_curve_"):
            mlp_history = {
                "n_iter": int(getattr(clf_step, "n_iter_", len(clf_step.loss_curve_))),
                "loss_curve": [round(float(v), 6) for v in clf_step.loss_curve_],
                "validation_scores": [
                    round(float(v), 6) for v in (getattr(clf_step, "validation_scores_", None) or [])
                ],
            }

        st_metrics = compute_all_evaluation_metrics(clean_st, st_probs)
        n_st_trips = max(1, st_metrics["n_trips"])
        n_st_obs = max(1, st_metrics["n_observations"])
        infer_ms_per_trip = (st_infer_time_s * 1000.0) / n_st_trips
        infer_us_per_obs = (st_infer_time_s * 1e6) / n_st_obs

        confusion_records[key] = {
            "tn": st_metrics["tn"],
            "fp": st_metrics["fp"],
            "fn": st_metrics["fn"],
            "tp": st_metrics["tp"],
        }

        # Per-trip breakdown on student-test trips
        for trip_id, trip_group in clean_st.groupby("trip_id", sort=True):
            trip_mask = (clean_st["trip_id"] == trip_id).to_numpy()
            trip_m = compute_all_evaluation_metrics(trip_group, st_probs[trip_mask])
            st_per_trip_rows.append(
                {
                    "model_key": key,
                    "model_name": spec.display_name,
                    "trip_id": str(trip_id),
                    "n_observations": trip_m["n_observations"],
                    "top1_accuracy": round(trip_m["top1_accuracy"], 5),
                    "macro_f1": round(trip_m["macro_f1"], 5),
                    "binary_f1": round(trip_m["binary_f1"], 5),
                    "precision": round(trip_m["precision"], 5),
                    "recall": round(trip_m["recall"], 5),
                    "log_loss": round(trip_m["log_loss"], 5),
                    "route_continuity": round(trip_m["route_continuity"], 5),
                    "candidate_miss_rate": round(trip_m["candidate_miss_rate"], 5),
                    "data_source": data_source_tag,
                }
            )

        row_record: Dict[str, Any] = {
            # Exact Report §3.1 Table Columns:
            "Model": spec.display_name,
            "Algorithm Family": spec.algorithm_family,
            "Key Configuration": _format_key_configuration(spec),
            "CV Macro F1-score": f"{cv_summary['cv_macro_f1_mean']:.4f} ± {cv_summary['cv_macro_f1_std']:.4f}",
            "Top-1 Accuracy": round(st_metrics["top1_accuracy"], 4),
            "Route Continuity": round(st_metrics["route_continuity"], 4),
            "Student-Test Macro F1-score": round(st_metrics["macro_f1"], 4),
            "Log Loss": round(st_metrics["log_loss"], 4),
            "Inference Time": f"{infer_ms_per_trip:.2f} ms/trip ({infer_us_per_obs:.1f} us/obs)",
            # Detailed numeric columns for downstream scripts (#6, #9, #12):
            "model_key": key,
            "cv_macro_f1_mean": round(cv_summary["cv_macro_f1_mean"], 5),
            "cv_macro_f1_std": round(cv_summary["cv_macro_f1_std"], 5),
            "cv_binary_f1_mean": round(cv_summary["cv_binary_f1_mean"], 5),
            "cv_top1_accuracy_mean": round(cv_summary["cv_top1_accuracy_mean"], 5),
            "cv_top1_accuracy_std": round(cv_summary["cv_top1_accuracy_std"], 5),
            "cv_precision_mean": round(cv_summary["cv_precision_mean"], 5),
            "cv_recall_mean": round(cv_summary["cv_recall_mean"], 5),
            "cv_accuracy_mean": round(cv_summary["cv_accuracy_mean"], 5),
            "cv_log_loss_mean": round(cv_summary["cv_log_loss_mean"], 5),
            "cv_route_continuity": round(cv_continuity, 5),
            "student_test_top1_accuracy": round(st_metrics["top1_accuracy"], 5),
            "student_test_top1_given_hit": round(st_metrics["top1_accuracy_given_hit"], 5),
            "student_test_macro_f1": round(st_metrics["macro_f1"], 5),
            "student_test_binary_f1": round(st_metrics["binary_f1"], 5),
            "student_test_precision": round(st_metrics["precision"], 5),
            "student_test_recall": round(st_metrics["recall"], 5),
            "student_test_accuracy": round(st_metrics["accuracy"], 5),
            "student_test_log_loss": round(st_metrics["log_loss"], 5),
            "student_test_roc_auc": round(st_metrics["roc_auc"], 5),
            "route_continuity": round(st_metrics["route_continuity"], 5),
            "candidate_miss_rate": round(st_metrics["candidate_miss_rate"], 5),
            "fit_time_s": round(full_fit_time_s, 5),
            "infer_time_s": round(st_infer_time_s, 5),
            "infer_ms_per_trip": round(infer_ms_per_trip, 4),
            "infer_us_per_obs": round(infer_us_per_obs, 4),
            "data_source": data_source_tag,
        }
        row_record["selection_score"] = round(_compute_selection_score(row_record), 6)
        comparison_rows.append(row_record)

    comparison_df = (
        pd.DataFrame(comparison_rows)
        .sort_values(
            by=["selection_score", "student_test_top1_accuracy", "cv_macro_f1_mean"],
            ascending=[False, False, False],
        )
        .reset_index(drop=True)
    )
    best_model_key = str(comparison_df.loc[0, "model_key"])
    comparison_df["selected_final_model"] = comparison_df["model_key"] == best_model_key

    # Export per-candidate probability tables (`trip_id, obs_idx, arc_id, prob`) for Member 3 (#7/#8/#9)
    best_oof = oof_probs_by_model[best_model_key]
    train_Merge = clean_train[["trip_id", "obs_idx", "arc_id"]].merge(
        best_oof[["trip_id", "obs_idx", "arc_id", "prob"]],
        on=["trip_id", "obs_idx", "arc_id"],
        how="left",
    )
    train_scores_df = build_candidate_scores_table(
        train_Merge,
        train_Merge["prob"].to_numpy(),
    )
    st_scores_df = build_candidate_scores_table(
        clean_st,
        st_probs_by_model[best_model_key],
    )

    # Verify compatibility with Member 3's hmm_viterbi decoder
    hmm_decode_check = verify_hmm_decoder_compatibility(st_scores_df)

    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_csv_path = output_dir / "model_comparison.csv"
    cv_folds_csv_path = output_dir / "per_fold_cv_metrics.csv"
    st_per_trip_csv_path = output_dir / "student_test_per_trip_metrics.csv"
    train_scores_path = output_dir / "candidate_scores_train.csv"
    st_scores_path = output_dir / "candidate_scores_student_test.csv"
    cm_json_path = output_dir / "confusion_matrices.json"
    manifest_path = output_dir / "evaluation_manifest.json"

    comparison_df.to_csv(comparison_csv_path, index=False)
    pd.DataFrame(cv_fold_rows).to_csv(cv_folds_csv_path, index=False)
    pd.DataFrame(st_per_trip_rows).to_csv(st_per_trip_csv_path, index=False)
    train_scores_df.to_csv(train_scores_path, index=False)
    st_scores_df.to_csv(st_scores_path, index=False)
    cm_json_path.write_text(json.dumps(confusion_records, indent=2), encoding="utf-8")

    # Generate all comparison plots
    figure_paths = generate_all_comparison_plots(
        comparison_df=comparison_df,
        confusion_records=confusion_records,
        mlp_history=mlp_history,
        figures_dir=figures_dir,
        is_mock=is_mock,
    )

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "is_mock_data": is_mock,
        "data_source": data_source_tag,
        "seed": seed,
        "n_splits": n_splits,
        "n_train_trips": int(clean_train["trip_id"].nunique()),
        "n_student_test_trips": int(clean_st["trip_id"].nunique()),
        "feature_columns": resolved_features,
        "selected_best_model_key": best_model_key,
        "selected_best_model_name": str(comparison_df.loc[0, "Model"]),
        "primary_selection_metric": (
            "Composite of Student-Test Observation-level Top-1 Accuracy (40%), "
            "5-Fold GroupKFold Macro F1 (25%), Student-Test Macro F1 (20%), "
            "and Probability Calibration Log Loss (15%)."
        ),
        "hmm_decoder_compatibility_verified": True,
        "hmm_sample_decoded_observations": len(hmm_decode_check.arc_ids),
        "mlp_training_history": mlp_history,
        "generated_files": {
            "model_comparison_csv": str(comparison_csv_path),
            "per_fold_cv_metrics_csv": str(cv_folds_csv_path),
            "student_test_per_trip_metrics_csv": str(st_per_trip_csv_path),
            "candidate_scores_train_csv": str(train_scores_path),
            "candidate_scores_student_test_csv": str(st_scores_path),
            "confusion_matrices_json": str(cm_json_path),
            "figures": {k: str(v) for k, v in figure_paths.items()},
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "comparison_df": comparison_df,
        "train_scores_df": train_scores_df,
        "student_test_scores_df": st_scores_df,
        "confusion_records": confusion_records,
        "selected_best_model_key": best_model_key,
        "figure_paths": figure_paths,
        "manifest": manifest,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate all 5 PathSnap candidate classifiers on CV and student-test trips (Issue #5)."
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run evaluation on clearly-flagged synthetic 80-train + 10-student-test tables.",
    )
    parser.add_argument(
        "--train-features",
        type=Path,
        help="Path to Member 1's training candidate feature CSV/Parquet (#3).",
    )
    parser.add_argument(
        "--student-test-features",
        type=Path,
        help="Path to Member 1's student-testing candidate feature CSV/Parquet (#3).",
    )
    parser.add_argument("--n-splits", type=int, default=5, help="GroupKFold splits (default: 5).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--mock-train-trips", type=int, default=80)
    parser.add_argument("--mock-student-test-trips", type=int, default=10)
    parser.add_argument("--mock-obs-per-trip", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=EVAL_RESULTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.mock:
        print(
            "[WARNING] Running Issue #5 evaluation in --mock mode for pipeline verification. "
            "Re-run with --train-features and --student-test-features once #3 merges."
        )
        train_df, st_df = generate_mock_train_and_student_test(
            n_train_trips=args.mock_train_trips,
            n_student_test_trips=args.mock_student_test_trips,
            obs_per_trip=args.mock_obs_per_trip,
            seed=args.seed,
        )
        train_src = None
        st_src = None
    else:
        if not args.train_features or not args.student_test_features:
            parser.error(
                "Must specify either --mock OR both --train-features and --student-test-features."
            )
        train_src = args.train_features
        st_src = args.student_test_features
        train_df = load_feature_table(train_src)
        st_df = load_feature_table(st_src)

    result = evaluate_all_models(
        train_df=train_df,
        student_test_df=st_df,
        n_splits=args.n_splits,
        seed=args.seed,
        output_dir=args.output_dir,
        figures_dir=args.figures_dir,
        train_source_path=train_src,
        student_test_source_path=st_src,
    )

    comp = result["comparison_df"]
    report_cols = [
        "Model",
        "Algorithm Family",
        "CV Macro F1-score",
        "Top-1 Accuracy",
        "Route Continuity",
        "Student-Test Macro F1-score",
        "Log Loss",
        "Inference Time",
        "selected_final_model",
    ]
    print("\n=== Report §3.1 Consolidated Model Comparison Table ===")
    print(comp[report_cols].to_string(index=False))
    print(f"\nSelected Best Candidate Classifier: {result['selected_best_model_key']}")
    print(f"Saved comparison CSV & HMM candidate score tables to: {args.output_dir}")
    print(f"Saved comparison plots to: {args.figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
