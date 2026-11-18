"""Bayesian hyperparameter optimization (TPE + GroupKFold CV) and final model export (Issue #6).

Usage examples:
    # 1. Run Bayesian optimization on the selected model in mock harness mode:
    python -m hyperparameter_optimisation.optimize --mock --n-trials 20 --seed 42

    # 2. Run Bayesian optimization on a specific model (e.g. random_forest or logistic_regression):
    python -m hyperparameter_optimisation.optimize --model random_forest --mock --n-trials 20

    # 3. Run Bayesian optimization on Member 1's real feature tables (#3):
    python -m hyperparameter_optimisation.optimize \
        --train-features feature_engineering/results/train_features.csv \
        --student-test-features feature_engineering/results/student_test_features.csv \
        --n-trials 30 --seed 42
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss

from data_processing.config import ARTIFACTS_DIR
from model_evaluation.evaluate import (
    EVAL_RESULTS_DIR,
    generate_mock_train_and_student_test,
)
from model_evaluation.metrics import compute_all_evaluation_metrics
from model_training.models import build_model_registry
from model_training.schema import (
    LABEL_COLUMN,
    load_feature_table,
    validate_and_prepare_features,
)
from model_training.train import (
    _extract_positive_probabilities,
    evaluate_top1_observation_accuracy,
    verify_group_kfold_splits,
)

from .inference import export_final_model_artifacts, predict_candidate_probs
from .search_space import (
    SEARCH_SPACE_CONFIG,
    build_pipeline_from_params,
    get_search_space_spec,
    sample_params_for_model,
)
from .tpe_engine import execute_tpe_study, trials_to_dataframe

OPT_RESULTS_DIR = Path(__file__).resolve().parent / "results"
OPT_FIGURES_DIR = Path(__file__).resolve().parent / "figures"


def _resolve_default_model_key() -> str:
    """Read `selected_best_model_key` from Issue #5's manifest if present; otherwise default to `random_forest`."""
    manifest_path = EVAL_RESULTS_DIR / "evaluation_manifest.json"
    if manifest_path.exists():
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            key = str(data.get("selected_best_model_key", "")).strip()
            if key in SEARCH_SPACE_CONFIG["models"]:
                return key
        except Exception:
            pass
    return "random_forest"


def _evaluate_params_group_cv(
    model_key: str,
    params: Dict[str, Any],
    df: pd.DataFrame,
    feature_cols: List[str],
    splits: List[Tuple[np.ndarray, np.ndarray]],
    seed: int = 42,
) -> Dict[str, float]:
    """Evaluate a hyperparameter configuration across 5-fold GroupKFold splits on training trips."""
    fold_macro_f1: List[float] = []
    fold_top1: List[float] = []
    fold_log_loss: List[float] = []
    fold_fit_s: List[float] = []

    for train_idx, val_idx in splits:
        train_sub = df.iloc[train_idx]
        val_sub = df.iloc[val_idx]

        X_tr = train_sub[feature_cols]
        y_tr = train_sub[LABEL_COLUMN].to_numpy(dtype=int)
        X_va = val_sub[feature_cols]
        y_va = val_sub[LABEL_COLUMN].to_numpy(dtype=int)

        pipe = build_pipeline_from_params(model_key, params, seed=seed)
        t0 = time.perf_counter()
        pipe.fit(X_tr, y_tr)
        fold_fit_s.append(time.perf_counter() - t0)

        probs = _extract_positive_probabilities(pipe, X_va)
        preds = (probs >= 0.5).astype(int)
        p_clip = np.clip(probs, 1e-7, 1.0 - 1e-7)

        top1_stats = evaluate_top1_observation_accuracy(val_sub, probs)
        fold_top1.append(float(top1_stats["top1_accuracy"]))
        fold_macro_f1.append(float(f1_score(y_va, preds, average="macro", zero_division=0)))
        fold_log_loss.append(float(log_loss(y_va, p_clip, labels=[0, 1])))

    mean_top1 = float(np.mean(fold_top1))
    mean_macro_f1 = float(np.mean(fold_macro_f1))
    mean_ll = float(np.mean(fold_log_loss))
    # Composite objective aligning with route-matching accuracy, class balance, and HMM probability calibration
    objective_val = 0.50 * mean_top1 + 0.35 * mean_macro_f1 + 0.15 * (1.0 / (1.0 + mean_ll))

    return {
        "objective_value": float(objective_val),
        "cv_top1_accuracy_mean": mean_top1,
        "cv_top1_accuracy_std": float(np.std(fold_top1, ddof=1)),
        "cv_macro_f1_mean": mean_macro_f1,
        "cv_macro_f1_std": float(np.std(fold_macro_f1, ddof=1)),
        "cv_log_loss_mean": mean_ll,
        "cv_log_loss_std": float(np.std(fold_log_loss, ddof=1)),
        "cv_fit_time_s_mean": float(np.mean(fold_fit_s)),
    }


def _plot_optimization_history(
    trials_df: pd.DataFrame,
    model_name: str,
    output_path: Path,
    is_mock: bool = False,
) -> Path:
    """Plot trial objective values and best-value-so-far progression across Bayesian trials."""
    fig, ax = plt.subplots(figsize=(8.4, 4.6), dpi=160)
    t_nums = trials_df["trial_number"].to_numpy(dtype=int)
    vals = trials_df["objective_value"].to_numpy(dtype=float)
    best_vals = trials_df["best_value_so_far"].to_numpy(dtype=float)

    ax.scatter(t_nums, vals, color="#1f77b4", s=55, alpha=0.85, edgecolors="#111", label="Trial Objective Score")
    ax.plot(t_nums, best_vals, color="#d62728", linewidth=2.2, label="Best Objective So Far")

    best_idx = int(np.argmax(vals))
    ax.scatter(
        [t_nums[best_idx]],
        [vals[best_idx]],
        color="#2ca02c",
        marker="*",
        s=220,
        edgecolors="#111",
        zorder=5,
        label=f"Best Trial (#{t_nums[best_idx]}: {vals[best_idx]:.4f})",
    )

    ax.set_xlabel("Bayesian Optimization Trial Number", fontsize=10, fontweight="bold")
    ax.set_ylabel("5-Fold GroupKFold Objective Value", fontsize=10, fontweight="bold")
    title = f"TPE Bayesian Hyperparameter Optimization History — {model_name}"
    if is_mock:
        title += " [MOCK HARNESS]"
    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", frameon=True, fontsize=9)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_tuned_vs_untuned(
    comparison_df: pd.DataFrame,
    model_name: str,
    output_path: Path,
    is_mock: bool = False,
) -> Path:
    """Plot Untuned vs. Tuned metrics across CV Macro F1, Student-Test Top-1, Student-Test Macro F1, and Route Continuity."""
    metrics = [
        ("cv_macro_f1_mean", "CV Macro F1"),
        ("student_test_top1_accuracy", "Student-Test Top-1"),
        ("student_test_macro_f1", "Student-Test Macro F1"),
        ("route_continuity", "Route Continuity"),
    ]
    untuned_row = comparison_df.loc[comparison_df["variant"] == "Untuned"].iloc[0]
    tuned_row = comparison_df.loc[comparison_df["variant"] == "Tuned (Bayesian TPE)"].iloc[0]

    x = np.arange(len(metrics))
    width = 0.34
    u_vals = np.array([float(untuned_row[col]) for col, _ in metrics])
    t_vals = np.array([float(tuned_row[col]) for col, _ in metrics])

    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=160)
    b1 = ax.bar(x - width / 2, u_vals, width, label="Untuned Baseline", color="#7f7f7f", edgecolor="#222")
    b2 = ax.bar(x + width / 2, t_vals, width, label="Tuned (Bayesian TPE)", color="#2ca02c", edgecolor="#222")

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics], fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Score [0 – 1]", fontsize=10, fontweight="bold")
    title = f"Untuned vs. Bayesian-Tuned Performance — {model_name}"
    if is_mock:
        title += " [MOCK HARNESS]"
    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=12)

    low_y = max(0.0, float(min(u_vals.min(), t_vals.min())) - 0.08)
    ax.set_ylim(low_y, 1.04)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", frameon=True, fontsize=9)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 0.005,
                f"{h:.4f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
            )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def run_bayesian_optimization(
    train_df: pd.DataFrame,
    student_test_df: pd.DataFrame,
    model_key: Optional[str] = None,
    feature_columns: Optional[Iterable[str]] = None,
    n_trials: int = 20,
    n_splits: int = 5,
    seed: int = 42,
    output_dir: Path = OPT_RESULTS_DIR,
    figures_dir: Path = OPT_FIGURES_DIR,
    artifacts_dir: Path = ARTIFACTS_DIR,
    train_source_path: Optional[Path] = None,
    student_test_source_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run TPE Bayesian hyperparameter optimization with GroupKFold CV and export the final tuned model."""
    target_key = model_key or _resolve_default_model_key()
    registry = build_model_registry(seed=seed)
    if target_key not in registry:
        raise KeyError(f"Unknown model_key '{target_key}'. Available: {sorted(registry.keys())}")
    spec = registry[target_key]
    space_spec = get_search_space_spec(target_key)

    is_mock = (
        (bool(train_df["is_mock"].any()) if "is_mock" in train_df.columns else False)
        or (bool(student_test_df["is_mock"].any()) if "is_mock" in student_test_df.columns else False)
    )
    data_source_tag = "MOCK_SYNTHETIC_DO_NOT_REPORT" if is_mock else "REAL_CANDIDATE_FEATURES"

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

    # Assert strict partition isolation between train and student_test
    overlap = set(clean_train["trip_id"].astype(str)) & set(clean_st["trip_id"].astype(str))
    if overlap:
        raise ValueError(f"Leakage error: overlapping trip_ids in train and student_test: {sorted(overlap)}")

    splits = verify_group_kfold_splits(clean_train, n_splits=n_splits)

    # 1. Evaluate Untuned baseline (default parameters) on GroupKFold CV and Student-Test
    default_params = {
        k: v["default"] for k, v in space_spec["parameters"].items()
    }
    untuned_cv = _evaluate_params_group_cv(
        model_key=target_key,
        params=default_params,
        df=clean_train,
        feature_cols=resolved_features,
        splits=splits,
        seed=seed,
    )

    X_train_full = clean_train[resolved_features]
    y_train_full = clean_train[LABEL_COLUMN].to_numpy(dtype=int)
    X_st = clean_st[resolved_features]

    untuned_pipe = build_pipeline_from_params(target_key, default_params, seed=seed)
    t0 = time.perf_counter()
    untuned_pipe.fit(X_train_full, y_train_full)
    untuned_fit_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    untuned_st_probs = _extract_positive_probabilities(untuned_pipe, X_st)
    untuned_infer_s = time.perf_counter() - t1
    untuned_st = compute_all_evaluation_metrics(clean_st, untuned_st_probs)

    # 2. Run Seeded TPE Bayesian Optimization on GroupKFold CV (80 training trips only)
    output_dir.mkdir(parents=True, exist_ok=True)
    sqlite_path = output_dir / "optuna_study.db"

    def _objective(trial: Any) -> float:
        # Trial 0 always evaluates the default untuned configuration as a warm-start baseline
        if int(getattr(trial, "number", -1)) == 0:
            for pname, p_cfg in space_spec["parameters"].items():
                # Register parameters on trial object
                ptype = p_cfg["type"]
                if ptype == "float":
                    trial.suggest_float(pname, float(p_cfg["low"]), float(p_cfg["high"]), log=bool(p_cfg.get("log", False)))
                elif ptype == "int":
                    trial.suggest_int(pname, int(p_cfg["low"]), int(p_cfg["high"]), step=int(p_cfg.get("step", 1)))
                else:
                    trial.suggest_categorical(pname, list(p_cfg["choices"]))
                if hasattr(trial, "params"):
                    trial.params[pname] = default_params[pname]
            cand_params = dict(default_params)
        else:
            cand_params = sample_params_for_model(trial, target_key)

        cv_res = _evaluate_params_group_cv(
            model_key=target_key,
            params=cand_params,
            df=clean_train,
            feature_cols=resolved_features,
            splits=splits,
            seed=seed,
        )
        trial.set_user_attr("cv_top1_accuracy_mean", round(cv_res["cv_top1_accuracy_mean"], 5))
        trial.set_user_attr("cv_macro_f1_mean", round(cv_res["cv_macro_f1_mean"], 5))
        trial.set_user_attr("cv_log_loss_mean", round(cv_res["cv_log_loss_mean"], 5))
        return float(cv_res["objective_value"])

    wall_start = time.perf_counter()
    trials, best_trial, sampler_backend = execute_tpe_study(
        objective_fn=_objective,
        n_trials=n_trials,
        seed=seed,
        sqlite_path=sqlite_path,
    )
    total_wall_time_s = time.perf_counter() - wall_start

    # Ensure trial 0 uses default_params if Optuna overwrote trial.params
    best_params = dict(best_trial.params)
    if best_trial.number == 0:
        best_params = dict(default_params)

    tuned_cv = _evaluate_params_group_cv(
        model_key=target_key,
        params=best_params,
        df=clean_train,
        feature_cols=resolved_features,
        splits=splits,
        seed=seed,
    )

    # 3. Refit the best hyperparameter configuration on all 80 training trips & evaluate on 10 student-test trips
    tuned_pipe = build_pipeline_from_params(target_key, best_params, seed=seed)
    t2 = time.perf_counter()
    tuned_pipe.fit(X_train_full, y_train_full)
    tuned_fit_s = time.perf_counter() - t2

    t3 = time.perf_counter()
    tuned_st_probs = _extract_positive_probabilities(tuned_pipe, X_st)
    tuned_infer_s = time.perf_counter() - t3
    tuned_st = compute_all_evaluation_metrics(clean_st, tuned_st_probs)

    n_st_trips = max(1, tuned_st["n_trips"])
    n_st_obs = max(1, tuned_st["n_observations"])

    comparison_df = pd.DataFrame(
        [
            {
                "variant": "Untuned",
                "model_key": target_key,
                "model_name": spec.display_name,
                "algorithm_family": spec.algorithm_family,
                "cv_macro_f1_mean": round(untuned_cv["cv_macro_f1_mean"], 5),
                "cv_macro_f1_std": round(untuned_cv["cv_macro_f1_std"], 5),
                "cv_top1_accuracy_mean": round(untuned_cv["cv_top1_accuracy_mean"], 5),
                "cv_log_loss_mean": round(untuned_cv["cv_log_loss_mean"], 5),
                "student_test_top1_accuracy": round(untuned_st["top1_accuracy"], 5),
                "student_test_macro_f1": round(untuned_st["macro_f1"], 5),
                "student_test_binary_f1": round(untuned_st["binary_f1"], 5),
                "student_test_precision": round(untuned_st["precision"], 5),
                "student_test_recall": round(untuned_st["recall"], 5),
                "student_test_log_loss": round(untuned_st["log_loss"], 5),
                "route_continuity": round(untuned_st["route_continuity"], 5),
                "fit_time_s": round(untuned_fit_s, 5),
                "infer_time_s": round(untuned_infer_s, 5),
                "infer_ms_per_trip": round((untuned_infer_s * 1000.0) / n_st_trips, 4),
                "infer_us_per_obs": round((untuned_infer_s * 1e6) / n_st_obs, 4),
                "hyperparameters": json.dumps(default_params, sort_keys=True),
                "data_source": data_source_tag,
            },
            {
                "variant": "Tuned (Bayesian TPE)",
                "model_key": target_key,
                "model_name": spec.display_name,
                "algorithm_family": spec.algorithm_family,
                "cv_macro_f1_mean": round(tuned_cv["cv_macro_f1_mean"], 5),
                "cv_macro_f1_std": round(tuned_cv["cv_macro_f1_std"], 5),
                "cv_top1_accuracy_mean": round(tuned_cv["cv_top1_accuracy_mean"], 5),
                "cv_log_loss_mean": round(tuned_cv["cv_log_loss_mean"], 5),
                "student_test_top1_accuracy": round(tuned_st["top1_accuracy"], 5),
                "student_test_macro_f1": round(tuned_st["macro_f1"], 5),
                "student_test_binary_f1": round(tuned_st["binary_f1"], 5),
                "student_test_precision": round(tuned_st["precision"], 5),
                "student_test_recall": round(tuned_st["recall"], 5),
                "student_test_log_loss": round(tuned_st["log_loss"], 5),
                "route_continuity": round(tuned_st["route_continuity"], 5),
                "fit_time_s": round(tuned_fit_s, 5),
                "infer_time_s": round(tuned_infer_s, 5),
                "infer_ms_per_trip": round((tuned_infer_s * 1000.0) / n_st_trips, 4),
                "infer_us_per_obs": round((tuned_infer_s * 1e6) / n_st_obs, 4),
                "hyperparameters": json.dumps(best_params, sort_keys=True),
                "data_source": data_source_tag,
            },
        ]
    )

    trials_df = trials_to_dataframe(trials)
    trials_csv_path = output_dir / "optimization_trials.csv"
    comp_csv_path = output_dir / "tuned_vs_untuned_comparison.csv"
    best_cfg_path = output_dir / "best_hyperparameters.json"
    tuned_scores_path = output_dir / "candidate_scores_student_test_tuned.csv"

    trials_df.to_csv(trials_csv_path, index=False)
    comparison_df.to_csv(comp_csv_path, index=False)

    computational_cost = {
        "n_trials": n_trials,
        "n_splits": n_splits,
        "total_cv_fits": n_trials * n_splits,
        "total_wall_time_s": round(total_wall_time_s, 4),
        "mean_trial_time_s": round(total_wall_time_s / max(1, n_trials), 4),
        "sampler_backend": sampler_backend,
    }

    tuned_vs_untuned_summary = {
        "untuned": comparison_df.iloc[0].to_dict(),
        "tuned": comparison_df.iloc[1].to_dict(),
        "delta": {
            "cv_macro_f1": round(
                float(comparison_df.iloc[1]["cv_macro_f1_mean"] - comparison_df.iloc[0]["cv_macro_f1_mean"]), 5
            ),
            "student_test_top1_accuracy": round(
                float(
                    comparison_df.iloc[1]["student_test_top1_accuracy"]
                    - comparison_df.iloc[0]["student_test_top1_accuracy"]
                ),
                5,
            ),
            "student_test_macro_f1": round(
                float(
                    comparison_df.iloc[1]["student_test_macro_f1"]
                    - comparison_df.iloc[0]["student_test_macro_f1"]
                ),
                5,
            ),
            "student_test_log_loss": round(
                float(
                    comparison_df.iloc[1]["student_test_log_loss"]
                    - comparison_df.iloc[0]["student_test_log_loss"]
                ),
                5,
            ),
        },
    }

    best_cfg_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_key": target_key,
        "model_name": spec.display_name,
        "algorithm_family": spec.algorithm_family,
        "seed": seed,
        "is_mock_data": is_mock,
        "data_source": data_source_tag,
        "best_trial_number": int(best_trial.number),
        "best_objective_value": round(float(best_trial.value), 6),
        "best_hyperparameters": best_params,
        "default_hyperparameters": default_params,
        "computational_cost": computational_cost,
        "tuned_vs_untuned": tuned_vs_untuned_summary,
    }
    best_cfg_path.write_text(json.dumps(best_cfg_payload, indent=2), encoding="utf-8")

    # 4. Export final model and model_card.json to `artifacts/`
    model_path, card_path = export_final_model_artifacts(
        pipeline=tuned_pipe,
        model_key=target_key,
        model_name=spec.display_name,
        algorithm_family=spec.algorithm_family,
        feature_columns=resolved_features,
        best_params=best_params,
        tuned_vs_untuned_summary=tuned_vs_untuned_summary,
        computational_cost=computational_cost,
        seed=seed,
        is_mock=is_mock,
        artifacts_dir=artifacts_dir,
    )

    # Verify load-and-predict smoke test from exported `artifacts/`
    tuned_scores_df = predict_candidate_probs(clean_st, artifacts_dir=artifacts_dir)
    tuned_scores_df.to_csv(tuned_scores_path, index=False)

    # 5. Generate plots
    figures_dir.mkdir(parents=True, exist_ok=True)
    hist_fig = _plot_optimization_history(
        trials_df=trials_df,
        model_name=spec.display_name,
        output_path=figures_dir / "bayesian_optimization_history.png",
        is_mock=is_mock,
    )
    comp_fig = _plot_tuned_vs_untuned(
        comparison_df=comparison_df,
        model_name=spec.display_name,
        output_path=figures_dir / "tuned_vs_untuned_comparison.png",
        is_mock=is_mock,
    )

    return {
        "model_key": target_key,
        "best_params": best_params,
        "trials_df": trials_df,
        "comparison_df": comparison_df,
        "computational_cost": computational_cost,
        "model_path": model_path,
        "model_card_path": card_path,
        "sqlite_path": sqlite_path,
        "figure_paths": {
            "optimization_history": hist_fig,
            "tuned_vs_untuned": comp_fig,
        },
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run TPE Bayesian hyperparameter optimization and export final PathSnap model (Issue #6)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model key to optimize (defaults to winner of Issue #5 or random_forest).",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run optimization on synthetic 80-train + 10-student-test tables.",
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
    parser.add_argument("--n-trials", type=int, default=20, help="Number of Bayesian trials (default: 20).")
    parser.add_argument("--n-splits", type=int, default=5, help="GroupKFold splits (default: 5).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--mock-train-trips", type=int, default=80)
    parser.add_argument("--mock-student-test-trips", type=int, default=10)
    parser.add_argument("--mock-obs-per-trip", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=OPT_RESULTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=OPT_FIGURES_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.mock:
        print(
            "[WARNING] Running Issue #6 Bayesian optimization in --mock mode for pipeline verification. "
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

    res = run_bayesian_optimization(
        train_df=train_df,
        student_test_df=st_df,
        model_key=args.model,
        n_trials=args.n_trials,
        n_splits=args.n_splits,
        seed=args.seed,
        output_dir=args.output_dir,
        figures_dir=args.figures_dir,
        artifacts_dir=args.artifacts_dir,
        train_source_path=train_src,
        student_test_source_path=st_src,
    )

    comp = res["comparison_df"]
    cols = [
        "variant",
        "model_key",
        "cv_macro_f1_mean",
        "student_test_top1_accuracy",
        "student_test_macro_f1",
        "student_test_log_loss",
        "route_continuity",
        "infer_ms_per_trip",
    ]
    print("\n=== Report §4 Untuned vs. Tuned Comparison ===")
    print(comp[cols].to_string(index=False))
    print(f"\nBest Hyperparameters: {json.dumps(res['best_params'])}")
    print(f"Computational Cost: {json.dumps(res['computational_cost'])}")
    print(f"Exported final model artifact to: {res['model_path']}")
    print(f"Exported model card to: {res['model_card_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
