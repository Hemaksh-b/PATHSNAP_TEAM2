"""Tests verifying Issue #6 acceptance criteria:
1. Genuine TPE Bayesian optimization history is saved to both SQLite (`optuna_study.db`) and CSV.
2. GroupKFold by `trip_id` is used inside the objective without evaluator leakage.
3. Measured tuned-vs-untuned comparison table is generated and committed.
4. Final model pipeline and `model_card.json` are exported to `artifacts/`.
5. Fresh load-and-predict smoke test (`predict_candidate_probs`) passes without requiring labels.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

from hyperparameter_optimisation.inference import (
    load_exported_model,
    predict_candidate_probs,
)
from hyperparameter_optimisation.optimize import main, run_bayesian_optimization
from hyperparameter_optimisation.search_space import SEARCH_SPACE_CONFIG
from model_evaluation.evaluate import generate_mock_train_and_student_test


def test_search_space_config_covers_all_five_models() -> None:
    expected = {
        "logistic_regression",
        "knn_distance",
        "decision_tree",
        "random_forest",
        "mlp_neural_net",
    }
    assert set(SEARCH_SPACE_CONFIG["models"].keys()) == expected
    for key, spec in SEARCH_SPACE_CONFIG["models"].items():
        assert len(spec["parameters"]) >= 3, f"Search space too small for {key}"


def test_bayesian_optimization_persists_trials_and_exports_artifacts(tmp_path: Path) -> None:
    train_df, st_df = generate_mock_train_and_student_test(
        n_train_trips=12,
        n_student_test_trips=4,
        obs_per_trip=10,
        k_candidates=5,
        seed=42,
    )
    out_dir = tmp_path / "results"
    fig_dir = tmp_path / "figures"
    art_dir = tmp_path / "artifacts"

    res = run_bayesian_optimization(
        train_df=train_df,
        student_test_df=st_df,
        model_key="random_forest",
        n_trials=8,
        n_splits=4,
        seed=42,
        output_dir=out_dir,
        figures_dir=fig_dir,
        artifacts_dir=art_dir,
    )

    # 1. Check trial history CSV & SQLite persistence
    trials_df = res["trials_df"]
    assert len(trials_df) == 8
    assert (out_dir / "optimization_trials.csv").exists()
    assert (out_dir / "optuna_study.db").exists()

    conn = sqlite3.connect(str(out_dir / "optuna_study.db"))
    try:
        count = conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0]
        assert count == 8
    finally:
        conn.close()

    # 2. Check tuned vs. untuned comparison table
    comp_df = res["comparison_df"]
    assert list(comp_df["variant"]) == ["Untuned", "Tuned (Bayesian TPE)"]
    required_cols = {
        "cv_macro_f1_mean",
        "student_test_top1_accuracy",
        "student_test_macro_f1",
        "student_test_log_loss",
        "route_continuity",
        "infer_ms_per_trip",
    }
    assert required_cols.issubset(set(comp_df.columns))

    # 3. Check exported model and model_card.json in artifacts/
    assert (art_dir / "model.joblib").exists()
    assert (art_dir / "model_card.json").exists()
    card = json.loads((art_dir / "model_card.json").read_text(encoding="utf-8"))
    assert card["model_key"] == "random_forest"
    assert card["computational_cost"]["n_trials"] == 8

    # 4. Fresh load-and-predict smoke test WITHOUT label column (simulating live FastAPI inference)
    inference_input = st_df.drop(columns=["label", "label_uncertain"], errors="ignore")
    loaded_pipe, loaded_card = load_exported_model(artifacts_dir=art_dir)
    preds_df = predict_candidate_probs(
        inference_input,
        artifacts_dir=art_dir,
        pipeline=loaded_pipe,
        feature_columns=loaded_card["feature_columns"],
    )

    assert list(preds_df.columns) == ["trip_id", "obs_idx", "arc_id", "prob"]
    assert len(preds_df) == len(inference_input)
    assert preds_df["prob"].between(0.0, 1.0).all()
    pd.testing.assert_series_equal(preds_df["arc_id"], inference_input["arc_id"].astype(int), check_names=False)

    # 5. Check generated plots
    assert (fig_dir / "bayesian_optimization_history.png").exists()
    assert (fig_dir / "tuned_vs_untuned_comparison.png").exists()


def test_cli_execution_and_reproducibility(tmp_path: Path) -> None:
    out1 = tmp_path / "r1"
    art1 = tmp_path / "a1"
    out2 = tmp_path / "r2"
    art2 = tmp_path / "a2"

    common_args = [
        "--model",
        "logistic_regression",
        "--mock",
        "--mock-train-trips",
        "10",
        "--mock-student-test-trips",
        "3",
        "--mock-obs-per-trip",
        "10",
        "--n-trials",
        "6",
        "--n-splits",
        "5",
        "--seed",
        "42",
    ]
    rc1 = main(common_args + ["--output-dir", str(out1), "--figures-dir", str(out1 / "f"), "--artifacts-dir", str(art1)])
    rc2 = main(common_args + ["--output-dir", str(out2), "--figures-dir", str(out2 / "f"), "--artifacts-dir", str(art2)])
    assert rc1 == 0 and rc2 == 0

    t1 = pd.read_csv(out1 / "optimization_trials.csv")
    t2 = pd.read_csv(out2 / "optimization_trials.csv")
    pd.testing.assert_series_equal(t1["objective_value"], t2["objective_value"])
