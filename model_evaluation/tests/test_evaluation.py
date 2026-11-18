"""Tests verifying Issue #5 acceptance criteria:
1. All 5 models are evaluated on both GroupKFold CV and the 10 student-testing trips.
2. `model_comparison.csv` contains all required Report §3.1 columns and valid metrics.
3. Route Continuity accurately distinguishes connected vs. disconnected Top-1 arc sequences.
4. Exported candidate score files (`trip_id, obs_idx, arc_id, prob`) preserve `obs_idx <-> arc_id`
   mapping 100% identically.
5. Candidate score files load directly into Member 3's `hmm_viterbi` decoder (`decode` + `classifier_emission_logp`).
6. All 6 comparison and diagnostic plots are generated in `figures/`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from model_evaluation.evaluate import (
    evaluate_all_models,
    generate_mock_train_and_student_test,
    main,
)
from model_evaluation.hmm_bridge import (
    SCORE_COLUMNS,
    build_candidate_scores_table,
    scores_df_to_hmm_inputs,
    verify_hmm_decoder_compatibility,
)
from model_evaluation.metrics import (
    compute_all_evaluation_metrics,
    compute_route_continuity,
)


def test_route_continuity_connected_vs_disconnected() -> None:
    # Trip t1: 3 observations, Top-1 arcs are 10 (0->1), 11 (1->2), 12 (2->3) -> 2/2 = 1.0 continuous
    # Trip t2: 3 observations, Top-1 arcs are 20 (0->1), 21 (5->6), 22 (8->9) -> 0/2 = 0.0 continuous
    df = pd.DataFrame(
        {
            "trip_id": ["t1", "t1", "t1", "t1", "t1", "t1", "t2", "t2", "t2"],
            "obs_idx": [0, 0, 1, 1, 2, 2, 0, 1, 2],
            "arc_id": [10, 99, 11, 98, 12, 97, 20, 21, 22],
            "u": [0, 50, 1, 60, 2, 70, 0, 5, 8],
            "v": [1, 51, 2, 61, 3, 71, 1, 6, 9],
            "label": [1, 0, 1, 0, 1, 0, 1, 1, 1],
        }
    )
    probs = np.array([0.9, 0.1, 0.85, 0.15, 0.95, 0.05, 0.9, 0.9, 0.9])

    c_t1 = compute_route_continuity(df.iloc[:6], probs[:6])
    c_t2 = compute_route_continuity(df.iloc[6:], probs[6:])
    c_all = compute_route_continuity(df, probs)

    assert pytest.approx(c_t1) == 1.0
    assert pytest.approx(c_t2) == 0.0
    assert pytest.approx(c_all) == 0.5


def test_candidate_scores_preserve_exact_obs_and_arc_mapping() -> None:
    _, st_df = generate_mock_train_and_student_test(
        n_train_trips=5,
        n_student_test_trips=3,
        obs_per_trip=8,
        k_candidates=5,
        seed=123,
    )
    rng = np.random.default_rng(123)
    probs = rng.uniform(0.01, 0.99, size=len(st_df))
    scores_df = build_candidate_scores_table(st_df, probs)

    assert list(scores_df.columns) == SCORE_COLUMNS
    assert len(scores_df) == len(st_df)
    pd.testing.assert_series_equal(scores_df["trip_id"], st_df["trip_id"], check_names=False)
    pd.testing.assert_series_equal(
        scores_df["obs_idx"], st_df["obs_idx"].astype(int), check_names=False
    )
    pd.testing.assert_series_equal(
        scores_df["arc_id"], st_df["arc_id"].astype(int), check_names=False
    )


def test_hmm_viterbi_decoder_loads_exported_candidate_scores() -> None:
    _, st_df = generate_mock_train_and_student_test(
        n_train_trips=5,
        n_student_test_trips=2,
        obs_per_trip=10,
        k_candidates=4,
        seed=77,
    )
    probs = np.where(st_df["label"].to_numpy() == 1, 0.85, 0.10)
    scores_df = build_candidate_scores_table(st_df, probs)

    trip_id = str(scores_df["trip_id"].iloc[0])
    candidates, emission_logp = scores_df_to_hmm_inputs(scores_df, trip_id=trip_id)
    assert len(candidates) == 10
    assert len(emission_logp) == 10
    for cands, emis in zip(candidates, emission_logp):
        assert len(cands) == 4
        assert emis.shape == (4,)
        assert np.all(np.isfinite(emis))

    decoded = verify_hmm_decoder_compatibility(scores_df, trip_id=trip_id)
    assert len(decoded.arc_ids) == 10
    assert all(a is not None for a in decoded.arc_ids)


def test_full_evaluation_pipeline_and_outputs(tmp_path: Path) -> None:
    train_df, st_df = generate_mock_train_and_student_test(
        n_train_trips=15,
        n_student_test_trips=5,
        obs_per_trip=12,
        k_candidates=5,
        seed=42,
    )
    out_dir = tmp_path / "results"
    fig_dir = tmp_path / "figures"

    res = evaluate_all_models(
        train_df=train_df,
        student_test_df=st_df,
        n_splits=5,
        seed=42,
        output_dir=out_dir,
        figures_dir=fig_dir,
    )

    comp = res["comparison_df"]
    assert len(comp) == 5

    # Verify all Report §3.1 required columns exist
    required_report_cols = {
        "Model",
        "Algorithm Family",
        "Key Configuration",
        "CV Macro F1-score",
        "Top-1 Accuracy",
        "Route Continuity",
        "Student-Test Macro F1-score",
        "Log Loss",
        "Inference Time",
    }
    assert required_report_cols.issubset(set(comp.columns))

    # Exactly 1 model is marked as selected_final_model
    assert int(comp["selected_final_model"].sum()) == 1

    # Verify all CSV/JSON artifacts and 6 PNG figures were written
    assert (out_dir / "model_comparison.csv").exists()
    assert (out_dir / "per_fold_cv_metrics.csv").exists()
    assert (out_dir / "student_test_per_trip_metrics.csv").exists()
    assert (out_dir / "candidate_scores_train.csv").exists()
    assert (out_dir / "candidate_scores_student_test.csv").exists()
    assert (out_dir / "confusion_matrices.json").exists()
    assert (out_dir / "evaluation_manifest.json").exists()

    expected_figs = {
        "cv_macro_f1_comparison.png",
        "student_test_top1_and_macro_f1.png",
        "log_loss_comparison.png",
        "inference_time_vs_accuracy.png",
        "confusion_matrices.png",
        "mlp_training_loss_curve.png",
    }
    for fname in expected_figs:
        assert (fig_dir / fname).exists(), f"Missing expected figure: {fname}"


def test_cli_mock_execution(tmp_path: Path) -> None:
    out_dir = tmp_path / "res"
    fig_dir = tmp_path / "figs"
    rc = main(
        [
            "--mock",
            "--mock-train-trips",
            "10",
            "--mock-student-test-trips",
            "4",
            "--mock-obs-per-trip",
            "10",
            "--n-splits",
            "5",
            "--seed",
            "42",
            "--output-dir",
            str(out_dir),
            "--figures-dir",
            str(fig_dir),
        ]
    )
    assert rc == 0
    assert (out_dir / "model_comparison.csv").exists()
