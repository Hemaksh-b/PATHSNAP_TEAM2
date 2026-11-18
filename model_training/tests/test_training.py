"""Unit tests verifying Issue #4 acceptance criteria:
1. At least 5 distinct models across algorithm families are registered and trainable.
2. GroupKFold splits by `trip_id` have zero trip overlap between train and validation folds.
3. Preprocessing (imputer/scaler) is fitted inside each fold via Pipeline without leakage.
4. Evaluator-testing data is strictly rejected by leakage guards.
5. Observation-level Top-1 accuracy and Candidate Miss Rate are computed accurately.
6. CLI and experiment logging are reproducible with fixed seeds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from model_training.mock_data import generate_mock_feature_table
from model_training.models import build_model_registry, get_model_pipeline
from model_training.schema import (
    DEFAULT_FEATURE_COLUMNS,
    assert_no_evaluator_leakage,
    validate_and_prepare_features,
)
from model_training.train import (
    evaluate_top1_observation_accuracy,
    main,
    run_group_cv_experiment,
    verify_group_kfold_splits,
)


def test_registry_has_at_least_five_distinct_algorithm_families() -> None:
    registry = build_model_registry(seed=42)
    assert len(registry) >= 5
    required_keys = {
        "logistic_regression",
        "knn_distance",
        "decision_tree",
        "random_forest",
        "hist_gradient_boosting",
        "mlp_neural_net",
    }
    assert required_keys.issubset(set(registry.keys()))
    families = {spec.algorithm_family for spec in registry.values()}
    assert len(families) >= 5


def test_group_kfold_strict_trip_disjointness() -> None:
    df = generate_mock_feature_table(n_trips=20, obs_per_trip=15, k_candidates=5, seed=101)
    clean_df, _ = validate_and_prepare_features(df)
    splits = verify_group_kfold_splits(clean_df, n_splits=5)

    assert len(splits) == 5
    seen_val_trips: set[str] = set()

    for train_idx, val_idx in splits:
        train_trips = set(clean_df.iloc[train_idx]["trip_id"].unique())
        val_trips = set(clean_df.iloc[val_idx]["trip_id"].unique())
        # No trip may appear in both train and validation folds
        assert train_trips.isdisjoint(val_trips)
        # Each trip appears in exactly one validation fold across K splits
        assert seen_val_trips.isdisjoint(val_trips)
        seen_val_trips.update(val_trips)

    assert seen_val_trips == set(clean_df["trip_id"].unique())


def test_in_fold_scaler_does_not_leak_validation_distribution() -> None:
    df = generate_mock_feature_table(n_trips=10, obs_per_trip=10, k_candidates=4, seed=7)
    clean_df, feature_cols = validate_and_prepare_features(df)
    splits = verify_group_kfold_splits(clean_df, n_splits=5)
    train_idx, val_idx = splits[0]

    # Inject extreme values into the validation fold only
    corrupted = clean_df.copy()
    corrupted.loc[corrupted.index[val_idx], "dist_m"] = 999999.0

    _, pipe_clean = get_model_pipeline("logistic_regression", seed=42)
    _, pipe_corrupted = get_model_pipeline("logistic_regression", seed=42)

    pipe_clean.fit(clean_df.iloc[train_idx][feature_cols], clean_df.iloc[train_idx]["label"])
    pipe_corrupted.fit(
        corrupted.iloc[train_idx][feature_cols], corrupted.iloc[train_idx]["label"]
    )

    # Fitted scaler means and scales must be identical because validation rows were never seen
    np.testing.assert_allclose(
        pipe_clean.named_steps["scaler"].mean_,
        pipe_corrupted.named_steps["scaler"].mean_,
    )
    np.testing.assert_allclose(
        pipe_clean.named_steps["scaler"].scale_,
        pipe_corrupted.named_steps["scaler"].scale_,
    )


def test_evaluator_leakage_guard_rejects_evaluator_partition_and_paths() -> None:
    df = generate_mock_feature_table(n_trips=6, obs_per_trip=5, k_candidates=3, seed=1)

    # 1. Rejects path containing evaluator_test
    with pytest.raises(ValueError, match="Data leakage violation"):
        assert_no_evaluator_leakage(df, source_path=Path("handouts/Student folder/evaluator_test/trip_01.csv"))

    # 2. Rejects dataframe with partition == 'evaluator_test'
    bad_df = df.copy()
    bad_df.loc[0, "partition"] = "evaluator_test"
    with pytest.raises(ValueError, match="Data leakage violation"):
        validate_and_prepare_features(bad_df)


def test_top1_observation_accuracy_and_miss_rate_calculation() -> None:
    # 3 observations:
    # obs 0: true arc is cand 1, model gives highest prob to cand 1 -> Top-1 HIT
    # obs 1: true arc is cand 0, model gives highest prob to cand 1 -> Top-1 MISS (but candidate present)
    # obs 2: no candidate has label=1 (candidate miss) -> Top-1 MISS & Candidate Miss
    toy_df = pd.DataFrame(
        {
            "trip_id": ["t1", "t1", "t1", "t1", "t1", "t1"],
            "obs_idx": [0, 0, 1, 1, 2, 2],
            "arc_id": [10, 11, 20, 21, 30, 31],
            "label": [0, 1, 1, 0, 0, 0],
        }
    )
    probs = np.array([0.2, 0.9, 0.3, 0.8, 0.6, 0.4])
    stats = evaluate_top1_observation_accuracy(toy_df, probs)

    assert stats["n_observations"] == 3
    assert pytest.approx(stats["candidate_miss_rate"]) == 1.0 / 3.0
    assert pytest.approx(stats["top1_accuracy"]) == 1.0 / 3.0
    assert pytest.approx(stats["top1_accuracy_given_hit"]) == 1.0 / 2.0


def test_all_models_train_and_produce_reproducible_logs(tmp_path: Path) -> None:
    df = generate_mock_feature_table(n_trips=15, obs_per_trip=12, k_candidates=5, seed=42)
    assert set(DEFAULT_FEATURE_COLUMNS).issubset(set(df.columns))

    summary_1, folds_1 = run_group_cv_experiment(
        df=df,
        n_splits=5,
        seed=42,
        output_dir=tmp_path / "run1",
    )
    summary_2, _ = run_group_cv_experiment(
        df=df,
        n_splits=5,
        seed=42,
        output_dir=tmp_path / "run2",
    )

    assert len(summary_1) >= 6
    assert len(folds_1) == len(summary_1) * 5
    assert (summary_1["data_source"] == "MOCK_SYNTHETIC_DO_NOT_REPORT").all()

    # Fixed seed must yield identical CV Macro F1 and Top-1 Accuracy across runs
    pd.testing.assert_series_equal(summary_1["cv_macro_f1_mean"], summary_2["cv_macro_f1_mean"])
    pd.testing.assert_series_equal(summary_1["cv_top1_accuracy_mean"], summary_2["cv_top1_accuracy_mean"])

    # Verify output files were created
    assert (tmp_path / "run1" / "mock_harness_summary_metrics.csv").exists()
    assert (tmp_path / "run1" / "mock_harness_per_fold_metrics.csv").exists()
    assert (tmp_path / "run1" / "mock_harness_experiment_manifest.json").exists()


def test_cli_single_model_execution(tmp_path: Path) -> None:
    rc = main(
        [
            "--model",
            "logistic_regression",
            "random_forest",
            "--mock",
            "--mock-trips",
            "10",
            "--mock-obs-per-trip",
            "10",
            "--n-splits",
            "5",
            "--seed",
            "42",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    summary = pd.read_csv(tmp_path / "mock_harness_summary_metrics.csv")
    assert set(summary["model_key"]) == {"logistic_regression", "random_forest"}
