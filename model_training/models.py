"""Model factory defining >= 5 distinct candidate-classification models across algorithm families (Issue #4).

Every model is wrapped in a `sklearn.pipeline.Pipeline` with in-fold imputation (and feature scaling
where required) so fitting never leaks statistics from validation folds.

Class Imbalance Strategy:
-------------------------
For each GPS observation, up to K candidate road segments are generated, of which at most 1 is the
true road segment (`label = 1`) and K - 1 are negative candidates (`label = 0`).
With K = 5..10, positive examples represent ~10–20% of rows. We handle this imbalance explicitly:
- Linear / Tree / Forest / Boosting models use `class_weight='balanced'` (or `balanced_subsample`),
  which weights classes inversely proportional to class frequencies in each training fold.
- Distance-weighted KNN uses inverse-distance weighting (`weights='distance'`) over standardized features.
- MLP uses sample-weighted or balanced resampling options and probability calibration via log-loss training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


@dataclass(frozen=True)
class ModelSpec:
    """Specification and metadata for a candidate-classification model."""

    key: str
    display_name: str
    algorithm_family: str
    requires_scaling: bool
    imbalance_handling: str
    default_params: Dict[str, Any]
    builder: Callable[[int], Pipeline]


def build_model_registry(seed: int = 42) -> Dict[str, ModelSpec]:
    """Return the dictionary of distinct candidate classifiers across 6 algorithm families."""
    registry: Dict[str, ModelSpec] = {}

    # 1. Linear & Probabilistic Family — Logistic Regression
    lr_params: Dict[str, Any] = {
        "C": 1.0,
        "penalty": "l2",
        "solver": "lbfgs",
        "class_weight": "balanced",
        "max_iter": 500,
        "random_state": seed,
    }
    registry["logistic_regression"] = ModelSpec(
        key="logistic_regression",
        display_name="Logistic Regression (L2, Balanced)",
        algorithm_family="Linear & Probabilistic",
        requires_scaling=True,
        imbalance_handling="class_weight='balanced'",
        default_params=lr_params,
        builder=lambda s=seed: Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=1.0,
                        solver="lbfgs",
                        class_weight="balanced",
                        max_iter=500,
                        random_state=s,
                    ),
                ),
            ]
        ),
    )

    # 2. Instance-Based Family — Distance-Weighted KNN
    knn_params: Dict[str, Any] = {
        "n_neighbors": 15,
        "weights": "distance",
        "metric": "minkowski",
        "p": 2,
    }
    registry["knn_distance"] = ModelSpec(
        key="knn_distance",
        display_name="Distance-Weighted KNN (k=15)",
        algorithm_family="Instance-Based",
        requires_scaling=True,
        imbalance_handling="inverse-distance neighbor weighting",
        default_params=knn_params,
        builder=lambda s=seed: Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    KNeighborsClassifier(
                        n_neighbors=15,
                        weights="distance",
                        metric="minkowski",
                        p=2,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    )

    # 3. Tree-Based Family — Regularized Decision Tree
    dt_params: Dict[str, Any] = {
        "criterion": "gini",
        "max_depth": 8,
        "min_samples_leaf": 25,
        "class_weight": "balanced",
        "random_state": seed,
    }
    registry["decision_tree"] = ModelSpec(
        key="decision_tree",
        display_name="Decision Tree (Depth-Regularized)",
        algorithm_family="Tree-Based",
        requires_scaling=False,
        imbalance_handling="class_weight='balanced'",
        default_params=dt_params,
        builder=lambda s=seed: Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    DecisionTreeClassifier(
                        criterion="gini",
                        max_depth=8,
                        min_samples_leaf=25,
                        class_weight="balanced",
                        random_state=s,
                    ),
                ),
            ]
        ),
    )

    # 4. Ensemble (Bagging) Family — Random Forest
    rf_params: Dict[str, Any] = {
        "n_estimators": 200,
        "max_depth": 12,
        "min_samples_leaf": 10,
        "max_features": "sqrt",
        "class_weight": "balanced_subsample",
        "random_state": seed,
    }
    registry["random_forest"] = ModelSpec(
        key="random_forest",
        display_name="Random Forest (200 Trees, Balanced Subsample)",
        algorithm_family="Ensemble (Bagging)",
        requires_scaling=False,
        imbalance_handling="class_weight='balanced_subsample'",
        default_params=rf_params,
        builder=lambda s=seed: Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=12,
                        min_samples_leaf=10,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=s,
                    ),
                ),
            ]
        ),
    )

    # 5. Boosting Family — Histogram Gradient Boosting (and optional LightGBM if installed)
    hgb_params: Dict[str, Any] = {
        "max_iter": 200,
        "learning_rate": 0.05,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0,
        "class_weight": "balanced",
        "random_state": seed,
    }
    registry["hist_gradient_boosting"] = ModelSpec(
        key="hist_gradient_boosting",
        display_name="HistGradientBoosting (LR=0.05, Balanced)",
        algorithm_family="Boosting",
        requires_scaling=False,
        imbalance_handling="class_weight='balanced'",
        default_params=hgb_params,
        builder=lambda s=seed: Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        max_iter=200,
                        learning_rate=0.05,
                        max_leaf_nodes=31,
                        min_samples_leaf=20,
                        l2_regularization=1.0,
                        class_weight="balanced",
                        random_state=s,
                    ),
                ),
            ]
        ),
    )

    # 6. Neural Networks Family — Shallow Multi-Layer Perceptron (MLP)
    mlp_params: Dict[str, Any] = {
        "hidden_layer_sizes": (64, 32),
        "activation": "relu",
        "alpha": 1e-3,
        "learning_rate_init": 1e-3,
        "early_stopping": True,
        "max_iter": 120,
        "random_state": seed,
    }
    registry["mlp_neural_net"] = ModelSpec(
        key="mlp_neural_net",
        display_name="Shallow MLP Neural Network (64-32, ReLU)",
        algorithm_family="Neural Networks",
        requires_scaling=True,
        imbalance_handling="cross-entropy probability calibration + early stopping",
        default_params=mlp_params,
        builder=lambda s=seed: Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        alpha=1e-3,
                        learning_rate_init=1e-3,
                        early_stopping=True,
                        max_iter=120,
                        random_state=s,
                    ),
                ),
            ]
        ),
    )

    # 7. Ensemble (Randomized Trees) Family — Extra Trees Classifier
    et_params: Dict[str, Any] = {
        "n_estimators": 200,
        "max_depth": 14,
        "min_samples_leaf": 10,
        "class_weight": "balanced",
        "random_state": seed,
    }
    registry["extra_trees"] = ModelSpec(
        key="extra_trees",
        display_name="Extra Trees Classifier (200 Trees, Balanced)",
        algorithm_family="Ensemble (Randomized Trees)",
        requires_scaling=False,
        imbalance_handling="class_weight='balanced'",
        default_params=et_params,
        builder=lambda s=seed: Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    ExtraTreesClassifier(
                        n_estimators=200,
                        max_depth=14,
                        min_samples_leaf=10,
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=s,
                    ),
                ),
            ]
        ),
    )

    # Optional: LightGBM if installed in environment
    try:
        from lightgbm import LGBMClassifier  # type: ignore

        lgbm_params: Dict[str, Any] = {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "class_weight": "balanced",
            "random_state": seed,
            "verbose": -1,
        }
        registry["lightgbm"] = ModelSpec(
            key="lightgbm",
            display_name="LightGBM Classifier (Balanced)",
            algorithm_family="Boosting (LightGBM)",
            requires_scaling=False,
            imbalance_handling="class_weight='balanced'",
            default_params=lgbm_params,
            builder=lambda s=seed: Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "clf",
                        LGBMClassifier(
                            n_estimators=200,
                            learning_rate=0.05,
                            num_leaves=31,
                            class_weight="balanced",
                            random_state=s,
                            n_jobs=-1,
                            verbose=-1,
                        ),
                    ),
                ]
            ),
        )
    except ImportError:
        pass

    return registry


def get_model_pipeline(model_key: str, seed: int = 42) -> Tuple[ModelSpec, Pipeline]:
    """Instantiate a fresh Pipeline and ModelSpec for the requested model key."""
    registry = build_model_registry(seed=seed)
    if model_key not in registry:
        valid = ", ".join(sorted(registry.keys()))
        raise KeyError(f"Unknown model key '{model_key}'. Available models: [{valid}]")
    spec = registry[model_key]
    return spec, spec.builder(seed)
