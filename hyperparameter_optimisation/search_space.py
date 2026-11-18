"""Hyperparameter search space definitions and pipeline builder for Bayesian optimization (Issue #6).

Reads the justified search space configuration from `search_space_config.json` and constructs
in-fold `sklearn.pipeline.Pipeline` instances for any trial parameter configuration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

CONFIG_PATH = Path(__file__).resolve().parent / "search_space_config.json"
SEARCH_SPACE_CONFIG: Dict[str, Any] = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def get_search_space_spec(model_key: str) -> Dict[str, Any]:
    """Return the parameter search space dictionary for `model_key`."""
    models_cfg = SEARCH_SPACE_CONFIG["models"]
    if model_key not in models_cfg:
        raise KeyError(
            f"Model '{model_key}' not found in search_space_config.json. "
            f"Valid models: {sorted(models_cfg.keys())}"
        )
    return models_cfg[model_key]


def sample_params_for_model(trial: Any, model_key: str) -> Dict[str, Any]:
    """Suggest hyperparameters from an Optuna (or TPE-compatible) `trial` object."""
    spec = get_search_space_spec(model_key)
    params: Dict[str, Any] = {}

    for param_name, p_cfg in spec["parameters"].items():
        ptype = p_cfg["type"]
        if ptype == "float":
            params[param_name] = float(
                trial.suggest_float(
                    param_name,
                    float(p_cfg["low"]),
                    float(p_cfg["high"]),
                    log=bool(p_cfg.get("log", False)),
                )
            )
        elif ptype == "int":
            params[param_name] = int(
                trial.suggest_int(
                    param_name,
                    int(p_cfg["low"]),
                    int(p_cfg["high"]),
                    step=int(p_cfg.get("step", 1)),
                )
            )
        elif ptype == "categorical":
            params[param_name] = trial.suggest_categorical(param_name, list(p_cfg["choices"]))
        else:
            raise ValueError(f"Unsupported parameter type '{ptype}' for {param_name}")

    return params


def build_pipeline_from_params(
    model_key: str,
    params: Dict[str, Any],
    seed: int = 42,
) -> Pipeline:
    """Instantiate an in-fold `Pipeline` for `model_key` with the given hyperparameters."""
    if model_key == "logistic_regression":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=float(params.get("C", 1.0)),
                        penalty="l2",
                        solver="lbfgs",
                        class_weight=params.get("class_weight", "balanced"),
                        tol=float(params.get("tol", 1e-4)),
                        max_iter=500,
                        random_state=seed,
                    ),
                ),
            ]
        )

    if model_key == "random_forest":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=int(params.get("n_estimators", 200)),
                        max_depth=int(params.get("max_depth", 12)),
                        min_samples_leaf=int(params.get("min_samples_leaf", 10)),
                        max_features=params.get("max_features", "sqrt"),
                        class_weight=params.get("class_weight", "balanced_subsample"),
                        n_jobs=-1,
                        random_state=seed,
                    ),
                ),
            ]
        )

    if model_key == "mlp_neural_net":
        h1 = int(params.get("hidden_layer_1", 64))
        h2 = int(params.get("hidden_layer_2", 32))
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(h1, h2),
                        activation="relu",
                        alpha=float(params.get("alpha", 1e-3)),
                        learning_rate_init=float(params.get("learning_rate_init", 1e-3)),
                        early_stopping=True,
                        max_iter=120,
                        random_state=seed,
                    ),
                ),
            ]
        )

    if model_key == "knn_distance":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    KNeighborsClassifier(
                        n_neighbors=int(params.get("n_neighbors", 15)),
                        weights=str(params.get("weights", "distance")),
                        metric="minkowski",
                        p=int(params.get("p", 2)),
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    if model_key == "decision_tree":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    DecisionTreeClassifier(
                        criterion=str(params.get("criterion", "gini")),
                        max_depth=int(params.get("max_depth", 8)),
                        min_samples_leaf=int(params.get("min_samples_leaf", 25)),
                        class_weight=params.get("class_weight", "balanced"),
                        random_state=seed,
                    ),
                ),
            ]
        )

    raise KeyError(f"Unsupported model_key '{model_key}'.")
