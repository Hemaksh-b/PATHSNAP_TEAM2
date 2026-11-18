"""Model artifact export and production inference interface for Members 3 & 4 (Issue #6).

Exports to `artifacts/`:
- `artifacts/model.joblib`: Fitted scikit-learn `Pipeline` (preprocessing + tuned classifier).
- `artifacts/model_card.json`: Feature schema, hyperparameters, package versions, random seed,
  and tuned vs. untuned performance summary.

Provides:
- `load_exported_model(artifacts_dir) -> Tuple[Pipeline, Dict[str, Any]]`
- `predict_candidate_probs(candidates_df, artifacts_dir=...) -> pd.DataFrame[trip_id, obs_idx, arc_id, prob]`
"""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn

from data_processing.config import ARTIFACTS_DIR
from model_evaluation.hmm_bridge import build_candidate_scores_table
from model_training.train import _extract_positive_probabilities

MODEL_FILENAME = "model.joblib"
MODEL_CARD_FILENAME = "model_card.json"


def export_final_model_artifacts(
    pipeline: Any,
    model_key: str,
    model_name: str,
    algorithm_family: str,
    feature_columns: List[str],
    best_params: Dict[str, Any],
    tuned_vs_untuned_summary: Dict[str, Any],
    computational_cost: Dict[str, Any],
    seed: int = 42,
    is_mock: bool = False,
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> Tuple[Path, Path]:
    """Save the refitted tuned `Pipeline` and `model_card.json` into `artifacts/`."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_dir / MODEL_FILENAME
    card_path = artifacts_dir / MODEL_CARD_FILENAME

    bundle = {
        "pipeline": pipeline,
        "model_key": model_key,
        "model_name": model_name,
        "feature_columns": list(feature_columns),
        "seed": seed,
        "is_mock": is_mock,
    }
    joblib.dump(bundle, model_path)

    model_card: Dict[str, Any] = {
        "model_key": model_key,
        "model_name": model_name,
        "algorithm_family": algorithm_family,
        "version": "1.0.0",
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "is_mock_data": is_mock,
        "data_source": "MOCK_SYNTHETIC_DO_NOT_REPORT" if is_mock else "REAL_CANDIDATE_FEATURES",
        "seed": seed,
        "feature_columns": list(feature_columns),
        "n_features": len(feature_columns),
        "best_hyperparameters": best_params,
        "tuned_vs_untuned": tuned_vs_untuned_summary,
        "computational_cost": computational_cost,
        "environment_versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
        "inference_contract": {
            "function": "hyperparameter_optimisation.inference.predict_candidate_probs",
            "required_id_columns": ["trip_id", "obs_idx", "arc_id"],
            "required_feature_columns": list(feature_columns),
            "output_columns": ["trip_id", "obs_idx", "arc_id", "prob"],
        },
    }
    card_path.write_text(json.dumps(model_card, indent=2), encoding="utf-8")
    return model_path, card_path


def load_exported_model(
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> Tuple[Any, Dict[str, Any]]:
    """Load the saved model pipeline and model card from `artifacts/`."""
    model_path = artifacts_dir / MODEL_FILENAME
    card_path = artifacts_dir / MODEL_CARD_FILENAME

    if not model_path.exists():
        raise FileNotFoundError(f"Exported model artifact not found at: {model_path}")
    if not card_path.exists():
        raise FileNotFoundError(f"Model card not found at: {card_path}")

    loaded = joblib.load(model_path)
    model_card = json.loads(card_path.read_text(encoding="utf-8"))

    if isinstance(loaded, dict) and "pipeline" in loaded:
        pipeline = loaded["pipeline"]
    else:
        pipeline = loaded

    return pipeline, model_card


def predict_candidate_probs(
    candidates_df: pd.DataFrame,
    artifacts_dir: Path = ARTIFACTS_DIR,
    pipeline: Optional[Any] = None,
    feature_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Predict per-candidate probabilities `DataFrame[trip_id, obs_idx, arc_id, prob]` for Members 3 & 4.

    Guarantees:
    - Never reads `.route` or ground-truth `label` columns (even if absent at inference time).
    - Preserves `(trip_id, obs_idx, arc_id)` order and values 100% identically.
    - Returns empty 4-column DataFrame if `candidates_df` is empty.
    """
    required_ids = ["trip_id", "obs_idx", "arc_id"]
    missing_ids = [c for c in required_ids if c not in candidates_df.columns]
    if missing_ids:
        raise KeyError(f"candidates_df missing required identifier columns: {missing_ids}")

    if candidates_df.empty:
        return pd.DataFrame(columns=["trip_id", "obs_idx", "arc_id", "prob"])

    if pipeline is None or feature_columns is None:
        loaded_pipe, card = load_exported_model(artifacts_dir=artifacts_dir)
        pipeline = pipeline or loaded_pipe
        feature_columns = feature_columns or list(card["feature_columns"])

    missing_feats = [c for c in feature_columns if c not in candidates_df.columns]
    if missing_feats:
        raise KeyError(
            f"candidates_df is missing required model feature columns: {missing_feats}"
        )

    X = candidates_df[feature_columns]
    probs = _extract_positive_probabilities(pipeline, X)
    return build_candidate_scores_table(candidates_df, probs)
