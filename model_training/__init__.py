"""PathSnap Candidate-Classification Model Training Package (Issue #4)."""

from .mock_data import generate_mock_feature_table
from .models import build_model_registry, get_model_pipeline
from .schema import (
    DEFAULT_FEATURE_COLUMNS,
    ID_COLUMNS,
    LABEL_COLUMN,
    UNCERTAIN_COLUMN,
    validate_and_prepare_features,
)

__all__ = [
    "DEFAULT_FEATURE_COLUMNS",
    "ID_COLUMNS",
    "LABEL_COLUMN",
    "UNCERTAIN_COLUMN",
    "build_model_registry",
    "generate_mock_feature_table",
    "get_model_pipeline",
    "validate_and_prepare_features",
]
