"""Runtime settings for the Laptop-1 API (Issue #10), read from environment variables.

PATHSNAP_TRIPS_DIR  folder whose sub-folders are the selectable trips.
                    Default: the evaluator-testing partition. For development
                    point it at handouts/Student folder/student_test.
PATHSNAP_MOCK       "1" -> return a clearly flagged synthetic response (UI work
                    without data). Off by default; never use it in the demo.
"""
from __future__ import annotations

import os
from pathlib import Path

from data_processing.config import BASE_DIR, EVALUATOR_TEST_DIR, STUDENT_TEST_DIR

LOG_DIR = BASE_DIR / "api" / "logs"


def trips_dir() -> Path:
    env_dir = os.environ.get("PATHSNAP_TRIPS_DIR")
    if env_dir:
        return Path(env_dir)
    if EVALUATOR_TEST_DIR.exists():
        return EVALUATOR_TEST_DIR
    return STUDENT_TEST_DIR


def mock_mode() -> bool:
    return os.environ.get("PATHSNAP_MOCK", "0") == "1"
