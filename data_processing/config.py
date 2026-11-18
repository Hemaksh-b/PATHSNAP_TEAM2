from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
HANDOUTS_DIR = BASE_DIR / "handouts" / "Student folder"
TRAIN_DIR = HANDOUTS_DIR / "train"
STUDENT_TEST_DIR = HANDOUTS_DIR / "student_test"
EVALUATOR_TEST_DIR = HANDOUTS_DIR / "evaluator_test"  # Reserved for evaluation

ARTIFACTS_DIR = BASE_DIR / "artifacts"
RESULTS_DIR = BASE_DIR / "supporting_results"
VIS_DIR = BASE_DIR / "visualisations"
