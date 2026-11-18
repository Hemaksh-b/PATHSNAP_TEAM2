# Candidate-Classifier Evaluation, Comparison & HMM Score Export (`model_evaluation/`)

**Owner:** Hemaksh Breja — Member 2 (ML & Tuning)  
**Issue:** `#5 [M2] Evaluate models and select the best candidate classifier`

---

## 1. Overview

This module evaluates all **5 candidate-classification models** from `model_training/` (`logistic_regression`, `knn_distance`, `decision_tree`, `random_forest`, `mlp_neural_net`) across:
1. **80 Training Trips** via 5-fold `GroupKFold` cross-validation (grouped by `trip_id`).
2. **10 Student-Testing Trips** after refitting each model pipeline on the full training partition.

It produces:
- `model_evaluation/results/model_comparison.csv` — Consolidated comparison table matching **Report §3.1** columns (`Model`, `Algorithm Family`, `Key Configuration`, `CV Macro F1-score`, `Top-1 Accuracy`, `Route Continuity`, `Student-Test Macro F1-score`, `Log Loss`, `Inference Time`) plus full machine-readable numeric columns.
- `model_evaluation/results/candidate_scores_train.csv` & `candidate_scores_student_test.csv` — Per-candidate probability tables (`trip_id, obs_idx, arc_id, prob`) for Member 3's HMM/Viterbi decoder (`#7`, `#8`, `#9`).
- `model_evaluation/figures/*.png` — All compulsory comparison plots for **Report §3.3**.
- [SELECTION_RATIONALE.md](file:///Users/hemakshbreja/ML_HACKATHON/model_evaluation/SELECTION_RATIONALE.md) — Evidence-based model selection rationale and answers to **Report §3.2 Compulsory Questions (A, B, C)** for all 5 models.

---

## 2. Reproducible CLI Commands

### A. Harness Verification Mode (`--mock`, before Issue `#3` merges)
```bash
python -m model_evaluation.evaluate --mock --seed 42
```

### B. Real Feature Matrix Mode (once Issue `#3` merges)
```bash
python -m model_evaluation.evaluate \
    --train-features feature_engineering/results/train_features.csv \
    --student-test-features feature_engineering/results/student_test_features.csv \
    --n-splits 5 \
    --seed 42
```

### C. Run Unit & HMM-Integration Tests
```bash
pytest model_evaluation/tests/test_evaluation.py -v
```

---

## 3. Integration with Member 3 (`hmm_viterbi/` and `visualisations/`)

Member 3 can load candidate probabilities and convert them directly into `hmm_viterbi.decode` inputs using `model_evaluation.hmm_bridge`:

```python
import pandas as pd
from model_evaluation.hmm_bridge import scores_df_to_hmm_inputs

scores_df = pd.read_csv("model_evaluation/results/candidate_scores_student_test.csv")
candidates, emission_logp = scores_df_to_hmm_inputs(scores_df, trip_id="00000000")
```
