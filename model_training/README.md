# Candidate-Classification Model Training (`model_training/`)

**Owner:** Hemaksh Breja — Member 2 (ML & Tuning)  
**Issue:** `#4 [M2] Train at least five ML candidate-classification models`

---

## 1. Overview

For each GPS observation $o_t$ in a trip, the candidate generator (`#2`) yields up to $K$ nearby road segments $\{c_{t,1}, \dots, c_{t,K}\}$, and the feature engineering module (`#3`) extracts spatial, geometric, and temporal features per `(trip_id, obs_idx, arc_id)` pair.

This module trains and cross-validates binary candidate classifiers that estimate the posterior probability:

$$P(\text{label} = 1 \mid \mathbf{x}_{t,k})$$

which serves both as the **classifier-only baseline** (`argmax` candidate per observation) and as the **emission probability model** for the HMM/Viterbi sequential route decoder (`#7`, `#8`).

---

## 2. Algorithm Families Implemented

We train **7 distinct models across 6 algorithm families** (exceeding the minimum 5 required by *Models and Training Guidelines.pdf* §2–3):

| Model Key | Display Name | Algorithm Family | In-Fold Scaling | Imbalance Handling |
| :--- | :--- | :--- | :--- | :--- |
| `logistic_regression` | Logistic Regression (L2, Balanced) | Linear & Probabilistic | `StandardScaler` | `class_weight='balanced'` |
| `knn_distance` | Distance-Weighted KNN ($k=15$) | Instance-Based | `StandardScaler` | Inverse-distance weighting |
| `decision_tree` | Decision Tree (Depth-Regularized) | Tree-Based | None (median imputer) | `class_weight='balanced'` |
| `random_forest` | Random Forest (200 Trees) | Ensemble (Bagging) | None (median imputer) | `class_weight='balanced_subsample'` |
| `extra_trees` | Extra Trees (200 Trees) | Ensemble (Randomized Trees) | None (median imputer) | `class_weight='balanced'` |
| `hist_gradient_boosting` | HistGradientBoosting (`lr=0.05`) | Boosting | None (median imputer) | `class_weight='balanced'` |
| `mlp_neural_net` | Shallow MLP (`64-32`, ReLU) | Neural Networks | `StandardScaler` | Cross-entropy log-loss + early stopping |

*(If `lightgbm` is installed in the active environment, `lightgbm` is also registered automatically under the Boosting family.)*

---

## 3. Experimental Design & Leakage Prevention

1. **Trip-Grouped Cross-Validation (`GroupKFold`):**
   - Observations within the same trip share road topology, vehicle speed profile, and GPS noise characteristics.
   - All cross-validation splits use `GroupKFold(n_splits=5)` grouped by `trip_id` across the 80 training trips.
   - Every fold explicitly asserts `set(train_trips).isdisjoint(set(val_trips))`.
2. **In-Fold Preprocessing (`sklearn.pipeline.Pipeline`):**
   - `SimpleImputer(strategy="median")` and `StandardScaler()` are encapsulated inside each model's `Pipeline` so statistics are fitted strictly on each fold's training trips and never leak from validation trips.
3. **Class Imbalance ($1 : K$):**
   - Because each GPS observation has at most 1 positive road arc among $K$ candidates (~15–20% positives), models weight classes inversely proportional to class frequencies (`class_weight='balanced'`) and are evaluated on **Observation-level Top-1 Accuracy** and **Macro F1** rather than raw binary accuracy.
4. **Evaluator Partition Guard:**
   - `schema.assert_no_evaluator_leakage()` blocks any path or dataframe referencing `evaluator_test`.

---

## 4. Reproducible CLI Commands

### A. Harness Verification Mode (`--mock`, before Issue `#3` merges)
> **Note:** Outputs generated with `--mock` are prefixed with `mock_harness_` and stamped with `MOCK_SYNTHETIC_DO_NOT_REPORT`. Never report mock numbers in the final report.

```bash
# Train and cross-validate all models on the synthetic 80-trip harness
python -m model_training.train --all --mock --seed 42

# Or train an individual model by key
python -m model_training.train --model logistic_regression --mock --seed 42
python -m model_training.train --model knn_distance --mock --seed 42
python -m model_training.train --model decision_tree --mock --seed 42
python -m model_training.train --model random_forest --mock --seed 42
python -m model_training.train --model hist_gradient_boosting --mock --seed 42
python -m model_training.train --model mlp_neural_net --mock --seed 42
```

### B. Real Feature Matrix Mode (once Issue `#3` merges)
```bash
# Train all models on Member 1's candidate feature table (#3)
python -m model_training.train \
    --all \
    --features-path feature_engineering/results/train_features.csv \
    --n-splits 5 \
    --seed 42
```

### C. Run Unit Tests
```bash
pytest model_training/tests/test_training.py -v
```
