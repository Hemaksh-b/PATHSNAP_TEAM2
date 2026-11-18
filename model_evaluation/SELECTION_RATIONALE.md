# Model Comparison, Selection Rationale & Compulsory Per-Model Analysis (Report §3.1, §3.2 & §4)

**Owner:** Hemaksh Breja — Member 2 (ML & Tuning)  
**Issues:** `#4`, `#5`, `#6`, `#28` (`model_training/`, `model_evaluation/`, `hyperparameter_optimisation/`, `artifacts/`)

---

## 1. Real-Dataset Overall Model Comparison Table (Report §3.1)

Evaluated on the official handout split:
- **Training & Cross-Validation:** **80 training trips** (`31,903` GPS observations, `142,029` candidate road-segment rows at $K=5, R=50\text{ m}$), evaluated with **5-fold `GroupKFold`** grouped by `trip_id`.
- **Development-Stage Evaluation:** **10 student-testing trips** (`3,862` GPS observations, `16,660` candidate road-segment rows).

| Model | Algorithm Family | Key Configuration | CV Macro F1-score | Top-1 Accuracy | Route continuity | Student-Test Macro F1-score | Log Loss | Inference Time |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model 1: Random Forest (200 Trees, Balanced Subsample)** | Ensemble (Bagging) | `n_estimators=200`, `max_depth=12`, `min_samples_leaf=10`, `balanced_subsample` | **0.8771 ± 0.0227** | **0.8576 (85.76%)** | **0.9354 (93.54%)** | **0.8979** | 0.1934 | 3.99 ms/trip (10.3 µs/obs) |
| **Model 2: Shallow MLP Neural Network (64-32, ReLU)** | Neural Networks | `hidden_layers=(64, 32)`, `activation=relu`, `alpha=0.001`, `early_stopping=True`, `StandardScaler` | 0.8696 ± 0.0223 | 0.8467 (84.67%) | 0.9333 (93.33%) | 0.8961 | **0.1835** | 0.42 ms/trip (1.1 µs/obs) |
| **Model 3: Decision Tree (Depth-Regularized)** | Tree-Based | `criterion=gini`, `max_depth=8`, `min_samples_leaf=25`, `balanced` | 0.8645 ± 0.0229 | 0.8449 (84.49%) | 0.9307 (93.07%) | 0.8907 | 0.2302 | 0.27 ms/trip (0.7 µs/obs) |
| **Model 4: Distance-Weighted KNN (k=15)** | Instance-Based | `k=15`, `weights=distance`, `Euclidean (p=2)`, `StandardScaler` | 0.8657 ± 0.0191 | 0.8402 (84.02%) | 0.9291 (92.91%) | 0.8949 | 0.3065 | 78.97 ms/trip (204.5 µs/obs) |
| **Model 5: Logistic Regression (L2, Balanced)** | Linear & Probabilistic | `L2 penalty`, `C=1.0`, `solver=lbfgs`, `class_weight=balanced`, `StandardScaler` | 0.8250 ± 0.0131 | 0.8278 (82.78%) | 0.9239 (92.39%) | 0.8417 | 0.2969 | **0.22 ms/trip (0.6 µs/obs)** |

---

## 2. Primary Metric & Final Model Selection Justification

### Selected Final Classifier: `Random Forest` (`random_forest` — Ensemble / Bagging)

1. **Why Binary Accuracy Alone Is Inappropriate:**
   - Each GPS observation has up to $K=5$ candidate road segments, of which at most 1 is the true road arc (`label = 1`, ~21% positive rate) and $K-1$ are negatives (`label = 0`, ~79% negative rate). Predicting `0` everywhere yields ~79% binary accuracy while achieving **0% route-matching accuracy**.
   - Therefore, our primary selection criteria combine **Student-Test Observation-level Top-1 Accuracy** (40% weight), **5-Fold GroupKFold Macro F1** (25% weight), **Student-Test Macro F1** (20% weight), and **Probability Calibration Log Loss** (15% weight).

2. **Why `Random Forest` Is Selected:**
   - **Highest Pointwise Route-Matching Accuracy (`Top-1 = 85.76%`):** Ranks the true road segment #1 on the 10 unseen student-testing trips more accurately than any other model (`85.76%` vs. `84.67%` for MLP, `84.49%` for Decision Tree, `84.02%` for KNN, and `82.78%` for Logistic Regression).
   - **Highest Class-Balanced Generalisation (`CV Macro F1 = 0.8771 ± 0.0227`, `Student-Test Macro F1 = 0.8979` → `0.9042` tuned):** Captures non-linear interactions between perpendicular distance (`dist_m`), distance gap to nearest (`dist_gap_to_nearest_m`), fractional position (`frac_along`), and heading alignment (`heading_diff_deg`) at intersections.
   - **Best Calibrated Posterior Probabilities After Bayesian Tuning (`Log Loss = 0.1633`):** After TPE Bayesian optimization (`n_estimators=160, max_depth=13, min_samples_leaf=17, max_features='log2', class_weight=None`), Random Forest's Student-Test Log Loss drops from `0.1934` to **`0.1633`** (the lowest of all models) while Student-Test Macro F1 rises to **`0.9042`**, providing well-calibrated log-probabilities $\log P(o_t \mid c_{t,k})$ for Member 3's HMM Viterbi decoder.
   - **Highest Route Continuity (`93.54%`) & Low Inference Latency (`2.98 ms/trip` tuned):** Runs 26× faster than KNN (`78.97 ms/trip`), supporting real-time FastAPI inference on Laptop 1.

---

## 3. Compulsory Questions for Every Model (Report §3.2)

### Model 1: Random Forest (`random_forest` — Ensemble / Bagging) [SELECTED]
- **A1. Suitability for PathSnap:** Aggregates 200 bootstrap decision trees with random feature subsets (`max_features='sqrt'`), learning non-linear spatial-kinematic rules while averaging across trees to produce smooth probability estimates for HMM emissions.
- **A2. Strengths & Limitations:**
  - *Strengths:* Highest Top-1 accuracy (`85.76%`), highest Macro F1 (`0.8979` untuned / `0.9042` tuned), highest Route Continuity (`93.54%`), invariant to monotonic feature scaling, robust to GPS speed outliers.
  - *Limitations:* Larger serialized artifact (`~21 MB`) than linear or shallow neural models.
- **B5. Important Hyperparameters & Initial Values:** `n_estimators=200`, `max_depth=12`, `min_samples_leaf=10`, `max_features='sqrt'`, `class_weight='balanced_subsample'`.
- **B6. Loss Function / Objective:** Bootstrap-subsampled class-weighted Gini impurity reduction.
- **C7–C9. Experimental Results & Overfitting Analysis:** Achieves **CV Macro F1 = `0.8771 ± 0.0227`** and **Student-Test Macro F1 = `0.8979`** (`Precision = 0.7803`, `Recall = 0.9254`, `Top-1 = 0.8576`, `Log Loss = 0.1934`). Because Student-Test Macro F1 (`0.8979`) slightly exceeds CV Macro F1 (`0.8771`), there is zero overfitting across trips.

---

### Model 2: Shallow MLP Neural Network (`mlp_neural_net` — Neural Networks)
- **A1. Suitability for PathSnap:** Two-hidden-layer feed-forward network (`64 -> 32` with ReLU activations) over standardized features that learns smooth non-linear combinations of spatial proximity and angular alignment.
- **A2. Strengths & Limitations:**
  - *Strengths:* Directly minimizes cross-entropy loss, achieving the lowest untuned Log Loss (`0.1835`) and sub-millisecond inference (`0.42 ms/trip`).
  - *Limitations:* Requires strict in-fold `StandardScaler` and validation-based early stopping to avoid overfitting.
- **B5. Important Hyperparameters & Initial Values:** `hidden_layer_sizes=(64, 32)`, `activation='relu'`, `alpha=1e-3`, `learning_rate_init=1e-3`, `early_stopping=True`, `max_iter=120`.
- **B6. Loss Function / Objective:** L2-regularized binary cross-entropy loss optimized via Adam.
- **C7–C9. Experimental Results & Overfitting Analysis:** Achieves **CV Macro F1 = `0.8696 ± 0.0223`**, **Student-Test Top-1 = `0.8467 (84.67%)`**, **Student-Test Macro F1 = `0.8961`**, **Log Loss = `0.1835`**, and **Route Continuity = `0.9333`**. The epoch-wise training loss and validation accuracy curves (`figures/mlp_training_loss_curve.png`) converge smoothly without divergence.

---

### Model 3: Regularized Decision Tree (`decision_tree` — Tree-Based)
- **A1. Suitability for PathSnap:** Learns interpretable axis-aligned threshold rules on `dist_m`, `dist_gap_to_nearest_m`, and `heading_diff_deg` without requiring feature standardization.
- **A2. Strengths & Limitations:**
  - *Strengths:* Fast inference (`0.27 ms/trip`) and transparent split rules.
  - *Limitations:* Single unbagged tree produces piecewise-constant leaf probabilities with higher Log Loss (`0.2302`) and lower accuracy than the 200-tree Random Forest ensemble.
- **B5. Important Hyperparameters & Initial Values:** `criterion='gini'`, `max_depth=8`, `min_samples_leaf=25`, `class_weight='balanced'`.
- **B6. Loss Function / Objective:** Class-weighted Gini impurity criterion.
- **C7–C9. Experimental Results & Overfitting Analysis:** Achieves **CV Macro F1 = `0.8645 ± 0.0229`**, **Student-Test Top-1 = `0.8449 (84.49%)`**, **Student-Test Macro F1 = `0.8907`**, **Log Loss = `0.2302`**, and **Route Continuity = `0.9307`**. Pruning via `max_depth=8` and `min_samples_leaf=25` prevented overfitting, though single-tree variance limits its ceiling.

---

### Model 4: Distance-Weighted KNN (`knn_distance` — Instance-Based)
- **A1. Suitability for PathSnap:** Classifies candidate arcs via inverse Euclidean distance voting over the $k=15$ nearest historical candidates in standardized 16D feature space.
- **A2. Strengths & Limitations:**
  - *Strengths:* Non-parametric local decision boundary (`Student-Test Macro F1 = 0.8949`).
  - *Limitations:* Highest Log Loss (`0.3065`, due to finite 15-neighbor probability support) and slowest inference (`78.97 ms/trip`, ~20× slower than Random Forest).
- **B5. Important Hyperparameters & Initial Values:** `n_neighbors=15`, `weights='distance'`, `metric='minkowski'`, `p=2`, with in-fold `StandardScaler`.
- **B6. Loss Function / Objective:** Inverse-distance weighted local neighborhood voting.
- **C7–C9. Experimental Results & Overfitting Analysis:** Achieves **CV Macro F1 = `0.8657 ± 0.0191`**, **Student-Test Top-1 = `0.8402 (84.02%)`**, **Student-Test Macro F1 = `0.8949`**, and **Log Loss = `0.3065`**. High query latency and poor tail probability calibration make it inferior to Random Forest and MLP.

---

### Model 5: Logistic Regression (`logistic_regression` — Linear & Probabilistic)
- **A1. Suitability for PathSnap:** Provides a fast, convex linear baseline over standardized geometric and kinematic features.
- **A2. Strengths & Limitations:**
  - *Strengths:* Fastest inference (`0.22 ms/trip`) and lowest fold variance (`±0.0131`).
  - *Limitations:* A single linear hyperplane cannot represent conditional interactions between perpendicular distance and heading alignment at intersections.
- **B5. Important Hyperparameters & Initial Values:** `penalty='l2'`, `C=1.0`, `solver='lbfgs'`, `class_weight='balanced'`, `max_iter=500`, with in-fold `StandardScaler`.
- **B6. Loss Function / Objective:** L2-regularized class-weighted binary cross-entropy.
- **C7–C9. Experimental Results & Overfitting Analysis:** Achieves **CV Macro F1 = `0.8250 ± 0.0131`**, **Student-Test Top-1 = `0.8278 (82.78%)`**, **Student-Test Macro F1 = `0.8417`**, and **Log Loss = `0.2969`**. The consistently lower scores on both CV and Student-Test demonstrate **underfitting** due to the linear model assumption.
