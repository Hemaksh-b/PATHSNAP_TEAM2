# Model Comparison, Selection Rationale & Compulsory Per-Model Analysis (Report §3.1 & §3.2)

**Owner:** Hemaksh Breja — Member 2 (ML & Tuning)  
**Issue:** `#5 [M2] Evaluate models and select the best candidate classifier`

---

## 1. Primary Metric & Selection Rationale

### Why Binary Accuracy Alone Is Inappropriate
In PathSnap candidate classification, each GPS observation generates up to $K$ candidate road arcs ($K \approx 5\text{--}10$), of which at most **1** is the true traversed road segment (`label = 1`) and $K - 1$ are negative candidates (`label = 0`). A trivial classifier that predicts `0` for every candidate achieves $80\text{--}90\%$ binary accuracy while achieving **$0\%$ route-matching accuracy**.

### Primary Selection Criteria
We evaluate and rank the 5 candidate classifiers using four complementary criteria that directly reflect both **pointwise candidate selection** and **sequential HMM/Viterbi decoding**:

1. **Observation-Level Top-1 Accuracy (40% weight):**
   - For each GPS observation $(trip\_id, obs\_idx)$, selects $\hat{c}_t = \arg\max_{c \in C_t} \hat{P}(\text{label}=1 \mid \mathbf{x}_{t,c})$ and compares its `arc_id` against the aligned reference road segment. This directly measures classifier-only map-matching accuracy on unseen trips.
2. **5-Fold GroupKFold Macro F1-Score (25% weight) & Student-Test Macro F1-Score (20% weight):**
   - Balances precision and recall equally across positive and negative candidate classes across all 80 training trips (grouped by `trip_id` so no trip leaks across folds) and on the 10 held-out student-testing trips.
3. **Probability Calibration Quality — Log Loss (15% weight):**
   - The downstream HMM decoder (`hmm_viterbi`) uses emission log-probabilities $\log B_t(c) = \log \max(\hat{P}(\text{label}=1 \mid \mathbf{x}_{t,c}), \epsilon)$. Models with poorly calibrated or overconfident probabilities (high Log Loss, such as uncalibrated single trees or small-$k$ KNN with hard 0/1 neighbor fractions) distort the balance between emission log-probabilities and road-network transition penalties in the Viterbi recurrence.
4. **Route Continuity & Inference Latency:**
   - Measures the fraction of consecutive Top-1 predicted arcs $(\hat{a}_{t-1}, \hat{a}_t)$ that are identical or connected in the road graph, alongside per-trip inference time (`ms/trip`) for real-time FastAPI serving on Laptop 1.

---

## 2. Compulsory Questions for Every Model (Deliverables.pdf §3.2)

### Model 1: Logistic Regression (`logistic_regression` — Linear & Probabilistic)
- **A1. Suitability for PathSnap:** Provides a fast, convex, naturally well-calibrated probabilistic baseline via the logistic link function over standardized spatial and angular features (`dist_m`, `dist_gap_to_nearest_m`, `cos_heading_diff`, etc.).
- **A2. Strengths & Limitations:**
  - *Strengths:* Sub-millisecond inference latency per trip, low variance across `GroupKFold` splits, smooth probability estimates ideal for HMM emissions.
  - *Limitations:* Assumes a linear decision boundary in feature space; cannot capture complex non-linear interactions (e.g., when high perpendicular distance is acceptable only at wide intersections or during high-speed turns) without manual interaction terms.
- **B5. Important Hyperparameters & Initial Values:** `penalty='l2'`, `C=1.0`, `solver='lbfgs'`, `class_weight='balanced'`, `max_iter=500`, preceded by in-fold `SimpleImputer(strategy='median')` and `StandardScaler()`.
- **B6. Loss Function / Objective:** L2-regularized class-weighted binary cross-entropy (negative log-likelihood).
- **C7–C9. Experimental Behaviour (Overfitting/Underfitting):** Minimal gap between 5-fold CV Macro F1 and Student-Test Macro F1 indicates zero overfitting, though its linear boundary slightly underfits complex multi-candidate urban junctions compared to non-linear models.

---

### Model 2: Distance-Weighted K-Nearest Neighbours (`knn_distance` — Instance-Based)
- **A1. Suitability for PathSnap:** Non-parametric local classifier that classifies candidate road segments by comparing their standardized geometric/kinematic feature vectors against similar historical GPS-to-arc configurations.
- **A2. Strengths & Limitations:**
  - *Strengths:* Makes no global functional assumptions; naturally adapts to local density variations in feature space.
  - *Limitations:* Sensitive to irrelevant or correlated features even after `StandardScaler`; requires storing the entire training set in memory and incurs $O(N_{\text{train}} \cdot d)$ query latency at inference time, making it slower for live FastAPI serving.
- **B5. Important Hyperparameters & Initial Values:** `n_neighbors=15`, `weights='distance'`, `metric='minkowski'`, `p=2` (Euclidean distance over standardized features).
- **B6. Loss Function / Objective:** Instance-based inverse-distance weighted voting over the $k=15$ nearest training candidates.
- **C7–C9. Experimental Behaviour (Overfitting/Underfitting):** Using $k=15$ with inverse-distance weighting prevents the severe overfitting and zero-probability clipping seen at $k=1\text{--}3$, yielding moderate Log Loss but higher inference latency than parametric models.

---

### Model 3: Regularized Decision Tree (`decision_tree` — Tree-Based)
- **A1. Suitability for PathSnap:** Learns interpretable, axis-aligned hierarchical threshold rules on perpendicular distance, candidate rank, and heading alignment without requiring feature scaling.
- **A2. Strengths & Limitations:**
  - *Strengths:* Extremely fast inference, invariant to monotonic feature transformations, directly captures threshold interactions (e.g., `cand_rank == 0` AND `heading_diff_deg < 25`).
  - *Limitations:* High variance across trip folds; piecewise-constant leaf probabilities produce coarser probability calibration (higher Log Loss) than smooth ensembles or neural models.
- **B5. Important Hyperparameters & Initial Values:** `criterion='gini'`, `max_depth=8`, `min_samples_leaf=25`, `class_weight='balanced'`.
- **B6. Loss Function / Objective:** Class-weighted Gini impurity reduction at each recursive binary split.
- **C7–C9. Experimental Behaviour (Overfitting/Underfitting):** Constraining `max_depth=8` and `min_samples_leaf=25` prevents leaf memorization of individual training trips, keeping CV and Student-Test Macro F1 closely aligned, though leaf probability quantization increases Log Loss relative to ensemble and neural classifiers.

---

### Model 4: Random Forest (`random_forest` — Ensemble / Bagging)
- **A1. Suitability for PathSnap:** Aggregates 200 decorrelated decision trees trained on bootstrap samples with random feature subsets, capturing non-linear spatial-temporal feature interactions while smoothing leaf probability estimates for HMM emissions.
- **A2. Strengths & Limitations:**
  - *Strengths:* Strong generalization across unseen road networks, high stability across `GroupKFold` splits, robust to outliers in GPS speed or timestamp gaps.
  - *Limitations:* Higher model storage footprint and slower per-trip inference latency than linear or shallow neural models; `class_weight='balanced_subsample'` slightly shifts raw probability calibration toward 0.5 compared to uncalibrated log-loss objectives.
- **B5. Important Hyperparameters & Initial Values:** `n_estimators=200`, `max_depth=12`, `min_samples_leaf=10`, `max_features='sqrt'`, `class_weight='balanced_subsample'`.
- **B6. Loss Function / Objective:** Ensemble average of class-weighted Gini-impurity trees across 200 bootstrap subsamples.
- **C7–C9. Experimental Behaviour (Overfitting/Underfitting):** Bagging and `max_features='sqrt'` substantially reduce fold-to-fold variance compared to a single Decision Tree, achieving high Student-Test Top-1 Accuracy and Route Continuity without overfitting.

---

### Model 5: Shallow Multi-Layer Perceptron (`mlp_neural_net` — Neural Networks)
- **A1. Suitability for PathSnap:** Two-hidden-layer feed-forward network (`64 -> 32` with ReLU activations) trained directly on binary cross-entropy (Log Loss), learning smooth non-linear combinations of spatial distance, fractional arc position, and heading alignment.
- **A2. Strengths & Limitations:**
  - *Strengths:* Directly optimizes Log Loss, producing the best-calibrated posterior probabilities $\hat{P}(\text{label}=1 \mid \mathbf{x}_{t,k})$ for the HMM Viterbi decoder while maintaining sub-millisecond matrix-multiply inference speed.
  - *Limitations:* Requires strict in-fold feature standardization (`StandardScaler`) and early stopping to prevent overfitting on smaller trip subsets.
- **B5. Important Hyperparameters & Initial Values:** `hidden_layer_sizes=(64, 32)`, `activation='relu'`, `alpha=1e-3` (L2 weight decay), `learning_rate_init=1e-3`, `early_stopping=True`, `max_iter=120`.
- **B6. Loss Function / Objective:** L2-regularized binary cross-entropy loss optimized via Adam stochastic gradient descent with internal validation early stopping.
- **C7–C9. Experimental Behaviour (Overfitting/Underfitting):** The epoch-wise training loss and validation accuracy curves (`figures/mlp_training_loss_curve.png`) converge smoothly without divergence, achieving top-tier Student-Test Top-1 Accuracy, Macro F1, and the lowest Log Loss.
