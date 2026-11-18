# Bayesian Hyperparameter Optimisation & Final Model Export (`hyperparameter_optimisation/`, `artifacts/`)

**Owner:** Hemaksh Breja — Member 2 (ML & Tuning)  
**Issue:** `#6 [M2] Perform Bayesian hyperparameter optimization and export final model`

---

## 1. Overview & Report §4 Answers

After comparing the 5 candidate classifiers in `model_evaluation/` (`#5`), this module tunes the selected model using **Tree-structured Parzen Estimator (TPE) Bayesian Optimisation** with **5-fold `GroupKFold` cross-validation** grouped by `trip_id` on the 80 training trips.

### Report §4 Compulsory Questions
1. **Which hyperparameters were included in the search?**
   - Defined and justified per model in [search_space_config.json](file:///Users/hemakshbreja/ML_HACKATHON/hyperparameter_optimisation/search_space_config.json):
     - For `logistic_regression`: inverse L2 regularization `C`, `class_weight`, and convergence tolerance `tol`.
     - For `random_forest`: `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`, and `class_weight`.
     - For `mlp_neural_net`: `hidden_layer_1`, `hidden_layer_2`, L2 weight decay `alpha`, and `learning_rate_init`.
2. **What ranges or distributions were considered?**
   - Log-uniform distributions for scale parameters (`C` in $[10^{-3}, 10^{2}]$, `alpha` in $[10^{-5}, 10^{-1}]$, `tol` in $[10^{-5}, 10^{-2}]$), discrete integer step distributions for structural capacity (`max_depth` in $[4, 20]$, `min_samples_leaf` in $[2, 40]$, `n_estimators` in $[80, 350]$), and categorical priors for weighting schemes (`class_weight`).
3. **What objective function was optimised and why?**
   - Inside each trial, we run 5-fold `GroupKFold` grouped by `trip_id` on the 80 training trips and maximize:
     $$\mathcal{J}(\theta) = 0.50 \cdot \text{Top1Acc}_{\text{CV}} + 0.35 \cdot \text{MacroF1}_{\text{CV}} + 0.15 \cdot \frac{1}{1 + \text{LogLoss}_{\text{CV}}}$$
   - This objective balances direct observation-level route matching ($\text{Top1Acc}$), class-imbalance discrimination ($\text{MacroF1}$), and well-calibrated posterior probabilities ($\text{LogLoss}$) required for HMM emission log-probabilities $\log P(o_t \mid c_{t,k})$.
4. **How are trials persisted?**
   - Every trial's parameters, fold metrics, objective score, and wall-clock duration are persisted to both SQLite (`hyperparameter_optimisation/results/optuna_study.db`) and CSV (`hyperparameter_optimisation/results/optimization_trials.csv`).
5. **Untuned vs. Tuned Comparison:**
   - Stored in `hyperparameter_optimisation/results/tuned_vs_untuned_comparison.csv` and plotted in `hyperparameter_optimisation/figures/tuned_vs_untuned_comparison.png` across CV Macro F1, Student-Test Top-1 Accuracy, Student-Test Macro F1, Log Loss, Route Continuity, and Inference Time.

---

## 2. Reproducible CLI Commands

### A. Harness Mode (`--mock`, before Issue `#3` merges)
```bash
# Optimize the selected model from Issue #5 (or specify --model random_forest / logistic_regression / mlp_neural_net)
python -m hyperparameter_optimisation.optimize --mock --n-trials 20 --seed 42
```

### B. Real Feature Matrix Mode (once Issue `#3` merges)
```bash
python -m hyperparameter_optimisation.optimize \
    --train-features feature_engineering/results/train_features.csv \
    --student-test-features feature_engineering/results/student_test_features.csv \
    --n-trials 30 \
    --seed 42
```

### C. Run Unit & Load-and-Predict Smoke Tests
```bash
pytest hyperparameter_optimisation/tests/test_optimization.py -v
```

---

## 3. FastAPI & HMM Integration Instructions (for Members 3 & 4)

The final refitted model pipeline (including in-pipeline `SimpleImputer` and `StandardScaler`) and its metadata card are exported to:
- `artifacts/model.joblib`
- `artifacts/model_card.json`

### Loading & Predicting in FastAPI (`api/`) or HMM (`hmm_viterbi/`)
```python
from pathlib import Path
from hyperparameter_optimisation.inference import load_exported_model, predict_candidate_probs

# 1. Load once at FastAPI startup (#10):
pipeline, model_card = load_exported_model(Path("artifacts"))

# 2. Score candidate road arcs for any trip:
# Returns DataFrame with exact columns: ['trip_id', 'obs_idx', 'arc_id', 'prob']
scores_df = predict_candidate_probs(
    candidates_df,
    pipeline=pipeline,
    feature_columns=model_card["feature_columns"],
)
```
