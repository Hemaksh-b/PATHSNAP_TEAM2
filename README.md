# PathSnap: Distributed Map-Matching & Route Reconstruction

End-to-end machine learning + HMM/Viterbi map-matching system with a Two-Laptop Distributed Deployment (FastAPI Backend + Evaluator UI).

## Repository Structure

| Folder | Required Component | Owner / Role |
| :--- | :--- | :--- |
| `data_processing/` | Raw data inspection, cleaning, coordinate projection, and candidate generation | Member 1 (Data & Features) |
| `feature_engineering/` | Spatial, geometric, and temporal feature extraction | Member 1 (Data & Features) |
| `model_training/` | Training scripts/notebooks for >= 5 ML models across algorithm families | Member 2 (ML & Tuning) |
| `model_evaluation/` | Group-aware CV, student-test evaluation, metrics, and comparison plots | Member 2 (ML & Tuning) |
| `hyperparameter_optimisation/` | Bayesian hyperparameter search configs, trial logs, and tuned model | Member 2 (ML & Tuning) |
| `hmm_viterbi/` | Hidden Markov Model & Viterbi sequential route reconstruction | Member 3 (HMM & Vis) |
| `visualisations/` | Interactive Folium maps and diagnostic/comparison plots | Member 3 (HMM & Vis) |
| `api/` | Laptop 1: FastAPI backend hosting trained classifier, HMM/Viterbi, and evaluator pipeline | Member 4 (Systems & Report) |
| `ui/` | Laptop 2: Evaluator-facing user interface displaying the 10 evaluation trips & route maps | Member 4 (Systems & Report) |
| `deployment/` | Two-laptop network configuration, startup scripts, and instructions | Member 4 (Systems & Report) |
| `artifacts/` | Exported model weights, feature scalers, and HMM configs | Shared |
| `supporting_results/` | Experiment CSVs, trial logs, and comparison tables | Shared |
| `report/` | Consolidated Technical Report (PDF) and assets | All Members |

## Quickstart

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Laptop 1 — Start FastAPI Backend
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Laptop 2 — Start Evaluator UI
```bash
streamlit run ui/app.py
```
