# PathSnap: Distributed Map-Matching & Route Reconstruction

End-to-end machine learning + HMM/Viterbi map-matching system with a Two-Laptop Distributed Deployment (FastAPI Backend + Evaluator UI).

## Team Details

| Member | Name | SRN | Role | GitHub |
| :--- | :--- | :--- | :--- | :--- |
| Member 1 | Hemavathi E | PES2UG24CS193 | Data & Features | [@Hema3156](https://github.com/Hema3156) |
| Member 2 | Hemaksh Breja | PES2UG24CS191 | ML & Tuning | [@Hemaksh-b](https://github.com/Hemaksh-b) |
| Member 3 | Hemant Ram | PES2UG24CS192 | HMM & Vis | [@Hemz1416](https://github.com/Hemz1416) |
| Member 4 | Himani Prabhu | PES2UG24CS194 | Systems & Report | [@Himani-ai](https://github.com/Himani-ai) |

## Repository Structure

| Folder | Required Component | Owner / Role |
| :--- | :--- | :--- |
| `data_processing/` | Raw data inspection, cleaning, coordinate projection, and candidate generation | Hemavathi E (Data & Features) |
| `feature_engineering/` | Spatial, geometric, and temporal feature extraction | Hemavathi E (Data & Features) |
| `model_training/` | Training scripts/notebooks for >= 5 ML models across algorithm families | Hemaksh Breja (ML & Tuning) |
| `model_evaluation/` | Group-aware CV, student-test evaluation, metrics, and comparison plots | Hemaksh Breja (ML & Tuning) |
| `hyperparameter_optimisation/` | Bayesian hyperparameter search configs, trial logs, and tuned model | Hemaksh Breja (ML & Tuning) |
| `hmm_viterbi/` | Hidden Markov Model & Viterbi sequential route reconstruction | Hemant Ram (HMM & Vis) |
| `visualisations/` | Interactive Folium maps and diagnostic/comparison plots | Hemant Ram (HMM & Vis) |
| `api/` | Laptop 1: FastAPI backend hosting trained classifier, HMM/Viterbi, and evaluator pipeline | Himani Prabhu (Systems & Report) |
| `ui/` | Laptop 2: Evaluator-facing user interface displaying the 10 evaluation trips & route maps | Himani Prabhu (Systems & Report) |
| `deployment/` | Two-laptop network configuration, startup scripts, and instructions | Himani Prabhu (Systems & Report) |
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
export PATHSNAP_MOCK=0
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 3. Laptop 2 — Start Evaluator UI (React + Vite)
```bash
cd ui
npm install
npm run dev
```

For complete from-scratch setup, firewall configuration, and two-laptop viva instructions, see [deployment/TWO_LAPTOP_EXECUTION_GUIDE.md](deployment/TWO_LAPTOP_EXECUTION_GUIDE.md).
