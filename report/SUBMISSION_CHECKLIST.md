# PathSnap – Final Hackathon Submission Checklist

**Team:** PATHSNAP_TEAM2  
**Coordinator / Member 4:** Himani  
**Repository:** `https://github.com/Hemaksh-b/PATHSNAP_TEAM2`  
**Reference Specification:** *Deliverables.pdf*

---

## 1. Codebase & Deliverable 1 Verification

| Item | Requirement | Status | Verification Path / Details |
| :--- | :--- | :---: | :--- |
| **Public GitHub Repo** | Code committed and pushed to public repository | [x] | `https://github.com/Hemaksh-b/PATHSNAP_TEAM2` |
| **Data Processing** | Loading, UTM projection, K-nearest candidate generation, miss rate sweep | [x] | `data_processing/loader.py`, `candidates.py`, `projection.py`, `miss_rate.py` |
| **Candidate Models (>= 5)** | 5 distinct model families trained & evaluated with GroupKFold | [x] | `model_training/models.py`, `train.py`, `schema.py` |
| **Model Comparison** | Consolidated comparison table (Macro F1, Top-1 Acc, Log Loss, Latency) | [x] | `model_evaluation/results/model_comparison.csv` |
| **Bayesian Tuning** | Optuna TPE Bayesian optimization on winning model family | [x] | `hyperparameter_optimisation/optimize.py`, `results/` |
| **Saved Model Weights** | Exported final tuned model artifact & model card | [x] | `artifacts/model.joblib`, `artifacts/model_card.json` |
| **HMM / Viterbi Decoder** | Log-space Viterbi sequential decoder with road-network transitions | [x] | `hmm_viterbi/viterbi.py`, `transitions.py`, `emissions.py` |
| **Folium Route Maps** | Interactive HTML maps with toggleable GPS, road, and predicted layers | [x] | `visualisations/route_map.py`, `maps/00000005_train.html` |
| **FastAPI Backend (Laptop 1)** | REST endpoints (`/health`, `/trips`, `/predict`, `/maps/{id}`) binding `0.0.0.0` | [x] | `api/main.py`, `pipeline.py`, `evaluation.py`, `api/README.md` |
| **Evaluator UI (Laptop 2)** | React + Vite + Tailwind dashboard with dynamic trips & Folium viewer | [x] | `ui/src/`, `ui/package.json`, `ui/README.md` |
| **Ground-Truth Isolation** | Reference routes strictly isolated server-side; zero UI leakage | [x] | `api/evaluation.py` enforces label isolation |
| **Request Logging** | Evaluator request logs persisted to terminal and file | [x] | `api/logs/requests.log` logged per request |

---

## 2. Distributed Live Evaluation Runbook (Two Laptops)

### Step 1: Laptop 1 Setup (FastAPI Backend & ML Pipeline)
```bash
# 1. Clone repository
git clone https://github.com/Hemaksh-b/PATHSNAP_TEAM2.git
cd PATHSNAP_TEAM2

# 2. Start FastAPI inference server (bind 0.0.0.0 on port 8000)
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```
* Note Laptop 1's local IP address using `ipconfig` (e.g. `192.168.1.50`).
* Health check: verify `http://localhost:8000/health`.

### Step 2: Laptop 2 Setup (Evaluator Dashboard UI)
```bash
cd PATHSNAP_TEAM2/ui
npm install
npm run dev
```
* Open `http://localhost:5173` in a web browser.
* In the **System Status** page, enter Laptop 1's IP (`http://192.168.1.50:8000`) and click **Update & Probe**.
* Once the indicator shows **Backend Online**, navigate to **Dashboard**.
* Select any evaluation trip and click **Reconstruct Route**.
* Observe the live observation accuracy, route continuity, latency, and interactive Folium trajectory.

---

## 3. Mandatory Academic Integrity Statement
* All models and figures in the report originate from real code executions and logged data in `supporting_results/`, `model_evaluation/`, and `hyperparameter_optimisation/`.
* No mock values or hardcoded evaluation responses are used during live evaluation mode.
