# PathSnap: End-to-End Execution & Two-Laptop Deployment Guide

This guide explains how to set up **both laptops from scratch**, run the full ML + HMM/Viterbi pipeline locally, and operate the **Two-Device Distributed System** during your live viva demonstration.

---

## 1. System Architecture Overview

```text
┌─────────────────────────────────────────────┐          Local Wi-Fi / Hotspot          ┌─────────────────────────────────────────────┐
│           LAPTOP 1 (Backend Server)         │ ◄────── GET /health, /evaluation/trips  │         LAPTOP 2 (Evaluator Client)         │
│                                             │ ◄────── POST /predict {trip_id}         │                                             │
│  • FastAPI + Uvicorn (0.0.0.0:8000)         │                                         │  • React 18 + Vite Evaluator Dashboard    │
│  • Data Loader + UTM Projection + Candidates│ ──────► Predicted Arc IDs, Latency,     │    (http://localhost:5173)                  │
│  • Tuned Random Forest (artifacts/model)    │         Folium HTML Map & Server Metrics│  • Never accesses .route ground truth     │
│  • HMM / Viterbi Decoder (artifacts/hmm)    │                                         │  • Configurable Laptop 1 IP in System Status│
│  • Server-side Evaluation + requests.log    │                                         │                                             │
└─────────────────────────────────────────────┘                                         └─────────────────────────────────────────────┘
```

- **Laptop 1** hosts the dataset (`handouts/Student folder/`), exported model (`artifacts/model.joblib`), HMM configuration (`artifacts/hmm_config.json`), FastAPI server (`api/main.py`), and server-side evaluation metrics.
- **Laptop 2** runs the Evaluator Dashboard (`ui/`), fetches trip IDs from Laptop 1, triggers `POST /predict`, and renders the returned Folium map and metrics. **Ground-truth `.route` files never leave Laptop 1.**

---

## 2. Network Prerequisites (Do This First)

1. **Connect BOTH Laptop 1 and Laptop 2 to the exact same network.**
   - **Recommended at Hackathon Venues:** Create a **mobile phone Wi-Fi hotspot** and connect both laptops to it. Campus/university Wi-Fi networks frequently enable *AP Client Isolation*, which blocks laptop-to-laptop traffic on port `8000`.
2. Make sure neither laptop is connected to a VPN.

---

## 3. Setting Up Laptop 1 (FastAPI Inference & Evaluation Server)

### Step 3.1 — Prerequisites
- **Python 3.11 – 3.13** (`python3 --version`)
- **Git** (`git --version`)

### Step 3.2 — Clone Repo & Install Python Dependencies
```bash
git clone https://github.com/Hemaksh-b/PATHSNAP_TEAM2.git
cd PATHSNAP_TEAM2

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows PowerShell

pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** `requirements.txt` pins `scikit-learn==1.7.2` so it matches the serialized `artifacts/model.joblib` cleanly without version-mismatch warnings.

### Step 3.3 — Place the Dataset Locally
`handouts/` is `.gitignore`d to keep the public repo clean. Ensure the dataset folder exists locally at:
```text
PATHSNAP_TEAM2/
└── handouts/
    └── Student folder/
        ├── train/           # 80 training trips (00000001 .. 00000099)
        ├── student_test/    # 10 student-testing trips
        └── evaluator_test/  # 10 reserved evaluator trips (place here on demo day)
```
*(Before `evaluator_test/` is placed on disk, `api/settings.py` automatically falls back to `student_test/` so you can test the full 10-trip flow end-to-end ahead of time. You can also override the folder explicitly via `PATHSNAP_TRIPS_DIR`.)*

### Step 3.4 — Verify Artifacts & Run Smoke Tests
```bash
# Confirm real (non-mock) model & HMM config exist in artifacts/
ls -lh artifacts/
# Expected: model.joblib (~21 MB), model_card.json ("is_mock_data": false), hmm_config.json

# Run unit tests (no dataset required for API mock tests)
python -m pytest api/ model_training/ model_evaluation/ hyperparameter_optimisation/ hmm_viterbi/ -q
```

### Step 3.5 — Find Laptop 1's Local IP Address
Run on **Laptop 1**:
- **macOS:**
  ```bash
  ipconfig getifaddr en0
  ```
- **Linux:**
  ```bash
  hostname -I | awk '{print $1}'
  ```
- **Windows (CMD / PowerShell):**
  ```cmd
  ipconfig
  ```
  Look for **IPv4 Address** under *Wireless LAN adapter Wi-Fi* (e.g., `192.168.1.45` or `172.20.10.2`).

### Step 3.6 — Allow Port `8000` Through Firewall (If Enabled)
- **macOS:** *System Settings → Network → Firewall* — either allow incoming connections for Python/`uvicorn`, or temporarily turn off the macOS firewall during the demo.
- **Windows (Admin PowerShell):**
  ```powershell
  New-NetFirewallRule -DisplayName "PathSnap FastAPI 8000" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow
  ```
- **Linux (`ufw`):**
  ```bash
  sudo ufw allow 8000/tcp
  ```

### Step 3.7 — Start the FastAPI Server on Laptop 1
```bash
cd PATHSNAP_TEAM2
source .venv/bin/activate        # or .venv\Scripts\activate on Windows

export PATHSNAP_MOCK=0           # IMPORTANT: must be 0 (real pipeline) for the demo!
# Optional if pointing to a custom trips folder:
# export PATHSNAP_TRIPS_DIR="handouts/Student folder/evaluator_test"

uvicorn api.main:app --host 0.0.0.0 --port 8000
```
**Sanity Check on Laptop 1:** Open `http://localhost:8000/health` in Laptop 1's browser. You should see:
```json
{"status": "ok", "mock": false, "trips_dir": "...", "n_trips": 10, "emission_source": "..."}
```
> **Keep this terminal visible on Laptop 1's screen during the viva** — evaluators inspect the incoming `POST /predict` logs printed here (also saved to `api/logs/requests.log`).

---

## 4. Setting Up Laptop 2 (Evaluator Dashboard UI)

### Step 4.1 — Prerequisites
- **Node.js v18+** and **npm v9+** (`node -v && npm -v`)
- **Git** (`git --version`)

### Step 4.2 — Clone Repo & Install UI Dependencies
*(Laptop 2 does **not** need Python, the dataset, or `artifacts/` — it only needs the `ui/` folder.)*
```bash
git clone https://github.com/Hemaksh-b/PATHSNAP_TEAM2.git
cd PATHSNAP_TEAM2/ui

npm install
```
> **Windows Users:** `ui/package.json` includes `@rollup/wasm-node` and a `postinstall` script so Vite runs smoothly even if Windows AppLocker blocks native `.node` binaries.

### Step 4.3 — Configure Laptop 1's Backend URL
Create `ui/.env` pointing to Laptop 1's IP address from Step 3.5 (replace `<LAPTOP_1_IP>`):
```bash
cp .env.example .env
# Edit .env so it contains:
# VITE_API_BASE_URL=http://<LAPTOP_1_IP>:8000
```
*(For example: `VITE_API_BASE_URL=http://172.20.10.2:8000`)*

### Step 4.4 — Start the Evaluator UI on Laptop 2
```bash
cd PATHSNAP_TEAM2/ui
npm run dev
```
Open **`http://localhost:5173`** in Laptop 2's browser.

### Step 4.5 — Connect / Verify Two-Laptop Link in the Browser
1. On Laptop 2's browser (`http://localhost:5173`), open **System Status** in the left sidebar.
2. In **FastAPI Backend IP Configuration (Laptop 1)**, confirm or type `http://<LAPTOP_1_IP>:8000` and click **Update & Probe**.
   *(This updates the target URL live in browser `localStorage` without even needing to restart Vite!)*
3. Verify the status badge turns green (**FastAPI Online**) and displays round-trip **Network Latency (ms)**.
4. Optionally test the contract probes on that page (`/health`, `/evaluation/trips`, `/maps/{trip_id}`).
5. Switch back to **Dashboard** in the sidebar.

---

## 5. Live Viva Evaluation Workflow

1. **Laptop 1** is running `uvicorn api.main:app --host 0.0.0.0 --port 8000` with its terminal window visible.
2. **Laptop 2** has the Evaluator Dashboard open at `http://localhost:5173` showing **FastAPI Online** and the **10 Evaluation Trips** loaded from `GET /evaluation/trips`.
3. Ask the evaluator to **select any trip** on Laptop 2 and click **Run Reconstruction**.
4. **Laptop 2** sends `POST http://<LAPTOP_1_IP>:8000/predict` with `{"trip_id": "<selected_id>"}`.
5. **Laptop 1** runs the full pipeline (`load_trip(include_route=False)` $\to$ UTM projection $\to$ $K$-candidates $\to$ 16 features $\to$ ML classifier / HMM Viterbi $\to$ Folium HTML $\to$ post-prediction server-side scoring) and logs the request:
   ```text
   2026-11-18 15:00:12 | 172.20.10.5 POST /predict -> 200 (285 ms)
   2026-11-18 15:00:12 | predict trip=00000007 obs=1354 matched=1354 segments=1 latency=285 ms mock=False
   ```
6. **Laptop 2** displays the **Observation Accuracy, Route Precision/Recall/F1, Route Continuity, Inference Latency**, and the interactive **Folium Route Map** (with pan/zoom, fullscreen, and layer toggles).

---

## 6. Reproducing the Full Offline ML + HMM Pipeline (Laptop 1 Only)

If an evaluator asks you to re-run training, model evaluation, Bayesian optimization, or HMM tuning from raw data:

```bash
source .venv/bin/activate

# 1. Data quality audit & EDA (Member 1)
python -m data_processing.eda

# 2. Extract candidate features & aligned labels for 80 train + 10 student_test trips
python -m model_training.real_features
# (or: python -m feature_engineering.pipeline --partition all)

# 3. Train all 5 models with 5-fold GroupKFold CV (Member 2 — Issue #4)
python -m model_training.train \
    --all \
    --features-path feature_engineering/results/train_features.csv \
    --seed 42

# 4. Evaluate all 5 models on 10 student-test trips & generate §3.3 charts (Member 2 — Issue #5)
python -m model_evaluation.evaluate \
    --train-features feature_engineering/results/train_features.csv \
    --student-test-features feature_engineering/results/student_test_features.csv \
    --seed 42

# 5. Run TPE Bayesian optimization & export artifacts/model.joblib (Member 2 — Issue #6)
python -m hyperparameter_optimisation.optimize \
    --model random_forest \
    --train-features feature_engineering/results/train_features.csv \
    --student-test-features feature_engineering/results/student_test_features.csv \
    --n-trials 15 \
    --seed 42

# 6. Tune HMM decoder & run Pointwise vs. HMM comparison (Member 3 — Issues #7/#8/#9)
python -m hmm_viterbi.run_experiments --tune --tune-trips 20
python -m hmm_viterbi.run_experiments --emissions classifier
python -m visualisations.comparison
```

---

## 7. Quick Troubleshooting Checklist

| Symptom | Cause | Fix |
| :--- | :--- | :--- |
| Laptop 2 shows **Disconnected** / request times out | Campus Wi-Fi blocks peer-to-peer traffic (AP isolation) or Laptop 1 firewall blocks port `8000` | Switch both laptops to a **mobile hotspot**; verify `curl http://<LAPTOP_1_IP>:8000/health` from Laptop 2's browser; allow port `8000` in firewall. |
| `uvicorn` only works on `localhost` and refuses Laptop 2 | Server started without `--host 0.0.0.0` | Restart with `uvicorn api.main:app --host 0.0.0.0 --port 8000`. |
| UI still hits `localhost:8000` after editing `.env` | Browser `localStorage` cached an old URL | Go to **System Status** in the UI, enter `http://<LAPTOP_1_IP>:8000`, and click **Update & Probe** (or click **Reset Default**). |
| `/trips` returns empty list `[]` | Dataset not placed in `handouts/Student folder/` | Verify `handouts/Student folder/evaluator_test` (or `student_test`) contains trip folders, or set `PATHSNAP_TRIPS_DIR`. |
| `Address already in use` on port `8000` | Previous `uvicorn` still running | Stop old process (`pkill -f uvicorn` on macOS/Linux, or use `--port 8001`). |
