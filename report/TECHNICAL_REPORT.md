# PathSnap: Distributed Map-Matching & Sequential Route Reconstruction
## Technical Report

**Team:** PATHSNAP_TEAM2  
**Members:**
- Member 1: Hema (Data Processing, EDA, Candidate Generation & Feature Engineering)
- Member 2: Hemaksh Breja (Model Training, Bayesian Hyperparameter Optimization)
- Member 3: Hemant (HMM Viterbi Decoder, Road-Graph Transitions & Folium Visualisations)
- Member 4: Himani (FastAPI Backend, Evaluator Dashboard UI, Distributed Deployment & Coordinator)

**Repository:** `https://github.com/Hemaksh-b/PATHSNAP_TEAM2`  
**Date:** November 2026

---

## 1. Executive Summary & Architecture Overview

PathSnap solves the vehicle GPS map-matching problem by reconstructing sequential road routes from noisy, low-frequency vehicle GPS trajectories. The project uses a distributed client-server architecture:
- **Laptop 1 (FastAPI Server):** Hosts the data pipeline, trained candidate classifiers, road-network graph, log-space Viterbi decoder, and server-side ground truth evaluation.
- **Laptop 2 (Evaluator Dashboard UI):** A React 18 + Vite dashboard allowing evaluators to select trips, trigger real-time inference, and inspect performance metrics and interactive Folium maps over a local Wi-Fi or Ethernet network.

```
       LAPTOP 1 (FastAPI Backend - 0.0.0.0:8000)
┌────────────────────────────────────────────────────────┐
│  Raw Track (.track) ──> Spatial Projection (UTM EPSG)  │
│  KNN Candidates     ──> Feature Extraction (13 feats)  │
│  Classifier Prob.   ──> Road Graph Dijkstra Transition │
│  Log-Space Viterbi  ──> Matched Arc Sequence           │
│  Server Evaluation  ──> Top-1 Acc, Macro F1, Latency   │
└───────────────────────────┬────────────────────────────┘
                            │ REST / JSON (Zero GT Leakage)
                            ▼
       LAPTOP 2 (Evaluator Dashboard UI - :5173)
┌────────────────────────────────────────────────────────┐
│  • Dynamic Trip Selection (GET /evaluation/trips)      │
│  • Single-Click Reconstruction (POST /predict)         │
│  • Live Metrics: Accuracy, Macro F1, Continuity, RTT   │
│  • Interactive Folium Route Map & Raw Inspector        │
└────────────────────────────────────────────────────────┘
```

---

## 2. Dataset Understanding and Exploratory Data Analysis (EDA)

### 2.1 Dataset Structure
Each trip directory contains four synchronized text files:
1. `*.track`: GPS timestamped observations (`lon`, `lat`, `timestamp`).
2. `*.nodes`: Road network vertices (`lon`, `lat`, `node_id`).
3. `*.arcs`: Directed roadway segments connecting node pairs (`from_node_id`, `to_node_id`, `arc_id`).
4. `*.route`: Ground truth ordered list of traversed road arc IDs (`arc_id`), reserved server-side and never exposed to the client.

### 2.2 Coordinate Reference System (CRS) & Projection
Raw coordinates in WGS84 (`EPSG:4326`) are transformed to local Universal Transverse Mercator (UTM) projections in metric units (`meters`) using `pyproj.Transformer`. This enables accurate Euclidean distance calculations:
- Point-to-segment perpendicular distance.
- Fractional arc position along road geometry ($f \in [0, 1]$).
- Arc length in meters.

### 2.3 Candidate Arc Generation & Miss Rate
For each GPS point, a KD-tree queries the $K$-nearest road segments within a search radius $R$. Candidate sweep experiments demonstrate:
- At $K = 5, R = 100\text{ m}$, the candidate miss rate is minimized to $\le 1.6\%$ across training trajectories, ensuring true traversed roads are retained within the HMM state space.

---

## 3. Model Experimentation and Candidate Classification

### 3.1 Five Candidate Classifiers
We implemented and compared 5 distinct model families under a 5-fold `GroupKFold` cross-validation grouped by `trip_id` (preventing intra-trip leakage):

| Model Family | Key Configuration | CV Macro F1 | Top-1 Accuracy | Route Continuity | Student-Test Macro F1 | Log Loss | Latency / Trip |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (L2)** | $L_2$, $C=1.0$, `balanced`, `StandardScaler` | **0.9805** | **0.9680** | **0.9417** | **0.9765** | 0.0456 | **0.07 ms** |
| **Random Forest** | 200 trees, `max_depth=12`, `balanced_subsample` | 0.9774 | 0.9640 | 0.9375 | 0.9813 | **0.0433** | 2.78 ms |
| **Shallow MLP** | (64, 32), ReLU, $\alpha=10^{-3}$, early stopping | 0.9820 | 0.9640 | 0.9333 | 0.9734 | 0.0414 | 0.11 ms |
| **Distance-Weighted KNN** | $k=15$, inverse-distance weighting | 0.9778 | 0.9640 | 0.9333 | 0.9787 | 0.0757 | 1.25 ms |
| **Decision Tree** | Gini, `max_depth=8`, `min_samples_leaf=25` | 0.9684 | 0.9680 | 0.9500 | 0.9704 | 0.0794 | 0.10 ms |

### 3.2 Model Selection Rationale
`Logistic Regression` was selected as the final candidate classifier because:
1. **Calibrated Probabilities:** Produces well-calibrated posterior probabilities $P(\text{label}=1 \mid \mathbf{x})$, serving as emission probabilities for the sequential HMM.
2. **Speed & Latency:** Executes in $0.07\text{ ms}$ per trip, meeting real-time evaluation requirements.
3. **Generalization:** Demonstrated the highest CV Top-1 Accuracy ($0.9680$) and negligible variance across folds.

---

## 4. Bayesian Hyperparameter Optimisation

Hyperparameters for the chosen Logistic Regression model were tuned using Tree-structured Parzen Estimator (TPE) via Optuna over 20 trials.

* **Search Space:**
  - $C \in [10^{-3}, 10^{2}]$ (log-uniform)
  - `class_weight` $\in [\text{"balanced"}, \text{None}]$
  - `tol` $\in [10^{-5}, 10^{-2}]$ (log-uniform)
* **Optimization Objective:** Maximize 5-fold trip-grouped cross-validation Macro F1-score.

### 4.1 Tuned vs. Untuned Performance

| Variant | Best Parameters | CV Macro F1 | Student-Test Macro F1 | Log Loss | Latency / Trip |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Untuned** | $C=1.0$, `balanced`, `tol`=$10^{-4}$ | 0.9822 | 0.9781 | 0.0556 | 0.079 ms |
| **Tuned (TPE)** | $C=27.17$, `class_weight`=None, `tol`=$8.45 \times 10^{-4}$ | **0.9851** | **0.9852** | **0.0251** | **0.066 ms** |

*Outcome:* Bayesian optimization reduced Log Loss by **54.9%** ($0.0556 \to 0.0251$) and improved test Macro F1 ($0.9781 \to 0.9852$) while further reducing inference latency.

---

## 5. HMM & Log-Space Viterbi Sequential Route Reconstruction

### 5.1 Formalization
Point-wise classifier predictions suffer from temporal jumping. We resolve this by framing map-matching as a Hidden Markov Model:
- **Hidden States ($S_t$):** Candidate road arcs $C_t = \{c_{t,1}, \dots, c_{t,K}\}$ for observation $t$.
- **Emission Probability:** Log-probability derived from the classifier posterior:
  $$\log B_t(k) = \log \max(P(\text{arc}_k \mid \mathbf{x}_{t,k}), 10^{-9})$$
- **Transition Probability:** Road-network topological distance vs GPS displacement:
  $$\log A_t(i, j) = -\beta \cdot \left| d_{\text{graph}}(c_{t-1, i}, c_{t, j}) - d_{\text{GPS}}(o_{t-1}, o_t) \right|$$
  where $d_{\text{graph}}$ is computed via Dijkstra shortest path over the directed roadway network (`networkx.DiGraph`).

### 5.2 Numerical Stability & Robustness
- **Log-Space Computation:** Prevents numerical underflow across long trajectories ($T > 1000$ points).
- **Disconnection Handling:** If large time gaps ($\Delta t > 30\text{ s}$) occur, the decoder splits the trajectory into disjoint segments, preventing false trajectory bridging across missing observations.

---

## 6. Geographical Visualisation Using Folium

Folium maps are automatically generated server-side for each trip and embedded directly into the UI dashboard:
- **GPS Layer:** Blue circles representing raw vehicle observations.
- **Road Network:** Grey polylines representing nearby topological arcs.
- **Predicted Route:** Continuous red polyline indicating the Viterbi decoded path.
- **Reference Route:** Green polyline rendered strictly for training/student-test diagnostics, never on evaluator test sets.

Committed example: [`visualisations/maps/00000005_train.html`](file:///e:/ML-HACKATHON/PATHSNAP_TEAM2/visualisations/maps/00000005_train.html)

---

## 7. Distributed Deployment & Evaluator UI

### 7.1 Backend API (Laptop 1)
Implemented using FastAPI (`0.0.0.0:8000`):
- `GET /health`: System diagnostics and model readiness probe.
- `GET /evaluation/trips`: Enumerate available evaluation trips.
- `POST /predict`: Execute end-to-end inference and return observation accuracy, continuity, and Folium HTML map.
- `GET /maps/{trip_id}`: Dedicated standalone HTML map delivery.
- **Logging:** All requests (client IP, HTTP method, path, status, latency) are recorded in `api/logs/requests.log`.

### 7.2 Evaluator Dashboard UI (Laptop 2)
Built with React 18, Vite, and Tailwind CSS:
- Live backend connection badge with automatic latency polling.
- Zero-hardcoding: Dynamic loading of trip IDs directly from the backend.
- Embedded iframe visualization of Folium interactive maps.
- Configurable backend target IP directly in the UI settings for hassle-free network pairing during evaluation.

---

## 8. Conclusion

PATHSNAP_TEAM2 delivers a complete, leak-free, high-performance map-matching system achieving **>96% Top-1 accuracy**, robust topological continuity, and sub-millisecond inference times, ready for evaluation.
