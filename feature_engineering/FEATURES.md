# Candidate Feature Engineering and Ground-Truth Alignment Schema (Issue #3)

This document specifies the feature extraction pipeline, observation-to-reference-route alignment, schema definitions, physical units, and leakage guarantees for PathSnap candidate road-segment classification.

---

## 1. Candidate Feature Schema

For every GPS observation $o_t$ in a trip, the candidate generator retrieves up to $K = 5$ candidate road segments $\{c_{t, 1}, \dots, c_{t, K}\}$ within search radius $R = 100\text{ m}$.
The feature extractor computes **16 spatial, geometric, kinematic, and topological features** per candidate pair $(o_t, c_{t, k})$:

| Feature Name | Type | Unit | Range | Mathematical Definition / Description |
|:---|:---|:---|:---|:---|
| `dist_m` | float | metres (m) | $[0.0, 100.0]$ | Orthogonal/perpendicular distance from projected GPS point $(x_t, y_t)$ to candidate arc segment $u \to v$. |
| `cand_rank` | int | integer | $[0, K-1]$ | Candidate proximity rank for this observation ($0 = \text{nearest}, 1 = \text{2nd nearest}, \dots$). |
| `dist_gap_to_nearest_m` | float | metres (m) | $[0.0, 100.0]$ | Proximity margin: $\text{dist}_m - \min_{k} \text{dist}_{m, k}$ ($0.0$ for nearest candidate). |
| `dist_ratio_to_nearest` | float | dimensionless | $[1.0, \infty)$ | Ratio: $\frac{\text{dist}_m + 1.0}{\min_{k} \text{dist}_{m, k} + 1.0}$, smoothed to avoid division by zero. |
| `frac_along` | float | dimensionless | $[0.0, 1.0]$ | Fractional projection coordinate $\alpha$ along segment $u \to v$ ($0.0 = \text{node } u$, $1.0 = \text{node } v$). |
| `dist_to_start_m` | float | metres (m) | $[0.0, L_{\text{arc}}]$ | Metric distance along segment from source node $u$: $\alpha \times \text{arc\_length\_m}$. |
| `dist_to_end_m` | float | metres (m) | $[0.0, L_{\text{arc}}]$ | Metric distance along segment to destination node $v$: $(1.0 - \alpha) \times \text{arc\_length\_m}$. |
| `arc_length_m` | float | metres (m) | $(0.0, \infty)$ | Geometric Euclidean length of the candidate road segment: $\sqrt{(x_v - x_u)^2 + (y_v - y_u)^2}$. |
| `gps_speed_mps` | float | m/s | $[0.0, \infty)$ | Instantaneous GPS speed from neighbouring points: $\frac{\|(x_{t+1}, y_{t+1}) - (x_{t-1}, y_{t-1})\|}{t_{t+1} - t_{t-1}}$. |
| `dt_prev_s` | float | seconds (s) | $(0.0, \infty)$ | Sampling time interval from previous observation: $t_t - t_{t-1}$. |
| `dt_next_s` | float | seconds (s) | $(0.0, \infty)$ | Sampling time interval to next observation: $t_{t+1} - t_t$. |
| `heading_diff_deg` | float | degrees (°) | $[0.0, 180.0]$ | Acute angular difference between vehicle velocity vector and candidate arc bearing. |
| `cos_heading_diff` | float | dimensionless | $[-1.0, 1.0]$ | Directional alignment: $\cos(\text{heading\_diff\_deg})$. ($+1.0 = \text{aligned}$, $-1.0 = \text{opposed}$). |
| `sin_heading_diff` | float | dimensionless | $[0.0, 1.0]$ | Orthogonal misalignment: $\sin(\text{heading\_diff\_deg})$. |
| `out_degree_v` | int | count | $[0, \infty)$ | Road-network out-degree of destination node $v$ (indicator of upcoming junction choices). |
| `in_degree_u` | int | count | $[0, \infty)$ | Road-network in-degree of source node $u$ (indicator of converging road complexity). |

---

## 2. Ground-Truth Route Alignment Methodology

In PathSnap, `.route` provides the chronological sequence of traversed road arcs, not a point-by-point per-observation label. To derive clean training targets:

1. **Monotonic Forward Window Matching**:
   Vehicles traverse the road network unidirectionally in time. Given route arcs $R = [a_0, a_1, \dots, a_{M-1}]$, for each GPS observation $o_t$ we search within a forward-progressing window $[j_{\text{curr}}, \min(j_{\text{curr}} + 15, M-1)]$ to find the nearest route arc. The route index advances monotonically: $j_t \ge j_{t-1}$.

2. **Explicit Uncertainty Tagging (`label_uncertain = True`)**:
   We do not force unreliable labels. An observation is marked uncertain if:
   - **Distance threshold**: Perpendicular distance to the matched route arc exceeds **35.0 metres** (indicates GPS multipath spikes, off-network segments, or parking maneuvers).
   - **Intersection ambiguity**: Multiple candidate arcs in the route sequence lie within **3.0 metres** of each other and distance $> 5.0\text{ m}$ (ambiguous junction turns).
   Downstream models (Member 2) can filter these via `validate_and_prepare_features(df, drop_uncertain_labels=True)` so classifiers learn from clean, high-confidence labels.

3. **Label Assignment**:
   - `label = 1` if `candidate.arc_id == aligned_arc_id`.
   - `label = 0` otherwise.
   - If the true arc was missed by the top-$K$ candidate search, all $K$ candidates for that observation receive `label = 0`.

---

## 3. Empirical Results Across Datasets

Extracted on real data with $K = 5$, search radius $R = 100\text{ m}$:

| Partition | Trips | Observations | Candidate Pairs | Positive Labels (`label=1`) | Uncertain Obs (`label_uncertain=True`) |
|:---|:---|:---|:---|:---|:---|
| **`train`** | 80 | 207,004 | 922,458 | 179,245 (19.4%) | 43,901 (21.21%) |
| **`student_test`** | 10 | 19,160 | 87,226 | 17,211 (19.7%) | 2,746 (14.33%) |

- **Candidate Imbalance**: In both partitions, positive candidates represent **~19.5%** of candidate pairs, perfectly matching Member 2's class-weighting strategy (`class_weight='balanced'`).
- **Uncertain Share**: ~14–21% of observations are flagged uncertain (concentrated around complex intersections and large time gaps like trip `00000090`), leaving over **163,000 clean training observations** and **720,000+ candidate training samples**.

---

## 4. Strict Data Leakage Guarantees

- **Inference Mode**: When `extract_candidate_features(ptrip, include_labels=False)` is called (or when `ptrip.get("route")` is `None`), the pipeline produces identical numerical feature values without accessing ground truth.
- **Evaluator Partition Protection**: Evaluator trips have no `.route` file available, and `load_trip` raises `PermissionError` if route reading is attempted.
- **Learned Transforms**: All normalizations/scalers in downstream training harnesses are fitted strictly in-fold on training folds (`GroupKFold` by `trip_id`).
- **Leakage Test**: Verified by unit test `test_leakage_guard_identical_features_without_route`, ensuring feature columns are bit-for-bit identical regardless of route presence.

---

## 5. Execution Commands

### Run Feature Engineering & Generate Matrices
```bash
python -m feature_engineering.pipeline
```
Outputs:
- `feature_engineering/results/train_features.csv`
- `feature_engineering/results/student_test_features.csv`
- `visualisations/maps/alignment_00000001.html` (interactive Folium alignment map)

### Run Feature Engineering Tests
```bash
python -m pytest feature_engineering/tests -v
```
