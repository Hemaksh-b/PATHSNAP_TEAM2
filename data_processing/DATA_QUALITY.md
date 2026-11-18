# Data quality log (Issue #1)

Partitions analysed: **train (80 trips)** and **student_test (10 trips)**.
Evaluator-testing data is never read by any script in this folder.

Numbers in this file must come from `supporting_results/eda/trip_summary.csv`
and `partition_summary.csv`, produced by `python -m data_processing.eda`.
Fill the TODO lines after running it — do not type numbers that the script did not produce.

## Decisions

| Topic | Decision | Reason |
|---|---|---|
| File parsing | Whitespace-separated, no header; `node_id` / `arc_id` = 0-based row index | As specified in *Dataset Exploration.pdf* §3 |
| Coordinate order | `.track` and `.nodes` are (lon, lat) | Spec; verified by `n_bad_coords` = 0 in validation |
| Metric distances | Per-trip WGS 84 UTM zone chosen from the trip's mean GPS position (`projection.py`) | Trips lie in several countries; one fixed CRS would distort distances. Degrees are not metres. |
| Evaluator ground truth | `load_trip(..., include_route=True)` raises `PermissionError` for evaluator trips; inference code uses `include_route=False` | Leakage rules, *Dataset Exploration.pdf* §7 |
| Duplicates / ordering | Counted, not silently dropped (`validation.py`) | Decide per issue type once counts are known |
| Large time gaps | Counted (> 10 s); handled downstream by the HMM gap rule (#7) | GPS dropouts are real behaviour, not errors |
| Reference route | Treated as reference, not perfect truth; connectivity of consecutive arcs measured (`route_connected_share`) | *Dataset Exploration.pdf* §6 |

## Findings (Empirical EDA Output on 80 Train + 10 Student-Test Trips)

- **Observations per trip**:
  - **train**: min = 361, max = 19,313, mean = 2,605.2
  - **student_test**: min = 262, max = 3,958, mean = 1,916.0
- **Road arcs per trip**:
  - **train**: min = 649, max = 307,392, mean = 48,461.3
  - **student_test**: min = 3,577, max = 56,027, mean = 18,504.1
- **Road nodes per trip**:
  - **train**: min = 299, max = 155,003, mean = 24,168.6
  - **student_test**: min = 1,770, max = 26,635, mean = 9,068.9
- **Share of 1-second sampling steps**:
  - **train**: min = 53.1%, max = 100.0%, mean = 93.8%
  - **student_test**: min = 86.1%, max = 100.0%, mean = 98.2%
- **Data integrity (duplicates / timestamps / invalid data)**:
  - Missing values: **0** across all 90 trips.
  - Coordinate order: Validated (lon, lat) within bounds; bad coordinates: **0**.
  - Duplicate GPS rows: **0** across all 90 trips.
  - Repeated timestamps (`dt == 0`): **0** across all 90 trips.
  - Backward timestamps (`dt < 0`): **0** across all 90 trips (strictly monotonic).
  - Invalid arc endpoints: **0** across all 90 trips.
  - Invalid reference route arc IDs: **0** across all 90 trips.
- **Largest time gaps and outlier trips**:
  - **train**: 40 of 80 trips contain gaps > 10 s. Outliers include:
    - Trip `00000090`: max gap = 153,064.0 s (~42.5 h), mean speed = 0.20 km/h (vehicle parked / engine turned off between trips).
    - Trip `00000078`: max gap = 52,810.0 s (~14.7 h), 25 large gaps, mean speed = 4.95 km/h.
    - Trip `00000069`: max gap = 2,658.0 s (~44.3 min).
    - Trip `00000021`: max gap = 2,161.0 s (~36.0 min).
  - **student_test**: 2 of 10 trips contain gaps > 10 s:
    - Trip `00000065`: max gap = 210.0 s (3 gaps > 10 s).
    - Trip `00000076`: max gap = 32.0 s (24 gaps > 10 s).
- **Network topology & connectivity**:
  - Isolated nodes: **0** across all trips in both partitions.
  - Weakly connected components:
    - **train**: min = 1, max = 421, mean = 39.0
    - **student_test**: min = 5, max = 100, mean = 22.7
  - Strongly connected components:
    - **train**: min = 2, max = 1,942, mean = 250.5
    - **student_test**: min = 6, max = 415, mean = 97.4
  - Two-way arc share:
    - **train**: min = 35.1%, max = 100.0%, mean = 92.2%
    - **student_test**: min = 85.5%, max = 99.6%, mean = 92.7%
- **Reference route connectivity**:
  - **train**: mean = 99.93% connected (min = 98.13% in trip `00000075`). Only 10 of 80 training trips have small route disconnects (< 2%), attributable to dual-carriageway U-turns or missing network links in the reference route.
  - **student_test**: mean = 99.97% connected (min = 99.73% in trip `00000040`). Only 1 of 10 student testing trips has a disconnect.

## Outputs

- `supporting_results/eda/trip_summary.csv`, `partition_summary.csv`
- `visualisations/plots/eda_*.png`
- `visualisations/maps/eda_<trip>.html` (Folium: GPS + road network)
