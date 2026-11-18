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

## Findings (fill from the EDA output)

- Observations per trip (train / student_test): TODO min–max
- Road arcs per trip: TODO min–max
- Share of 1-second sampling steps: TODO
- Duplicate GPS rows / repeated / backward timestamps: TODO
- Largest time gaps and which trips: TODO (check trips with extreme gaps or near-zero mean speed by hand on the EDA map)
- Weakly / strongly connected components, isolated nodes: TODO
- Two-way arc share (one-way streets): TODO
- Reference route connectivity (`route_connected_share` below 1.0 → which trips, why): TODO

## Outputs

- `supporting_results/eda/trip_summary.csv`, `partition_summary.csv`
- `visualisations/plots/eda_*.png`
- `visualisations/maps/eda_<trip>.html` (Folium: GPS + road network)
