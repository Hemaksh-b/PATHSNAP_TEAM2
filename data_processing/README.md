
# PATHSNAP Data Processing (Issue #1)

## Purpose
Loads, validates, projects, and explores GPS trajectory and road-network data for PathSnap.

## Raw Dataset Location and Layout
The default dataset is located in `handouts/Student folder` (configured in `data_processing/config.py`):

```
handouts/Student folder/
├── train/                  # 80 training trips (00000001 ... 00000099)
│   └── <trip_id>/
│       ├── <trip_id>.track # GPS observations: lon, lat, timestamp (s)
│       ├── <trip_id>.nodes # Road network nodes: lon, lat (row index = node_id)
│       ├── <trip_id>.arcs  # Road network arcs: from_node_id, to_node_id (row index = arc_id)
│       └── <trip_id>.route # Reference route: chronological sequence of arc_ids
├── student_test/           # 10 student-testing trips (00000000, 00000007, ...)
└── evaluator_test/         # Reserved partition for final evaluation
```

Files are whitespace-separated with no headers.

Optionally override the root directory via environment variable:
```bash
export PATHSNAP_DATA_ROOT="handouts/Student folder"
```

> **Note**: Raw dataset files are git-ignored or managed through the handouts directory and should not be committed if they exceed repository size limits.

## Deliverable Interface (for Members 2, 3 and 4)

Module: `data_processing.pipeline` or `data_processing.loader`

### 1. `load_trip(trip_dir, partition=None, allow_route=True)`
Returns a `Trip` object (or dictionary) containing:
- `track`: DataFrame with columns `[obs_idx, lon, lat, t, x, y]`
- `nodes`: DataFrame with columns `[node_id, lon, lat, x, y]`
- `arcs`: DataFrame with columns `[arc_id, u, v, length_m]`
- `route`: list of reference `arc_id`s (for `train` and `student_test` only)
- `crs`: EPSG code string of the local WGS 84 UTM projection (e.g. `EPSG:32632`)

### 2. `list_trips(partition)`
Returns a sorted list of trip directory paths for `"train"` (80 trips) or `"student_test"` (10 trips).

### 3. `load_evaluator_trip(trip_dir)` / `load_trip(..., include_route=False)`
Strictly protected inference path for evaluator testing. Never accesses `.route` files and raises `PermissionError` if `.route` reading is attempted.

## Coordinate Projection
- Trajectories span multiple UTM zones across different continents.
- A local WGS 84 UTM CRS is chosen automatically per trip from its mean GPS coordinate (`data_processing/projection.py`), keeping spatial distortion under 0.1%.
- Longitude and latitude (degrees) are converted to metric planar coordinates `(x, y)` in metres using PyProj, with an exact inverse projection available.

## Running Tests
Run unit tests for parsing, index validity, evaluator protection, and projection round-trip:

```bash
python -m pytest data_processing/tests/test_issue1_loading.py tests/test_data_processing.py -v
```

## Running Exploratory Data Analysis (EDA)
Generate all summary CSV tables, distribution plots, and interactive Folium maps with one command:

```bash
python -m data_processing.eda
```

Outputs:
- Tables: `supporting_results/eda/trip_summary.csv`, `supporting_results/eda/partition_summary.csv`
- Distribution plots: `visualisations/plots/eda_*.png`
- Folium interactive map: `visualisations/maps/eda_00000001.html`
- Data quality log: `data_processing/DATA_QUALITY.md`

