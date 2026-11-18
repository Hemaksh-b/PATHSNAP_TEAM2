
# PATHSNAP Data Processing

## Purpose
Loads and validates GPS trajectory and road-network data for PATHSNAP.

## Expected dataset files

Each trip folder should contain:

- `.track` — longitude, latitude, timestamp
- `.nodes` — longitude, latitude, node ID
- `.arcs` — source node ID, destination node ID
- `.route` — chronological arc IDs (training/student-test only)

## Dataset location

Set the dataset root before running:

```bash
export PATHSNAP_DATA_ROOT="/path/to/your/dataset"
```

Do not commit raw dataset files.

## Features

- Loads track, nodes, arcs and permitted route labels.
- Validates coordinates, timestamps and node/arc IDs.
- Projects GPS coordinates into local UTM metres using PyProj.
- Calculates road arc lengths.
- Protects evaluator route labels from being loaded.
- Generates EDA reports and plots.

## Run tests

```bash
python -m pytest tests/test_data_processing.py -v
```

## Run EDA

```bash
python -m data_processing.run_eda
```

Outputs are saved under `data_processing/eda_outputs/`.
