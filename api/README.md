# PathSnap FastAPI Backend (Issue #10)

Runs on **Laptop 1**. Serves trip listings, runs the full map-matching pipeline, and returns
server-computed evaluation metrics. Ground-truth reference routes never leave this machine.

## Quick start (repo root)

```bash
# activate venv first
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000/docs` for the interactive API explorer.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `PATHSNAP_TRIPS_DIR` | `handouts/Student folder/evaluator_test` | Folder whose sub-folders are selectable trips |
| `PATHSNAP_MOCK` | `0` | Set to `1` for synthetic responses during UI development; **must be `0` for the demo** |

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | API status, trip count, emission source, mock flag |
| `GET` | `/trips` | List of available trip IDs |
| `POST` | `/predict` | Run map-matching pipeline for a trip; returns arc IDs, map HTML, and server-side metrics |

## Running tests

```bash
python -m pytest api -v
```

All tests run in mock mode against a tiny synthetic trip folder — no dataset required.

## Request logging

Every request is printed to the terminal **and** written to `api/logs/requests.log`:

```
2026-11-18 10:00:01 | 192.168.1.42 POST /predict -> 200 (312 ms)
2026-11-18 10:00:01 | predict trip=00000001 obs=47 matched=47 segments=3 latency=312 ms mock=False
```

Evaluators are expected to inspect these logs on Laptop 1 during the demo.
