# PathSnap – Evaluator Dashboard (Phase 1)

A modern, high-performance web interface designed for university machine learning hackathon evaluators to inspect, grade, and interact with the **PathSnap Distributed Map-Matching & Route Reconstruction** pipeline.

The application communicates with a **FastAPI backend** running on a separate laptop (Laptop 1) over a local network (Wi-Fi or Ethernet), rendering sequential HMM/Viterbi route matches and interactive Folium maps.

---

## Features

- **Evaluator Dashboard**: Clean dark dashboard displaying active trip selection, available evaluation trips count, backend connectivity, and execution triggers.
- **Dynamic Trip Selection**: Fetches official evaluation trip IDs directly from the backend via `GET /evaluation/trips`. Disables triggers when no trip is selected or inference is in progress.
- **Model Evaluation Metrics**: Displays actual observation-level Top-1 Accuracy, Macro F1-Score, Route Continuity, Log Loss (if computed), and Inference Time returned in the backend payload.
- **Interactive Folium Map**: Responsive container with full pan/zoom, fullscreen toggle, and dual-layer visualization (Raw GPS observations in Amber vs. Predicted Viterbi road segments in Blue/Cyan).
- **Route Explorer**: Deep-dive trajectory inspection view with a session run history table and raw JSON backend payload inspector.
- **Distributed System Diagnostics**: Real-time health monitoring, network latency tracking, live Laptop 1 IP configuration, and interactive endpoint contract probes.
- **Zero Fabrication**: No hardcoded predictions, no client-side metric calculations, and no exposure of ground-truth labels.

---

## Technology Stack

- **Framework**: React 18 with Vite 6
- **Styling**: Tailwind CSS (custom dark theme with deep navy/charcoal `#070a11` / `#0c1220` and subtle blue accents)
- **Icons**: Lucide React
- **Network / Communication**: Native Fetch API with timeout controllers and connection retry logic
- **Map Engine**: Embedded Folium HTML via responsive iframe (`srcDoc` & `src` fallback)

---

## Folder Structure

```
ui/
├── index.html
├── package.json
├── vite.config.js
├── tailwind.config.js
├── postcss.config.js
├── .env.example
├── .env
├── src/
│   ├── main.jsx
│   ├── App.jsx
│   ├── styles/
│   │   └── index.css                     # Tailwind layers, scrollbars, and card styles
│   ├── services/
│   │   └── api.js                        # Centralized API service with timeout & normalization
│   ├── hooks/
│   │   ├── useBackendStatus.js           # Real-time health check & latency hook
│   │   └── useEvaluation.js              # Trips loader, prediction state & run history
│   ├── components/
│   │   ├── Common/
│   │   │   └── ConnectionBadge.jsx       # Pulsing status badge & latency indicator
│   │   ├── Navigation/
│   │   │   ├── Sidebar.jsx               # Brand header, navigation links & architecture pill
│   │   │   └── Header.jsx                # Top header bar with page context & status
│   │   └── Dashboard/
│   │       ├── OverviewBanner.jsx        # Project overview, metric pills & dev mode alerts
│   │       ├── TripSelector.jsx          # Trip selection cards, search & reconstruction CTA
│   │       ├── MetricsCard.jsx           # Top-1 Accuracy, Macro F1, Continuity & Inference Time
│   │       └── MapViewer.jsx             # Responsive Folium map viewer with fullscreen & legend
│   └── pages/
│       ├── DashboardPage.jsx             # Evaluator command center
│       ├── RouteExplorerPage.jsx         # Geographical inspection & session history
│       └── SystemStatusPage.jsx          # Distributed diagnostics & live backend URL editor
```

---

## Installation & Setup

### Prerequisites
- Node.js `v18+` (tested on Node `v24.19.0`)
- npm `v9+`

### 1. Install Dependencies
```bash
cd PATHSNAP_TEAM2/ui
npm install
```

> **Note for Windows Users:** The `package.json` includes `@rollup/wasm-node` and an automatic postinstall script ensuring compatibility on Windows environments where native binary DLLs might be restricted by AppLocker / Application Control policies.

### 2. Configure Backend Host
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Edit `.env` to point to Laptop 1:
```env
# Laptop 1 FastAPI address (e.g., local network IP)
VITE_API_BASE_URL=http://localhost:8000
```

*(You can also adjust the backend IP dynamically at runtime in the **System Status** page without restarting the dev server!)*

### 3. Run Development Server
```bash
npm run dev
```
The application will be accessible at:
- Local: `http://localhost:5173`
- Network: `http://<your-ip>:5173`

### 4. Build for Production
```bash
npm run build
```
Generates an optimized bundle in `dist/`.

---

## Two-Laptop Distributed Deployment Workflow

For the university hackathon evaluation:

1. **Laptop 1 (FastAPI Backend)**:
   ```bash
   cd PATHSNAP_TEAM2
   uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
   ```
   Find Laptop 1's local IP address (`ipconfig` on Windows or `ifconfig` on macOS/Linux), e.g. `192.168.1.50`.

2. **Laptop 2 (Evaluator Dashboard UI)**:
   ```bash
   cd PATHSNAP_TEAM2/ui
   npm run dev
   ```
   Open `http://localhost:5173` in a browser. Navigate to **System Status**, enter `http://192.168.1.50:8000`, and click **Update & Probe**.
   Once the indicator turns green (**Backend Online**), the evaluation trips will populate automatically on the **Dashboard**.

---

## Proposed API Contract

The API integration layer (`src/services/api.js`) communicates via:

| Endpoint | Method | Description | Request / Response Structure |
| :--- | :--- | :--- | :--- |
| `/health` | `GET` | Health check & model readiness probe | `{ "status": "ok" }` |
| `/evaluation/trips` | `GET` | List available evaluation trip IDs | `["trip_01", "trip_02", ...]` or `{ "trips": [...] }` |
| `/predict` | `POST` | Execute map-matching & return metrics | Body: `{ "trip_id": "trip_01" }`<br>Returns: Top-1 Accuracy, Macro F1, Route Continuity, Inference Time, and optional map HTML |
| `/maps/{trip_id}` | `GET` | Deliver standalone Folium HTML map | HTML string or `{ "map_html": "..." }` |

The service layer is intentionally flexible and auto-normalizes variations in backend dictionary keys without requiring UI refactoring.
