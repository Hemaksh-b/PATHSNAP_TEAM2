/**
 * PathSnap API Service
 * 
 * Handles all network communication between the Evaluator UI and the FastAPI Backend.
 * Supports configurable backend URL (via .env or evaluator runtime configuration in System Status).
 * 
 * Target Endpoints:
 * - GET  /health           -> Check backend online status
 * - GET  /evaluation/trips -> Fetch available evaluation trip identifiers
 * - POST /predict          -> Request map-matching & route reconstruction for selected trip
 * - GET  /maps/{trip_id}   -> Fetch Folium HTML map visualization
 */

const STORAGE_KEY_BASE_URL = 'pathsnap_backend_url';

/**
 * Returns current API base URL, preferring runtime override if set, else env var or default.
 */
export function getApiBaseUrl() {
  const customUrl = localStorage.getItem(STORAGE_KEY_BASE_URL);
  if (customUrl && customUrl.trim()) {
    return customUrl.trim().replace(/\/+$/, '');
  }
  const envUrl = import.meta.env.VITE_API_BASE_URL;
  if (envUrl && envUrl.trim()) {
    return envUrl.trim().replace(/\/+$/, '');
  }
  return 'http://localhost:8000';
}

/**
 * Update the backend URL at runtime (e.g. when connecting across two laptops via local Wi-Fi/LAN).
 */
export function setApiBaseUrl(url) {
  if (!url || !url.trim()) {
    localStorage.removeItem(STORAGE_KEY_BASE_URL);
  } else {
    localStorage.setItem(STORAGE_KEY_BASE_URL, url.trim().replace(/\/+$/, ''));
  }
}

/**
 * Reset to default environment URL
 */
export function resetApiBaseUrl() {
  localStorage.removeItem(STORAGE_KEY_BASE_URL);
}

/**
 * Base fetch wrapper with timeout and standardized error handling.
 */
async function apiRequest(endpoint, options = {}, timeoutMs = 12000) {
  const baseUrl = getApiBaseUrl();
  const url = `${baseUrl}${endpoint.startsWith('/') ? endpoint : `/${endpoint}`}`;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  const defaultHeaders = {
    'Accept': 'application/json',
  };

  if (options.body && typeof options.body === 'string') {
    defaultHeaders['Content-Type'] = 'application/json';
  }

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
      headers: {
        ...defaultHeaders,
        ...options.headers,
      },
    });

    clearTimeout(timeoutId);

    // If response is not JSON (e.g. 500 HTML or raw text)
    const contentType = response.headers.get('content-type') || '';
    let parsedData = null;

    if (contentType.includes('application/json')) {
      parsedData = await response.json();
    } else {
      parsedData = await response.text();
    }

    if (!response.ok) {
      const errorMessage = typeof parsedData === 'object' && parsedData?.detail
        ? (typeof parsedData.detail === 'string' ? parsedData.detail : JSON.stringify(parsedData.detail))
        : (typeof parsedData === 'string' && parsedData.length < 200 ? parsedData : `HTTP Error ${response.status} (${response.statusText})`);

      return {
        success: false,
        status: response.status,
        error: errorMessage,
        raw: parsedData,
      };
    }

    return {
      success: true,
      status: response.status,
      data: parsedData,
    };
  } catch (err) {
    clearTimeout(timeoutId);

    let clientError = 'Network connection failed';
    if (err.name === 'AbortError') {
      clientError = `Request timed out after ${timeoutMs / 1000}s. Check if FastAPI backend on ${baseUrl} is reachable.`;
    } else if (err.message) {
      clientError = `Unable to connect to backend at ${baseUrl}: ${err.message}. Ensure the FastAPI server is running on Laptop 1 with --host 0.0.0.0.`;
    }

    return {
      success: false,
      status: 0,
      error: clientError,
      isNetworkError: true,
    };
  }
}

/**
 * Health check endpoint
 * GET /health
 */
export async function checkBackendHealth() {
  const startTime = performance.now();
  const result = await apiRequest('/health', { method: 'GET' }, 5000);
  const latency = Math.round(performance.now() - startTime);

  return {
    ...result,
    latencyMs: latency,
    baseUrl: getApiBaseUrl(),
  };
}

/**
 * Fetch available evaluation trip identifiers
 * GET /evaluation/trips
 * 
 * Normalizes different possible response shapes:
 * - ["trip_1", "trip_2", ...]
 * - { "trips": ["trip_1", ...] }
 * - { "trip_ids": ["trip_1", ...] }
 * - { "data": [...] }
 */
export async function fetchEvaluationTrips() {
  const res = await apiRequest('/evaluation/trips', { method: 'GET' }, 8000);

  if (!res.success) {
    return res;
  }

  let tripsArray = [];
  const data = res.data;

  if (Array.isArray(data)) {
    tripsArray = data;
  } else if (data && typeof data === 'object') {
    if (Array.isArray(data.trips)) {
      tripsArray = data.trips;
    } else if (Array.isArray(data.trip_ids)) {
      tripsArray = data.trip_ids;
    } else if (Array.isArray(data.evaluation_trips)) {
      tripsArray = data.evaluation_trips;
    } else if (Array.isArray(data.data)) {
      tripsArray = data.data;
    } else {
      // Object keys might be trip ids if it's a map
      tripsArray = Object.keys(data);
    }
  }

  // Normalize items to { id, label, raw }
  const normalizedTrips = tripsArray.map((item, index) => {
    if (typeof item === 'string' || typeof item === 'number') {
      return {
        id: String(item),
        label: `Trip ${String(item)}`,
        raw: item,
      };
    }
    if (item && typeof item === 'object') {
      const id = item.trip_id || item.id || item.name || `trip_${index + 1}`;
      return {
        id: String(id),
        label: item.title || item.label || `Trip ${id}`,
        pointsCount: item.points_count || item.num_observations || item.length,
        duration: item.duration,
        raw: item,
      };
    }
    return {
      id: `unknown_${index}`,
      label: `Trip ${index + 1}`,
      raw: item,
    };
  });

  return {
    success: true,
    trips: normalizedTrips,
    count: normalizedTrips.length,
    raw: res.data,
  };
}

/**
 * Run route reconstruction & prediction for selected trip
 * POST /predict
 * 
 * Body: { trip_id: string }
 */
export async function runRouteReconstruction(tripId) {
  if (!tripId) {
    return { success: false, error: 'No trip ID provided' };
  }

  const payload = {
    trip_id: tripId,
  };

  const startTime = performance.now();
  const res = await apiRequest('/predict', {
    method: 'POST',
    body: JSON.stringify(payload),
  }, 30000); // Allow up to 30s for ML inference and Viterbi decoding
  const roundTripTime = Math.round(performance.now() - startTime);

  if (!res.success) {
    return res;
  }

  const data = res.data || {};

  // Extract metrics strictly from actual backend response
  // Normalizes possible metric key names without inventing values
  const metrics = extractMetrics(data, roundTripTime);

  return {
    success: true,
    tripId: tripId,
    metrics: metrics,
    mapData: data.map_html || data.map_url || data.map || null,
    predictedRoute: data.predicted_route || data.route || data.path || null,
    observations: data.observations || data.gps_points || null,
    roundTripTimeMs: roundTripTime,
    raw: data,
  };
}

/**
 * Helper to safely extract metrics from backend prediction payload.
 * Strictly checks existence; returns null if metric was not provided by backend.
 */
function extractMetrics(data, fallbackInferenceMs) {
  const getVal = (...keys) => {
    for (const key of keys) {
      if (data && data[key] !== undefined && data[key] !== null) {
        const val = Number(data[key]);
        if (!isNaN(val)) return val;
      }
      if (data?.metrics && data.metrics[key] !== undefined && data.metrics[key] !== null) {
        const val = Number(data.metrics[key]);
        if (!isNaN(val)) return val;
      }
    }
    return null;
  };

  // Top-1 Observation Accuracy
  const accuracy = getVal('top1_accuracy', 'observation_accuracy', 'accuracy', 'top_1_accuracy');

  // Macro F1-score
  const macroF1 = getVal('macro_f1', 'f1_score', 'macro_f1_score', 'f1');

  // Log Loss (optional, only if backend computed it)
  const logLoss = getVal('log_loss', 'logloss', 'loss');

  // Route Continuity
  const routeContinuity = getVal('route_continuity', 'continuity', 'continuity_ratio', 'continuity_score');

  // Inference time
  let inferenceTime = getVal('inference_time_ms', 'inference_time', 'execution_time_ms', 'latency_ms');
  let inferenceTimeUnit = 'ms';

  if (inferenceTime === null) {
    const inferenceTimeSec = getVal('inference_time_s', 'inference_seconds', 'duration_s');
    if (inferenceTimeSec !== null) {
      inferenceTime = inferenceTimeSec;
      inferenceTimeUnit = 's';
    } else {
      inferenceTime = fallbackInferenceMs;
      inferenceTimeUnit = 'ms (client RTT)';
    }
  }

  return {
    accuracy,
    macroF1,
    logLoss,
    routeContinuity,
    inferenceTime,
    inferenceTimeUnit,
  };
}

/**
 * Returns full URL to Folium map endpoint
 * GET /maps/{trip_id}
 */
export function getMapUrl(tripId) {
  if (!tripId) return '';
  const baseUrl = getApiBaseUrl();
  return `${baseUrl}/maps/${encodeURIComponent(tripId)}`;
}

/**
 * Fetches Folium HTML map content from GET /maps/{trip_id}
 */
export async function fetchMapHtml(tripId) {
  if (!tripId) {
    return { success: false, error: 'Trip ID required' };
  }

  const baseUrl = getApiBaseUrl();
  const url = `${baseUrl}/maps/${encodeURIComponent(tripId)}`;

  try {
    const response = await fetch(url, {
      headers: {
        'Accept': 'text/html, application/xhtml+xml, application/json, */*',
      },
    });

    if (!response.ok) {
      return {
        success: false,
        status: response.status,
        error: `Failed to load map: HTTP ${response.status} (${response.statusText})`,
      };
    }

    const contentType = response.headers.get('content-type') || '';
    if (contentType.includes('application/json')) {
      const json = await response.json();
      return {
        success: true,
        html: json.map_html || json.html || (typeof json === 'string' ? json : null),
        raw: json,
      };
    }

    const html = await response.text();
    return {
      success: true,
      html: html,
    };
  } catch (err) {
    return {
      success: false,
      error: `Network error loading map from ${url}: ${err.message}`,
    };
  }
}
