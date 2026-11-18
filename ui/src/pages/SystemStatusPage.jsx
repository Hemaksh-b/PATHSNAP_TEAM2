import React, { useState } from 'react';
import { Activity, Server, Network, Laptop, CheckCircle2, XCircle, RefreshCw, Send } from 'lucide-react';
import { checkBackendHealth, fetchEvaluationTrips, getMapUrl } from '../services/api';

export function SystemStatusPage({ backendStatus, evaluation }) {
  const {
    status,
    isOnline,
    isChecking,
    latencyMs,
    lastCheckTime,
    error,
    baseUrl,
    checkNow,
    updateUrl,
    resetUrl,
  } = backendStatus;

  const [inputUrl, setInputUrl] = useState(baseUrl);
  const [saveMessage, setSaveMessage] = useState(null);

  const [endpointTesting, setEndpointTesting] = useState({});
  const [endpointResults, setEndpointResults] = useState({});

  const handleSaveUrl = (e) => {
    e.preventDefault();
    if (!inputUrl.trim()) return;
    updateUrl(inputUrl);
    setSaveMessage('Backend URL updated. Re-checking health...');
    setTimeout(() => setSaveMessage(null), 4000);
  };

  const handleResetUrl = () => {
    resetUrl();
    setInputUrl(backendStatus.baseUrl);
    setSaveMessage('Reset to environment default.');
    setTimeout(() => setSaveMessage(null), 3000);
  };

  const testEndpoint = async (endpointKey) => {
    setEndpointTesting((prev) => ({ ...prev, [endpointKey]: true }));
    const startTime = performance.now();

    try {
      if (endpointKey === 'health') {
        const res = await checkBackendHealth();
        setEndpointResults((prev) => ({
          ...prev,
          health: {
            success: res.success,
            status: res.status,
            latency: res.latencyMs,
            body: res.data || res.error,
          },
        }));
      } else if (endpointKey === 'trips') {
        const res = await fetchEvaluationTrips();
        const rtt = Math.round(performance.now() - startTime);
        setEndpointResults((prev) => ({
          ...prev,
          trips: {
            success: res.success,
            status: res.status,
            latency: rtt,
            body: res.trips ? `Loaded ${res.trips.length} trips` : res.error,
          },
        }));
      } else if (endpointKey === 'maps') {
        const testTripId = evaluation.selectedTripId || 'trip_1';
        const url = getMapUrl(testTripId);
        const res = await fetch(url);
        const rtt = Math.round(performance.now() - startTime);
        setEndpointResults((prev) => ({
          ...prev,
          maps: {
            success: res.ok,
            status: res.status,
            latency: rtt,
            body: res.ok ? `Folium HTML delivered (Status ${res.status})` : `Failed: HTTP ${res.status}`,
          },
        }));
      }
    } catch (err) {
      setEndpointResults((prev) => ({
        ...prev,
        [endpointKey]: {
          success: false,
          status: 0,
          latency: null,
          body: err.message,
        },
      }));
    } finally {
      setEndpointTesting((prev) => ({ ...prev, [endpointKey]: false }));
    }
  };

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Page Title */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-bold text-slate-900 flex items-center gap-2">
            <Activity className="w-5 h-5 text-brand-600" />
            System Status & Two-Laptop Network Diagnostics
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Real-time health monitor and network configuration for the Distributed Deployment
          </p>
        </div>

        <button
          onClick={checkNow}
          disabled={isChecking}
          className="px-4 py-2 rounded-xl bg-white hover:bg-slate-50 border border-slate-200 text-xs font-semibold text-slate-700 hover:text-slate-900 flex items-center gap-2 self-start sm:self-auto transition-colors disabled:opacity-50 shadow-soft"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${isChecking ? 'animate-spin' : ''}`} />
          <span>Refresh Diagnostics</span>
        </button>
      </div>

      {/* Main Status Grid */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {/* Card 1: API Connection Status */}
        <div className="card-panel p-5 space-y-1.5">
          <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400 block">
            API Connection
          </span>
          <div className="flex items-center gap-2">
            <span
              className={`w-3 h-3 rounded-full ${
                isOnline ? 'bg-emerald-500 shadow-sm shadow-emerald-500/50' : isChecking ? 'bg-amber-400' : 'bg-rose-500'
              }`}
            ></span>
            <span className="text-base font-bold text-slate-900">
              {isOnline ? 'FastAPI Online' : isChecking ? 'Checking...' : 'Disconnected'}
            </span>
          </div>
          <p className="text-[11px] text-slate-500">
            {isOnline ? 'FastAPI responding normally' : 'Cannot reach backend on local network'}
          </p>
        </div>

        {/* Card 2: Round-Trip Latency */}
        <div className="card-panel p-5 space-y-1.5">
          <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400 block">
            Network Latency
          </span>
          <div className="text-base font-bold font-mono text-brand-600">
            {latencyMs !== null ? `${latencyMs} ms` : '—'}
          </div>
          <p className="text-[11px] text-slate-500">
            Round-trip ping time to Laptop 1
          </p>
        </div>

        {/* Card 3: Last Health Check */}
        <div className="card-panel p-5 space-y-1.5">
          <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400 block">
            Last Health Probe
          </span>
          <div className="text-base font-bold font-mono text-slate-900">
            {lastCheckTime ? lastCheckTime.toLocaleTimeString() : 'Never'}
          </div>
          <p className="text-[11px] text-slate-500 font-mono text-[10px]">
            {lastCheckTime ? lastCheckTime.toLocaleDateString() : 'Awaiting check'}
          </p>
        </div>

        {/* Card 4: Current Processing State */}
        <div className="card-panel p-5 space-y-1.5">
          <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400 block">
            Processing State
          </span>
          <div className="text-base font-bold text-slate-900 flex items-center gap-2">
            {evaluation.isReconstructing ? (
              <>
                <RefreshCw className="w-4 h-4 text-brand-600 animate-spin" />
                <span className="text-brand-600 text-sm">Reconstructing</span>
              </>
            ) : (
              <span className="text-slate-700 text-sm">Idle / Ready</span>
            )}
          </div>
          <p className="text-[11px] text-slate-500 truncate">
            {evaluation.isReconstructing
              ? `Trip ${evaluation.selectedTripId}`
              : 'Standby for evaluator command'}
          </p>
        </div>
      </div>

      {/* Backend Target IP / Port Configuration */}
      <div className="card-panel p-6 space-y-4">
        <div>
          <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
            <Server className="w-4 h-4 text-brand-600" />
            FastAPI Backend IP Configuration (Laptop 1)
          </h3>
          <p className="text-xs text-slate-500 mt-1">
            When operating across two laptops over local Wi-Fi or Ethernet, enter the IP address of Laptop 1 hosting FastAPI.
          </p>
        </div>

        <form onSubmit={handleSaveUrl} className="flex flex-col sm:flex-row gap-3">
          <input
            type="text"
            value={inputUrl}
            onChange={(e) => setInputUrl(e.target.value)}
            placeholder="e.g. http://192.168.1.50:8000"
            className="flex-1 px-4 py-2 bg-slate-50 border border-slate-200 rounded-xl text-xs font-mono text-slate-900 placeholder-slate-400 focus:outline-none focus:border-brand-500 transition-colors"
          />

          <button
            type="submit"
            className="px-5 py-2 rounded-xl bg-brand-600 hover:bg-brand-700 text-white text-xs font-semibold shadow-soft transition-all cursor-pointer"
          >
            Update & Probe
          </button>

          <button
            type="button"
            onClick={handleResetUrl}
            className="px-4 py-2 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-medium transition-colors"
          >
            Reset Default
          </button>
        </form>

        {saveMessage && (
          <div className="text-xs text-emerald-600 font-medium">
            {saveMessage}
          </div>
        )}

        {error && (
          <div className="p-3.5 rounded-xl bg-rose-50 border border-rose-200 text-xs text-rose-800 font-mono">
            <strong>Connection Error:</strong> {error}
          </div>
        )}
      </div>

      {/* API Endpoint Contracts & Interactive Probe */}
      <div className="card-panel p-6 space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
              <Network className="w-4 h-4 text-brand-600" />
              Proposed API Endpoints Contract
            </h3>
            <p className="text-xs text-slate-500 mt-1">
              Verify communication with each proposed FastAPI route
            </p>
          </div>
          <span className="text-[11px] text-slate-500 font-mono">
            Host: {baseUrl}
          </span>
        </div>

        <div className="space-y-3">
          {/* Endpoint 1: GET /health */}
          <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 flex flex-col md:flex-row md:items-center justify-between gap-3">
            <div className="space-y-0.5">
              <div className="flex items-center gap-2 font-mono text-xs">
                <span className="px-2 py-0.5 rounded bg-emerald-100 text-emerald-800 font-bold text-[10px]">
                  GET
                </span>
                <span className="font-semibold text-slate-900">/health</span>
              </div>
              <p className="text-xs text-slate-500">
                Health check probe returning API status and model readiness
              </p>
            </div>

            <div className="flex items-center gap-3">
              {endpointResults.health && (
                <span
                  className={`text-xs font-mono px-2 py-1 rounded border ${
                    endpointResults.health.success
                      ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                      : 'bg-rose-50 text-rose-700 border-rose-200'
                  }`}
                >
                  HTTP {endpointResults.health.status} ({endpointResults.health.latency}ms)
                </span>
              )}
              <button
                onClick={() => testEndpoint('health')}
                disabled={endpointTesting.health}
                className="px-3 py-1.5 rounded-lg bg-white hover:bg-slate-100 border border-slate-200 text-xs text-slate-700 transition-colors disabled:opacity-50 flex items-center gap-1.5 shadow-soft"
              >
                <Send className="w-3 h-3 text-brand-600" />
                <span>{endpointTesting.health ? 'Testing...' : 'Test'}</span>
              </button>
            </div>
          </div>

          {/* Endpoint 2: GET /evaluation/trips */}
          <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 flex flex-col md:flex-row md:items-center justify-between gap-3">
            <div className="space-y-0.5">
              <div className="flex items-center gap-2 font-mono text-xs">
                <span className="px-2 py-0.5 rounded bg-emerald-100 text-emerald-800 font-bold text-[10px]">
                  GET
                </span>
                <span className="font-semibold text-slate-900">/evaluation/trips</span>
              </div>
              <p className="text-xs text-slate-500">
                Returns the list of evaluation trip IDs reserved for grading
              </p>
            </div>

            <div className="flex items-center gap-3">
              {endpointResults.trips && (
                <span
                  className={`text-xs font-mono px-2 py-1 rounded border ${
                    endpointResults.trips.success
                      ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                      : 'bg-rose-50 text-rose-700 border-rose-200'
                  }`}
                >
                  {endpointResults.trips.body}
                </span>
              )}
              <button
                onClick={() => testEndpoint('trips')}
                disabled={endpointTesting.trips}
                className="px-3 py-1.5 rounded-lg bg-white hover:bg-slate-100 border border-slate-200 text-xs text-slate-700 transition-colors disabled:opacity-50 flex items-center gap-1.5 shadow-soft"
              >
                <Send className="w-3 h-3 text-brand-600" />
                <span>{endpointTesting.trips ? 'Testing...' : 'Test'}</span>
              </button>
            </div>
          </div>

          {/* Endpoint 3: POST /predict */}
          <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 flex flex-col md:flex-row md:items-center justify-between gap-3">
            <div className="space-y-0.5">
              <div className="flex items-center gap-2 font-mono text-xs">
                <span className="px-2 py-0.5 rounded bg-brand-100 text-brand-800 font-bold text-[10px]">
                  POST
                </span>
                <span className="font-semibold text-slate-900">/predict</span>
                <span className="text-[10px] text-slate-500">Payload: {'{ "trip_id": string }'}</span>
              </div>
              <p className="text-xs text-slate-500">
                Executes ML scoring + Viterbi decoding; returns accuracy, F1, continuity & latency
              </p>
            </div>

            <span className="text-xs text-slate-500 font-mono">
              Triggered via "Run Reconstruction"
            </span>
          </div>

          {/* Endpoint 4: GET /maps/{trip_id} */}
          <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 flex flex-col md:flex-row md:items-center justify-between gap-3">
            <div className="space-y-0.5">
              <div className="flex items-center gap-2 font-mono text-xs">
                <span className="px-2 py-0.5 rounded bg-emerald-100 text-emerald-800 font-bold text-[10px]">
                  GET
                </span>
                <span className="font-semibold text-slate-900">/maps/{'{trip_id}'}</span>
              </div>
              <p className="text-xs text-slate-500">
                Delivers self-contained Folium HTML map with GPS and Predicted layers for iframe embedding
              </p>
            </div>

            <div className="flex items-center gap-3">
              {endpointResults.maps && (
                <span
                  className={`text-xs font-mono px-2 py-1 rounded border ${
                    endpointResults.maps.success
                      ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                      : 'bg-rose-50 text-rose-700 border-rose-200'
                  }`}
                >
                  {endpointResults.maps.body}
                </span>
              )}
              <button
                onClick={() => testEndpoint('maps')}
                disabled={endpointTesting.maps}
                className="px-3 py-1.5 rounded-lg bg-white hover:bg-slate-100 border border-slate-200 text-xs text-slate-700 transition-colors disabled:opacity-50 flex items-center gap-1.5 shadow-soft"
              >
                <Send className="w-3 h-3 text-brand-600" />
                <span>{endpointTesting.maps ? 'Testing...' : 'Test'}</span>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
