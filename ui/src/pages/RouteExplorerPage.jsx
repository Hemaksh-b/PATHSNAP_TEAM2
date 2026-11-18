import React, { useState } from 'react';
import { MapViewer } from '../components/Dashboard/MapViewer';
import { Compass, History, FileText, ChevronRight } from 'lucide-react';

export function RouteExplorerPage({ evaluation, backendStatus }) {
  const {
    selectedTripId,
    selectTrip,
    runReconstruction,
    predictionResult,
    mapHtml,
    mapUrl,
    loadingMap,
    mapError,
    runHistory,
  } = evaluation;

  const [activeView, setActiveView] = useState('map'); // 'map' | 'raw_payload'

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Route Explorer Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-bold text-slate-900 flex items-center gap-2">
            <Compass className="w-5 h-5 text-brand-600" />
            Route Explorer & Deep Trajectory Inspection
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Detailed geographical and sequential analysis of reconstructed HMM/Viterbi trajectories
          </p>
        </div>

        {/* View Toggle */}
        <div className="flex items-center gap-1.5 p-1 rounded-xl bg-slate-200/80 border border-slate-200 text-xs shadow-soft">
          <button
            onClick={() => setActiveView('map')}
            className={`px-3 py-1.5 rounded-lg font-medium transition-all ${
              activeView === 'map'
                ? 'bg-white text-slate-900 shadow-sm'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Folium Map View
          </button>
          <button
            onClick={() => setActiveView('raw_payload')}
            className={`px-3 py-1.5 rounded-lg font-medium transition-all ${
              activeView === 'raw_payload'
                ? 'bg-white text-slate-900 shadow-sm'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Backend JSON Payload
          </button>
        </div>
      </div>

      {/* Main View Area */}
      {activeView === 'map' ? (
        <div className="space-y-6">
          <MapViewer
            mapHtml={mapHtml}
            mapUrl={mapUrl}
            selectedTripId={selectedTripId}
            isLoading={loadingMap}
            error={mapError}
            onReloadMap={() => runReconstruction(selectedTripId)}
          />

          {/* Evaluation Run History Table */}
          <div className="card-panel p-6 space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <History className="w-4 h-4 text-brand-600" />
                <h3 className="text-sm font-bold text-slate-900">Evaluation Session History</h3>
              </div>
              <span className="text-xs text-slate-500 font-mono">
                {runHistory.length} reconstructions executed
              </span>
            </div>

            {runHistory.length === 0 ? (
              <div className="py-8 text-center text-xs text-slate-400 border border-dashed border-slate-200 rounded-xl bg-slate-50">
                No reconstruction runs recorded yet in this session. Run a trip from the Dashboard to record results.
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead>
                    <tr className="border-b border-slate-200 text-slate-400 font-mono uppercase tracking-wider text-[10px]">
                      <th className="py-2.5 px-3">Time</th>
                      <th className="py-2.5 px-3">Trip ID</th>
                      <th className="py-2.5 px-3">Top-1 Accuracy</th>
                      <th className="py-2.5 px-3">Macro F1</th>
                      <th className="py-2.5 px-3">Continuity</th>
                      <th className="py-2.5 px-3">Latency / RTT</th>
                      <th className="py-2.5 px-3 text-right">Action</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {runHistory.map((run, i) => (
                      <tr key={i} className="hover:bg-slate-50 font-mono transition-colors">
                        <td className="py-3 px-3 text-slate-500">{run.timestamp}</td>
                        <td className="py-3 px-3 font-bold text-slate-900">{run.tripId}</td>
                        <td className="py-3 px-3 text-indigo-600 font-semibold">
                          {run.metrics?.accuracy !== null && run.metrics?.accuracy !== undefined
                            ? `${(run.metrics.accuracy * 100).toFixed(2)}%`
                            : 'N/A'}
                        </td>
                        <td className="py-3 px-3 text-cyan-600 font-semibold">
                          {run.metrics?.macroF1 !== null && run.metrics?.macroF1 !== undefined
                            ? `${(run.metrics.macroF1 * 100).toFixed(2)}%`
                            : 'N/A'}
                        </td>
                        <td className="py-3 px-3 text-emerald-600 font-semibold">
                          {run.metrics?.routeContinuity !== null && run.metrics?.routeContinuity !== undefined
                            ? `${(run.metrics.routeContinuity * 100).toFixed(2)}%`
                            : 'N/A'}
                        </td>
                        <td className="py-3 px-3 text-slate-600">{run.roundTripTimeMs} ms</td>
                        <td className="py-3 px-3 text-right">
                          <button
                            onClick={() => {
                              selectTrip(run.tripId);
                              runReconstruction(run.tripId);
                            }}
                            className="text-xs text-brand-600 hover:text-brand-800 font-sans font-medium underline underline-offset-2"
                          >
                            Re-run
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      ) : (
        /* Raw Backend Payload Inspector */
        <div className="card-panel p-6 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
              <FileText className="w-4 h-4 text-brand-600" />
              Latest Prediction Response Payload
            </h3>
            <span className="text-[11px] font-mono text-slate-500">
              Trip: {selectedTripId || 'None'}
            </span>
          </div>

          {predictionResult?.raw ? (
            <pre className="p-4 rounded-xl bg-slate-900 font-mono text-xs text-slate-100 overflow-x-auto max-h-[500px]">
              {JSON.stringify(predictionResult.raw, null, 2)}
            </pre>
          ) : (
            <div className="py-12 text-center text-xs text-slate-400 border border-dashed border-slate-200 rounded-xl bg-slate-50">
              No prediction payload received yet. Run a reconstruction to inspect the raw response from FastAPI.
            </div>
          )}
        </div>
      )}
    </div>
  );
}
