import React from 'react';
import { Route, Layers, AlertCircle, ArrowUpRight } from 'lucide-react';

export function OverviewBanner({
  tripCount,
  selectedTripId,
  isOnline,
  isChecking,
  baseUrl,
  onNavigateToStatus,
}) {
  return (
    <div className="space-y-4">
      {/* Slim, Non-Intrusive Offline Banner */}
      {!isOnline && !isChecking && (
        <div className="p-3.5 rounded-xl bg-amber-50 border border-amber-200/90 flex items-center justify-between text-amber-900 text-xs shadow-soft">
          <div className="flex items-center gap-2.5">
            <span className="w-2 h-2 rounded-full bg-amber-500 animate-pulse"></span>
            <span>
              <strong>FastAPI Disconnected:</strong> Running in local UI preview mode. Connect to Laptop 1 backend at{' '}
              <code className="font-mono bg-amber-100 px-1.5 py-0.5 rounded text-amber-950 font-semibold">{baseUrl}</code>
            </span>
          </div>

          {onNavigateToStatus && (
            <button
              onClick={onNavigateToStatus}
              className="font-medium text-amber-800 hover:text-amber-950 underline underline-offset-2 flex items-center gap-1 text-[11px] ml-3 flex-shrink-0"
            >
              Configure IP &rarr;
            </button>
          )}
        </div>
      )}

      {/* Sleek Hero Banner */}
      <div className="card-panel p-6 md:p-7 relative overflow-hidden bg-gradient-to-r from-slate-900 via-slate-850 to-slate-900 text-white shadow-md">
        <div className="relative z-10 flex flex-col md:flex-row md:items-center justify-between gap-6">
          <div className="space-y-2 max-w-2xl">
            <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-full bg-brand-500/20 border border-brand-400/30 text-brand-300 text-xs font-medium">
              <span className="w-1.5 h-1.5 rounded-full bg-brand-400"></span>
              Distributed Map-Matching System
            </div>
            <h1 className="text-2xl md:text-3xl font-extrabold tracking-tight text-white font-sans">
              Path<span className="text-brand-400">Snap</span> Evaluator
            </h1>
            <p className="text-xs md:text-sm text-slate-300 font-normal leading-relaxed">
              Real-time trajectory reconstruction combining candidate road feature scoring and sequential Hidden Markov Model (HMM) Viterbi path decoding.
            </p>
          </div>

          {/* Quick Metrics Pills */}
          <div className="flex items-center gap-3">
            <div className="px-5 py-3 rounded-xl bg-white/10 backdrop-blur-md border border-white/10 min-w-[130px]">
              <span className="text-[10px] uppercase tracking-wider text-slate-400 font-semibold block mb-0.5">
                Evaluation Trips
              </span>
              <div className="text-2xl font-bold font-mono text-white">
                {tripCount > 0 ? tripCount : '0'}
                <span className="text-xs text-slate-400 font-sans font-normal ml-1.5">loaded</span>
              </div>
            </div>

            <div className="px-5 py-3 rounded-xl bg-white/10 backdrop-blur-md border border-white/10 min-w-[140px]">
              <span className="text-[10px] uppercase tracking-wider text-slate-400 font-semibold block mb-0.5">
                Active Selection
              </span>
              <div className="text-lg font-bold font-mono text-brand-300 truncate">
                {selectedTripId || 'None'}
              </div>
            </div>
          </div>
        </div>

        {/* Ambient subtle light gradient */}
        <div className="absolute -right-16 -top-16 w-64 h-64 bg-brand-500/10 rounded-full blur-3xl pointer-events-none"></div>
      </div>
    </div>
  );
}
