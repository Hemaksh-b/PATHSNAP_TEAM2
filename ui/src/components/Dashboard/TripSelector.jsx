import React, { useState } from 'react';
import { Play, RefreshCw, Navigation, Search, CheckCircle2, Compass, Radio } from 'lucide-react';

export function TripSelector({
  trips = [],
  selectedTripId,
  onSelectTrip,
  onRunReconstruction,
  isReconstructing,
  reconstructionProgress,
  loadingTrips,
  onRefreshTrips,
  isOnline,
  tripsError,
}) {
  const [searchTerm, setSearchTerm] = useState('');

  const filteredTrips = trips.filter((trip) =>
    trip.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
    trip.label.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const canRun = Boolean(selectedTripId) && !isReconstructing && isOnline;

  return (
    <div className="card-panel p-6 space-y-4">
      {/* Header and Action Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-sm font-bold text-slate-900 flex items-center gap-2">
            <Compass className="w-4 h-4 text-brand-600" />
            Evaluation Trip Selection
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Select an evaluation trip served by FastAPI to run map-matching & trajectory reconstruction
          </p>
        </div>

        <div className="flex items-center gap-2.5">
          {/* Reload trips button */}
          <button
            onClick={onRefreshTrips}
            disabled={loadingTrips || isReconstructing}
            title="Reload evaluation trips from backend"
            className="px-3 py-2 rounded-xl bg-slate-100 hover:bg-slate-200 text-xs text-slate-700 font-medium flex items-center gap-1.5 transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loadingTrips ? 'animate-spin' : ''}`} />
            <span>Reload</span>
          </button>

          {/* Run Reconstruction CTA Button */}
          <button
            onClick={() => onRunReconstruction(selectedTripId)}
            disabled={!canRun}
            className={`px-5 py-2 rounded-xl text-xs font-semibold flex items-center gap-2 transition-all ${
              canRun
                ? 'bg-brand-600 hover:bg-brand-700 text-white shadow-md shadow-brand-600/30 cursor-pointer active:scale-95'
                : 'bg-slate-150 text-slate-400 cursor-not-allowed'
            }`}
          >
            {isReconstructing ? (
              <>
                <RefreshCw className="w-3.5 h-3.5 animate-spin text-white" />
                <span>Reconstructing...</span>
              </>
            ) : (
              <>
                <Play className="w-3.5 h-3.5 fill-current text-white" />
                <span>Run Reconstruction</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Progress banner during reconstruction */}
      {isReconstructing && (
        <div className="p-3.5 rounded-xl bg-brand-50 border border-brand-200 flex items-center gap-3 text-brand-900 text-xs animate-pulse">
          <div className="w-2.5 h-2.5 rounded-full bg-brand-600 animate-ping"></div>
          <span className="font-semibold">
            {reconstructionProgress || 'Executing distributed map-matching and Viterbi sequence decoding...'}
          </span>
        </div>
      )}

      {/* Search Bar if trips exist */}
      {trips.length > 5 && (
        <div className="relative">
          <Search className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Filter evaluation trips..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-8 pr-4 py-1.5 bg-slate-50 border border-slate-200 rounded-lg text-xs text-slate-800 placeholder-slate-400 focus:outline-none focus:border-brand-500 transition-colors"
          />
        </div>
      )}

      {/* Trip Cards Container */}
      <div>
        {loadingTrips ? (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-2.5 py-2">
            {[1, 2, 3, 4, 5].map((idx) => (
              <div
                key={idx}
                className="h-16 rounded-xl bg-slate-100 animate-pulse flex flex-col justify-center p-3 space-y-2"
              >
                <div className="h-3 bg-slate-200 rounded w-16"></div>
                <div className="h-2 bg-slate-200/60 rounded w-10"></div>
              </div>
            ))}
          </div>
        ) : !isOnline ? (
          <div className="py-7 px-4 text-center rounded-xl bg-slate-50 border border-dashed border-slate-200 space-y-2">
            <Radio className="w-7 h-7 text-slate-400 mx-auto" />
            <div className="text-xs font-semibold text-slate-700">Awaiting FastAPI Connection</div>
            <p className="text-xs text-slate-500 max-w-md mx-auto">
              Evaluation trips are retrieved dynamically from the FastAPI backend on Laptop 1.
              Start FastAPI with <code className="font-mono text-slate-700 bg-slate-200 px-1.5 py-0.5 rounded text-[11px]">uvicorn api.main:app --port 8000</code> to view and test trips.
            </p>
          </div>
        ) : tripsError ? (
          <div className="py-5 px-4 text-center rounded-xl bg-rose-50 border border-rose-200 space-y-1">
            <div className="text-xs font-semibold text-rose-800">Failed to load evaluation trips</div>
            <p className="text-xs text-rose-600 font-mono">{tripsError}</p>
          </div>
        ) : trips.length === 0 ? (
          <div className="py-7 px-4 text-center rounded-xl bg-slate-50 border border-dashed border-slate-200 space-y-2">
            <Navigation className="w-7 h-7 text-slate-400 mx-auto" />
            <div className="text-xs font-semibold text-slate-700">No Evaluation Trips Returned</div>
            <p className="text-xs text-slate-500 max-w-md mx-auto">
              FastAPI responded with 0 trips on <code className="font-mono text-slate-700">/evaluation/trips</code>.
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-2.5 max-h-56 overflow-y-auto p-0.5">
            {filteredTrips.map((trip) => {
              const isSelected = selectedTripId === trip.id;
              return (
                <button
                  key={trip.id}
                  onClick={() => onSelectTrip(trip.id)}
                  disabled={isReconstructing}
                  className={`p-3 rounded-xl border text-left transition-all relative flex flex-col justify-between ${
                    isSelected
                      ? 'bg-brand-50 border-brand-500 text-brand-950 shadow-soft ring-1 ring-brand-500'
                      : 'bg-white border-slate-200 text-slate-700 hover:bg-slate-50 hover:border-slate-300'
                  }`}
                >
                  <div className="flex items-center justify-between gap-1 w-full mb-1">
                    <span className="text-[10px] font-mono uppercase tracking-wider text-slate-400">
                      Trip ID
                    </span>
                    {isSelected && (
                      <CheckCircle2 className="w-3.5 h-3.5 text-brand-600 flex-shrink-0" />
                    )}
                  </div>

                  <div className="font-mono font-bold text-xs text-slate-900 truncate w-full" title={trip.id}>
                    {trip.id}
                  </div>

                  {trip.pointsCount && (
                    <div className="mt-1 text-[10px] text-slate-500 font-mono">
                      {trip.pointsCount} points
                    </div>
                  )}
                </button>
              );
            })}
          </div>
        )}
      </div>

      {/* Selected Trip Indicator Footer */}
      {selectedTripId && (
        <div className="flex items-center justify-between px-3.5 py-2 rounded-xl bg-slate-50 border border-slate-200/80 text-xs text-slate-600">
          <div className="flex items-center gap-2">
            <span className="text-slate-500">Selected Trip:</span>
            <span className="font-mono font-bold text-brand-700 bg-brand-50 px-2 py-0.5 rounded border border-brand-200">
              {selectedTripId}
            </span>
          </div>
          <span className="text-[11px] text-slate-500">Ready for evaluation</span>
        </div>
      )}
    </div>
  );
}
