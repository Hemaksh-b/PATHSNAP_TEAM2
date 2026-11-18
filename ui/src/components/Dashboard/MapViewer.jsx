import React, { useState } from 'react';
import { Map, Maximize2, Minimize2, ExternalLink, RefreshCw, Compass, MapPin } from 'lucide-react';

export function MapViewer({
  mapHtml,
  mapUrl,
  selectedTripId,
  isLoading,
  error,
  onReloadMap,
}) {
  const [isFullscreen, setIsFullscreen] = useState(false);

  const toggleFullscreen = () => {
    setIsFullscreen((prev) => !prev);
  };

  const hasMapContent = Boolean(mapHtml || mapUrl);

  return (
    <div
      className={`card-panel flex flex-col transition-all duration-300 ${
        isFullscreen
          ? 'fixed inset-4 z-50 bg-white border-slate-300 shadow-2xl overflow-hidden'
          : 'relative min-h-[540px] overflow-hidden'
      }`}
    >
      {/* Map Header Toolbar */}
      <div className="h-14 px-6 border-b border-slate-200 bg-slate-50/70 flex items-center justify-between flex-shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-white border border-slate-200 flex items-center justify-center text-brand-600 shadow-soft">
            <Map className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-xs md:text-sm font-bold text-slate-900 flex items-center gap-2">
              Interactive Route Map
              {selectedTripId && (
                <span className="font-mono text-xs px-2 py-0.5 rounded-full bg-brand-50 border border-brand-200 text-brand-700 font-semibold">
                  {selectedTripId}
                </span>
              )}
            </h3>
            <p className="text-[11px] text-slate-500">
              Folium Map Engine: GPS Observations vs. HMM Predicted Road Alignment
            </p>
          </div>
        </div>

        {/* Legend and Actions */}
        <div className="flex items-center gap-3">
          {/* Layer Legend */}
          <div className="hidden lg:flex items-center gap-3.5 px-3 py-1 rounded-full bg-white border border-slate-200 text-xs shadow-soft">
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-amber-500 ring-2 ring-amber-200"></span>
              <span className="text-slate-600 text-[11px] font-medium">GPS Points</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-4 h-1 rounded-full bg-brand-600 shadow-sm shadow-brand-500/50"></span>
              <span className="text-slate-600 text-[11px] font-medium">Viterbi Route</span>
            </div>
          </div>

          {/* Action buttons */}
          <div className="flex items-center gap-1">
            {onReloadMap && hasMapContent && (
              <button
                onClick={onReloadMap}
                disabled={isLoading}
                title="Reload Map"
                className="p-1.5 rounded-lg bg-white hover:bg-slate-100 border border-slate-200 text-slate-600 hover:text-slate-900 transition-colors disabled:opacity-50 shadow-soft"
              >
                <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin' : ''}`} />
              </button>
            )}

            {mapUrl && (
              <a
                href={mapUrl}
                target="_blank"
                rel="noreferrer"
                title="Open standalone map in new tab"
                className="p-1.5 rounded-lg bg-white hover:bg-slate-100 border border-slate-200 text-slate-600 hover:text-slate-900 transition-colors shadow-soft"
              >
                <ExternalLink className="w-3.5 h-3.5" />
              </a>
            )}

            <button
              onClick={toggleFullscreen}
              title={isFullscreen ? 'Exit Fullscreen' : 'Fullscreen Map'}
              className="p-1.5 rounded-lg bg-white hover:bg-slate-100 border border-slate-200 text-slate-600 hover:text-slate-900 transition-colors shadow-soft"
            >
              {isFullscreen ? (
                <Minimize2 className="w-3.5 h-3.5 text-brand-600" />
              ) : (
                <Maximize2 className="w-3.5 h-3.5" />
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Map Content Area */}
      <div className="relative flex-1 w-full bg-slate-100 flex flex-col min-h-[460px]">
        {isLoading ? (
          <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-white/80 backdrop-blur-sm space-y-3">
            <div className="relative">
              <div className="w-12 h-12 rounded-full border-2 border-brand-200 border-t-brand-600 animate-spin"></div>
              <Compass className="w-5 h-5 text-brand-600 absolute inset-0 m-auto" />
            </div>
            <div className="text-center space-y-0.5">
              <div className="text-sm font-semibold text-slate-800">
                Rendering Folium Map Layers...
              </div>
              <p className="text-xs text-slate-500">
                Aligning GPS observations with predicted road segments
              </p>
            </div>
          </div>
        ) : error ? (
          <div className="flex-1 flex flex-col items-center justify-center p-8 text-center space-y-2">
            <div className="w-10 h-10 rounded-full bg-rose-100 text-rose-600 flex items-center justify-center">
              <MapPin className="w-5 h-5" />
            </div>
            <div className="text-sm font-semibold text-rose-800">
              Map Visualization Unavailable
            </div>
            <p className="text-xs text-rose-600 max-w-md font-mono">
              {error}
            </p>
          </div>
        ) : hasMapContent ? (
          <iframe
            title={`Folium Route Map - ${selectedTripId || 'Prediction'}`}
            srcDoc={mapHtml || undefined}
            src={!mapHtml && mapUrl ? mapUrl : undefined}
            className="w-full h-full min-h-[460px] flex-1 border-0"
            sandbox="allow-scripts allow-same-origin allow-popups"
          />
        ) : (
          <div className="flex-1 flex flex-col items-center justify-center p-8 text-center space-y-3 bg-slate-50">
            <div className="w-14 h-14 rounded-2xl bg-white border border-slate-200 flex items-center justify-center text-slate-400 shadow-soft">
              <Map className="w-7 h-7 text-slate-400" />
            </div>

            <div className="space-y-1 max-w-sm">
              <h4 className="text-sm font-semibold text-slate-800">
                Awaiting Reconstruction
              </h4>
              <p className="text-xs text-slate-500 leading-relaxed">
                When route reconstruction runs, the interactive Folium map visualization will render here with full pan, zoom, and layer inspection.
              </p>
            </div>

            <div className="flex items-center gap-5 pt-1 text-[11px] text-slate-400 font-mono">
              <span className="flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-slate-400"></span>
                Folium Engine
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-slate-400"></span>
                OpenStreetMap
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-slate-400"></span>
                Viterbi Polyline
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
