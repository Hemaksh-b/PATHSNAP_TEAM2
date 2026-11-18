import React from 'react';
import { RefreshCw } from 'lucide-react';

export function ConnectionBadge({ status, latencyMs, baseUrl, onRefresh, compact = false }) {
  const isOnline = status === 'online';
  const isChecking = status === 'checking';

  if (compact) {
    return (
      <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-slate-100 border border-slate-200 text-xs">
        <span className="relative flex h-2 w-2">
          {isOnline && (
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
          )}
          <span
            className={`relative inline-flex rounded-full h-2 w-2 ${
              isOnline ? 'bg-emerald-500' : isChecking ? 'bg-amber-400' : 'bg-rose-500'
            }`}
          ></span>
        </span>
        <span className="text-slate-700 font-medium font-mono text-[11px]">
          {isOnline ? 'FastAPI Online' : isChecking ? 'Checking...' : 'FastAPI Offline'}
        </span>
        {latencyMs !== null && isOnline && (
          <span className="text-slate-500 text-[10px]">({latencyMs}ms)</span>
        )}
      </div>
    );
  }

  return (
    <div className="flex items-center gap-2.5 px-3.5 py-1.5 rounded-full bg-slate-100 border border-slate-200/90 text-xs shadow-soft">
      <div className="flex items-center gap-2">
        <span className="relative flex h-2.5 w-2.5">
          {isOnline && (
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
          )}
          <span
            className={`relative inline-flex rounded-full h-2.5 w-2.5 ${
              isOnline ? 'bg-emerald-500' : isChecking ? 'bg-amber-400' : 'bg-rose-500'
            }`}
          ></span>
        </span>
        <div className="flex flex-col">
          <div className="flex items-center gap-1.5">
            <span className="text-xs font-semibold text-slate-800">
              {isOnline ? 'FastAPI Connected' : isChecking ? 'Connecting...' : 'FastAPI Offline'}
            </span>
            {latencyMs !== null && isOnline && (
              <span className="px-1.5 py-0.2 rounded bg-emerald-100 text-[10px] font-mono text-emerald-700 font-medium">
                {latencyMs}ms
              </span>
            )}
          </div>
          <span className="text-[10px] text-slate-500 font-mono truncate max-w-[140px]" title={baseUrl}>
            {baseUrl}
          </span>
        </div>
      </div>

      {onRefresh && (
        <button
          onClick={onRefresh}
          disabled={isChecking}
          title="Probe connection"
          className="p-1 hover:bg-slate-200 rounded-full text-slate-500 hover:text-slate-800 transition-colors disabled:opacity-50 ml-1"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${isChecking ? 'animate-spin' : ''}`} />
        </button>
      )}
    </div>
  );
}
