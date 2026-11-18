import React from 'react';
import { ConnectionBadge } from '../Common/ConnectionBadge';
import { ShieldCheck } from 'lucide-react';

export function Header({ title, subtitle, status, latencyMs, baseUrl, onRefresh }) {
  return (
    <header className="h-16 px-8 border-b border-slate-200 bg-white/95 backdrop-blur-sm flex items-center justify-between sticky top-0 z-30 shadow-sm">
      <div className="flex flex-col">
        <h1 className="text-base md:text-lg font-bold text-slate-900 tracking-tight font-sans">
          {title}
        </h1>
        {subtitle && (
          <p className="text-xs text-slate-500 font-normal hidden sm:block">
            {subtitle}
          </p>
        )}
      </div>

      <div className="flex items-center gap-3">
        {/* Evaluator Badge */}
        <div className="hidden md:flex items-center gap-1.5 px-3 py-1 rounded-full bg-slate-100 border border-slate-200 text-slate-600 text-xs font-medium">
          <ShieldCheck className="w-3.5 h-3.5 text-brand-600" />
          <span>Evaluation Session</span>
        </div>

        {/* Backend Connection Indicator */}
        <ConnectionBadge
          status={status}
          latencyMs={latencyMs}
          baseUrl={baseUrl}
          onRefresh={onRefresh}
        />
      </div>
    </header>
  );
}
