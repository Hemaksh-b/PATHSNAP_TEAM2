import React from 'react';
import { Target, BarChart2, Activity, Zap, CheckCircle2, AlertCircle, Info, Clock } from 'lucide-react';

export function MetricsCard({
  predictionResult,
  selectedTripId,
  isReconstructing,
  error,
}) {
  if (error) {
    return (
      <div className="card-panel p-6 border-rose-200 bg-rose-50/50 space-y-2">
        <div className="flex items-center gap-2 text-rose-700">
          <AlertCircle className="w-5 h-5 flex-shrink-0" />
          <h3 className="font-semibold text-sm">Reconstruction Failed</h3>
        </div>
        <p className="text-xs text-rose-800 font-mono bg-white p-3 rounded-xl border border-rose-200 shadow-soft">
          {error}
        </p>
      </div>
    );
  }

  if (!predictionResult) {
    return (
      <div className="card-panel p-5 text-center space-y-2 bg-slate-50/60 border-dashed">
        <div className="flex items-center justify-center gap-2 text-slate-500 text-xs font-medium">
          <Activity className="w-4 h-4 text-slate-400" />
          <span>
            {isReconstructing
              ? 'Computing route reconstruction and evaluation metrics...'
              : 'Evaluation metrics (Top-1 Accuracy, Macro F1, Route Continuity) will display here upon reconstruction'}
          </span>
        </div>
      </div>
    );
  }

  const { tripId, metrics, roundTripTimeMs } = predictionResult;

  const formatMetricValue = (val, asPercent = false) => {
    if (val === null || val === undefined) {
      return { main: 'N/A', sub: 'Not provided' };
    }
    const num = Number(val);
    if (isNaN(num)) {
      return { main: String(val), sub: '' };
    }

    if (asPercent && num >= 0 && num <= 1) {
      return {
        main: `${(num * 100).toFixed(2)}%`,
        sub: `raw: ${num.toFixed(4)}`,
      };
    }

    return {
      main: num.toFixed(4),
      sub: '',
    };
  };

  const accuracyData = formatMetricValue(metrics?.accuracy, true);
  const f1Data = formatMetricValue(metrics?.macroF1, true);
  const continuityData = formatMetricValue(metrics?.routeContinuity, true);
  const logLossData = metrics?.logLoss !== null && metrics?.logLoss !== undefined
    ? formatMetricValue(metrics.logLoss, false)
    : null;

  return (
    <div className="card-panel p-6 space-y-5">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-100">
        <div className="flex items-center gap-2.5">
          <span className="px-2.5 py-1 rounded-full text-[11px] font-mono uppercase tracking-wider bg-emerald-50 border border-emerald-200 text-emerald-700 font-semibold flex items-center gap-1">
            <CheckCircle2 className="w-3.5 h-3.5" />
            Evaluation Metrics
          </span>
          <span className="text-xs text-slate-500 font-mono">
            Trip: <strong className="text-slate-900 font-mono">{tripId || selectedTripId}</strong>
          </span>
        </div>

        {/* Inference time badge */}
        <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-slate-100 text-slate-600 text-xs">
          <Clock className="w-3.5 h-3.5 text-brand-600" />
          <span>Inference Time:</span>
          <span className="font-mono font-bold text-slate-900">
            {metrics?.inferenceTime !== null && metrics?.inferenceTime !== undefined
              ? `${metrics.inferenceTime} ${metrics.inferenceTimeUnit || 'ms'}`
              : `${roundTripTimeMs} ms (RTT)`}
          </span>
        </div>
      </div>

      {/* Metrics Cards Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* 1. Observation Top-1 Accuracy */}
        <div className="p-4 rounded-xl bg-slate-50/80 border border-slate-200/80 space-y-1.5 hover:bg-slate-50 transition-colors">
          <div className="flex items-center justify-between text-slate-500">
            <span className="text-[11px] font-bold uppercase tracking-wider text-slate-600">
              Top-1 Accuracy
            </span>
            <div className="w-7 h-7 rounded-lg bg-indigo-50 border border-indigo-100 flex items-center justify-center text-indigo-600">
              <Target className="w-4 h-4" />
            </div>
          </div>

          <div className="pt-0.5">
            <div className="text-2xl font-bold font-mono text-slate-900 tracking-tight">
              {accuracyData.main}
            </div>
            {accuracyData.sub && (
              <div className="text-[10px] font-mono text-slate-400">
                {accuracyData.sub}
              </div>
            )}
          </div>

          <p className="text-[11px] text-slate-500 leading-snug">
            Observation-level candidate match accuracy
          </p>
        </div>

        {/* 2. Macro F1-score */}
        <div className="p-4 rounded-xl bg-slate-50/80 border border-slate-200/80 space-y-1.5 hover:bg-slate-50 transition-colors">
          <div className="flex items-center justify-between text-slate-500">
            <span className="text-[11px] font-bold uppercase tracking-wider text-slate-600">
              Macro F1-Score
            </span>
            <div className="w-7 h-7 rounded-lg bg-cyan-50 border border-cyan-100 flex items-center justify-center text-cyan-600">
              <BarChart2 className="w-4 h-4" />
            </div>
          </div>

          <div className="pt-0.5">
            <div className="text-2xl font-bold font-mono text-slate-900 tracking-tight">
              {f1Data.main}
            </div>
            {f1Data.sub && (
              <div className="text-[10px] font-mono text-slate-400">
                {f1Data.sub}
              </div>
            )}
          </div>

          <p className="text-[11px] text-slate-500 leading-snug">
            Unweighted harmonic mean across road segments
          </p>
        </div>

        {/* 3. Route Continuity */}
        <div className="p-4 rounded-xl bg-slate-50/80 border border-slate-200/80 space-y-1.5 hover:bg-slate-50 transition-colors">
          <div className="flex items-center justify-between text-slate-500">
            <span className="text-[11px] font-bold uppercase tracking-wider text-slate-600">
              Route Continuity
            </span>
            <div className="w-7 h-7 rounded-lg bg-emerald-50 border border-emerald-100 flex items-center justify-center text-emerald-600">
              <Activity className="w-4 h-4" />
            </div>
          </div>

          <div className="pt-0.5">
            <div className="text-2xl font-bold font-mono text-slate-900 tracking-tight">
              {continuityData.main}
            </div>
            {continuityData.sub && (
              <div className="text-[10px] font-mono text-slate-400">
                {continuityData.sub}
              </div>
            )}
          </div>

          <p className="text-[11px] text-slate-500 leading-snug">
            Sequential road graph reachability
          </p>
        </div>

        {/* 4. Log Loss (if computed by backend) */}
        {logLossData ? (
          <div className="p-4 rounded-xl bg-slate-50/80 border border-slate-200/80 space-y-1.5 hover:bg-slate-50 transition-colors">
            <div className="flex items-center justify-between text-slate-500">
              <span className="text-[11px] font-bold uppercase tracking-wider text-slate-600">
                Log Loss
              </span>
              <div className="w-7 h-7 rounded-lg bg-amber-50 border border-amber-100 flex items-center justify-center text-amber-600">
                <Zap className="w-4 h-4" />
              </div>
            </div>

            <div className="pt-0.5">
              <div className="text-2xl font-bold font-mono text-slate-900 tracking-tight">
                {logLossData.main}
              </div>
            </div>

            <p className="text-[11px] text-slate-500 leading-snug">
              Cross-entropy loss on candidate probabilities
            </p>
          </div>
        ) : (
          <div className="p-4 rounded-xl bg-slate-50/40 border border-dashed border-slate-200 space-y-1.5 flex flex-col justify-between">
            <div className="flex items-center justify-between text-slate-400">
              <span className="text-[11px] font-bold uppercase tracking-wider text-slate-500">
                Log Loss
              </span>
              <Info className="w-4 h-4 text-slate-400" />
            </div>

            <div className="py-1">
              <div className="text-sm font-mono text-slate-400">
                Not Computed
              </div>
              <p className="text-[10px] text-slate-400 mt-0.5">
                Backend provided discrete Viterbi output
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
