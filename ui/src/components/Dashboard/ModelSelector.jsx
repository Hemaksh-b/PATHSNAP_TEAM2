import React from 'react';
import { Cpu, CheckCircle2, Lock } from 'lucide-react';

/**
 * Choose which decoding model Laptop 1 should run (GET /models).
 *
 * @param {{
 *   models: Array<{id: string, label: string, description: string, uses_hmm: boolean,
 *                  uses_classifier: boolean, available: boolean}>,
 *   selectedModel: string | null,
 *   onSelectModel: (id: string) => void,
 *   classifierName: string | null,
 *   isDisabled: boolean,
 *   error: string | null,
 * }} props
 */
export function ModelSelector({ models = [], selectedModel, onSelectModel, classifierName, isDisabled, error }) {
  if (error) {
    return (
      <div className="card-panel p-4 text-xs text-rose-700 bg-rose-50 border border-rose-200">
        Could not load models from the backend: <span className="font-mono">{error}</span>
      </div>
    );
  }
  if (!models.length) {
    return null;
  }

  return (
    <div className="card-panel p-6 space-y-4">
      <div>
        <h2 className="text-sm font-bold text-slate-900 flex items-center gap-2">
          <Cpu className="w-4 h-4 text-brand-600" />
          Model Selection
        </h2>
        <p className="text-xs text-slate-500 mt-0.5">
          Choose how Laptop 1 decodes the route. Exported classifier:{' '}
          <span className="font-mono text-slate-700">{classifierName || 'none'}</span>
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2.5">
        {models.map((m) => {
          const isSelected = selectedModel === m.id;
          const isLocked = !m.available;
          return (
            <button
              key={m.id}
              type="button"
              onClick={() => onSelectModel(m.id)}
              disabled={isDisabled || isLocked}
              title={isLocked ? 'Needs an exported (non-mock) classifier on Laptop 1' : m.description}
              className={`p-3 rounded-xl border text-left transition-all flex flex-col gap-1.5 ${
                isSelected
                  ? 'bg-brand-50 border-brand-500 ring-1 ring-brand-500'
                  : 'bg-white border-slate-200 hover:bg-slate-50 hover:border-slate-300'
              } ${isLocked ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
              <div className="flex items-center justify-between gap-2">
                <span className="text-xs font-bold text-slate-900">{m.label}</span>
                {isSelected && <CheckCircle2 className="w-3.5 h-3.5 text-brand-600 flex-shrink-0" />}
                {isLocked && <Lock className="w-3.5 h-3.5 text-slate-400 flex-shrink-0" />}
              </div>
              <p className="text-[11px] text-slate-500 leading-snug">{m.description}</p>
              <div className="flex gap-1.5 mt-auto">
                <span className={`text-[10px] px-1.5 py-0.5 rounded ${m.uses_classifier ? 'bg-violet-100 text-violet-700' : 'bg-slate-100 text-slate-500'}`}>
                  {m.uses_classifier ? 'classifier' : 'distance'}
                </span>
                <span className={`text-[10px] px-1.5 py-0.5 rounded ${m.uses_hmm ? 'bg-emerald-100 text-emerald-700' : 'bg-slate-100 text-slate-500'}`}>
                  {m.uses_hmm ? 'HMM + Viterbi' : 'no HMM'}
                </span>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
