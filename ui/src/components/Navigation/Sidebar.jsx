import React from 'react';
import { LayoutDashboard, MapPin, Activity, Laptop, Network } from 'lucide-react';

export function Sidebar({ activeTab, onSelectTab, isOnline, tripCount }) {
  const navItems = [
    {
      id: 'dashboard',
      label: 'Dashboard',
      icon: LayoutDashboard,
      badge: tripCount > 0 ? `${tripCount} Trips` : null,
    },
    {
      id: 'explorer',
      label: 'Route Explorer',
      icon: MapPin,
    },
    {
      id: 'status',
      label: 'System Status',
      icon: Activity,
      badge: isOnline ? 'Online' : 'Offline',
      badgeColor: isOnline ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' : 'bg-rose-500/10 text-rose-400 border-rose-500/20',
    },
  ];

  return (
    <aside className="w-64 flex-shrink-0 bg-slate-900 border-r border-slate-800 flex flex-col justify-between select-none">
      <div>
        {/* Brand Header */}
        <div className="h-16 flex items-center gap-3 px-6 border-b border-slate-800/80 bg-slate-950/60">
          <div className="w-9 h-9 rounded-xl bg-brand-600 flex items-center justify-center text-white shadow-md shadow-brand-600/30">
            <svg
              className="w-5 h-5 text-white"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <circle cx="6" cy="6" r="3" />
              <circle cx="18" cy="18" r="3" />
              <path d="M9 7.5L15 16.5" strokeDasharray="2 2" />
              <path d="M6 9v3a3 3 0 0 0 3 3h6" />
            </svg>
          </div>
          <div className="flex flex-col">
            <span className="font-bold tracking-tight text-white text-base font-sans">
              Path<span className="text-brand-400">Snap</span>
            </span>
            <span className="text-[10px] uppercase font-mono tracking-wider text-slate-400">
              Evaluator v1.0
            </span>
          </div>
        </div>

        {/* Navigation Items */}
        <div className="p-4 space-y-1.5">
          <div className="px-3 py-1.5 text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
            Workspace
          </div>
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => onSelectTab(item.id)}
                className={`w-full flex items-center justify-between px-3.5 py-2.5 rounded-xl text-sm font-medium transition-all ${
                  isActive
                    ? 'bg-brand-600 text-white shadow-md shadow-brand-600/20'
                    : 'text-slate-300 hover:text-white hover:bg-slate-800/60'
                }`}
              >
                <div className="flex items-center gap-3">
                  <Icon className={`w-4 h-4 ${isActive ? 'text-white' : 'text-slate-400'}`} />
                  <span>{item.label}</span>
                </div>
                {item.badge && (
                  <span
                    className={`text-[10px] font-mono px-2 py-0.5 rounded-full border ${
                      isActive
                        ? 'bg-brand-700 text-white border-brand-500'
                        : item.badgeColor || 'bg-slate-800 text-slate-300 border-slate-700'
                    }`}
                  >
                    {item.badge}
                  </span>
                )}
              </button>
            );
          })}
        </div>
      </div>

      {/* Distributed Setup Footer */}
      <div className="p-4 border-t border-slate-800/80 bg-slate-950/40">
        <div className="p-3.5 rounded-xl bg-slate-850/80 border border-slate-800 space-y-2 text-xs">
          <div className="flex items-center justify-between text-slate-400">
            <span className="font-semibold text-[11px] uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
              <Network className="w-3.5 h-3.5 text-brand-400" />
              Distributed Setup
            </span>
            <span className="text-[10px] font-mono text-brand-400">Team 2</span>
          </div>

          <div className="space-y-1.5 text-[11px] text-slate-400">
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-1.5 text-slate-300">
                <Laptop className="w-3 h-3 text-slate-400" />
                Laptop 1:
              </span>
              <span className="font-mono text-[10px] text-slate-300">FastAPI ML</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-1.5 text-slate-300">
                <Laptop className="w-3 h-3 text-brand-400" />
                Laptop 2:
              </span>
              <span className="font-mono text-[10px] text-brand-300">Evaluator UI</span>
            </div>
          </div>
        </div>
      </div>
    </aside>
  );
}
