import React, { useState } from 'react';
import { Sidebar } from './components/Navigation/Sidebar';
import { Header } from './components/Navigation/Header';
import { DashboardPage } from './pages/DashboardPage';
import { RouteExplorerPage } from './pages/RouteExplorerPage';
import { SystemStatusPage } from './pages/SystemStatusPage';
import { useBackendStatus } from './hooks/useBackendStatus';
import { useEvaluation } from './hooks/useEvaluation';

export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard'); // 'dashboard' | 'explorer' | 'status'

  // Central hooks
  const backendStatus = useBackendStatus(12000);
  const evaluation = useEvaluation(backendStatus.isOnline);

  const getPageInfo = () => {
    switch (activeTab) {
      case 'dashboard':
        return {
          title: 'PathSnap Evaluation Dashboard',
          subtitle: 'Distributed Map-Matching & Route Reconstruction for University ML Hackathon',
        };
      case 'explorer':
        return {
          title: 'Route Explorer & Geographical Inspection',
          subtitle: 'Analyze reconstructed trajectories, GPS observations, and Viterbi sequences',
        };
      case 'status':
        return {
          title: 'System Diagnostics & Network Configuration',
          subtitle: 'Two-Laptop deployment monitor and FastAPI API contract probe',
        };
      default:
        return {
          title: 'PathSnap',
          subtitle: '',
        };
    }
  };

  const pageInfo = getPageInfo();

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-slate-50 text-slate-900 font-sans">
      {/* Sidebar Navigation */}
      <Sidebar
        activeTab={activeTab}
        onSelectTab={setActiveTab}
        isOnline={backendStatus.isOnline}
        tripCount={evaluation.trips.length}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col h-full overflow-hidden bg-slate-50">
        {/* Top Header */}
        <Header
          title={pageInfo.title}
          subtitle={pageInfo.subtitle}
          status={backendStatus.status}
          latencyMs={backendStatus.latencyMs}
          baseUrl={backendStatus.baseUrl}
          onRefresh={backendStatus.checkNow}
        />

        {/* Scrollable Page Viewport */}
        <main className="flex-1 overflow-y-auto p-6 md:p-8 bg-slate-50">
          {activeTab === 'dashboard' && (
            <DashboardPage
              evaluation={evaluation}
              backendStatus={backendStatus}
              onNavigateToStatus={() => setActiveTab('status')}
            />
          )}

          {activeTab === 'explorer' && (
            <RouteExplorerPage
              evaluation={evaluation}
              backendStatus={backendStatus}
            />
          )}

          {activeTab === 'status' && (
            <SystemStatusPage
              backendStatus={backendStatus}
              evaluation={evaluation}
            />
          )}
        </main>
      </div>
    </div>
  );
}
