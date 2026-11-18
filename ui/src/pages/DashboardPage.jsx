import React from 'react';
import { OverviewBanner } from '../components/Dashboard/OverviewBanner';
import { TripSelector } from '../components/Dashboard/TripSelector';
import { ModelSelector } from '../components/Dashboard/ModelSelector';
import { MetricsCard } from '../components/Dashboard/MetricsCard';
import { MapViewer } from '../components/Dashboard/MapViewer';

export function DashboardPage({
  evaluation,
  backendStatus,
  onNavigateToStatus,
}) {
  const {
    models,
    selectedModel,
    selectModel,
    classifierName,
    modelsError,
    trips,
    loadingTrips,
    tripsError,
    selectedTripId,
    selectTrip,
    loadTrips,
    runReconstruction,
    isReconstructing,
    reconstructionProgress,
    predictionResult,
    predictionError,
    mapHtml,
    mapUrl,
    loadingMap,
    mapError,
  } = evaluation;

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Overview Banner */}
      <OverviewBanner
        tripCount={trips.length}
        selectedTripId={selectedTripId}
        isOnline={backendStatus.isOnline}
        isChecking={backendStatus.isChecking}
        error={backendStatus.error}
        baseUrl={backendStatus.baseUrl}
        onNavigateToStatus={onNavigateToStatus}
      />

      {/* Model Selection Section */}
      <ModelSelector
        models={models}
        selectedModel={selectedModel}
        onSelectModel={selectModel}
        classifierName={classifierName}
        isDisabled={isReconstructing}
        error={modelsError}
      />

      {/* Trip Selection Section */}
      <TripSelector
        trips={trips}
        selectedTripId={selectedTripId}
        onSelectTrip={selectTrip}
        onRunReconstruction={runReconstruction}
        isReconstructing={isReconstructing}
        reconstructionProgress={reconstructionProgress}
        loadingTrips={loadingTrips}
        onRefreshTrips={loadTrips}
        isOnline={backendStatus.isOnline}
        tripsError={tripsError}
      />

      {/* Route Results & Metrics Section */}
      <MetricsCard
        predictionResult={predictionResult}
        selectedTripId={selectedTripId}
        isReconstructing={isReconstructing}
        error={predictionError}
      />

      {/* Interactive Folium Route Map */}
      <MapViewer
        mapHtml={mapHtml}
        mapUrl={mapUrl}
        selectedTripId={selectedTripId}
        isLoading={loadingMap}
        error={mapError}
        onReloadMap={() => runReconstruction(selectedTripId)}
      />
    </div>
  );
}
