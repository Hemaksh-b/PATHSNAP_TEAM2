import { useState, useEffect, useCallback } from 'react';
import { fetchEvaluationTrips, fetchModels, runRouteReconstruction, getMapUrl, fetchMapHtml } from '../services/api';

export function useEvaluation(isBackendOnline) {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState(null);
  const [classifierName, setClassifierName] = useState(null);
  const [modelsError, setModelsError] = useState(null);

  const [trips, setTrips] = useState([]);
  const [loadingTrips, setLoadingTrips] = useState(false);
  const [tripsError, setTripsError] = useState(null);

  const [selectedTripId, setSelectedTripId] = useState(null);
  const [isReconstructing, setIsReconstructing] = useState(false);
  const [reconstructionProgress, setReconstructionProgress] = useState(null); // 'Sending payload', 'Viterbi map matching', etc.

  const [predictionResult, setPredictionResult] = useState(null);
  const [predictionError, setPredictionError] = useState(null);

  const [mapHtml, setMapHtml] = useState(null);
  const [mapUrl, setMapUrl] = useState(null);
  const [loadingMap, setLoadingMap] = useState(false);
  const [mapError, setMapError] = useState(null);

  const [runHistory, setRunHistory] = useState([]);

  // Fetch available trips whenever backend comes online or explicitly requested
  const loadTrips = useCallback(async () => {
    setLoadingTrips(true);
    setTripsError(null);

    const res = await fetchEvaluationTrips();

    if (res.success) {
      setTrips(res.trips);
      // Auto-select first trip if none selected
      if (!selectedTripId && res.trips.length > 0) {
        setSelectedTripId(res.trips[0].id);
      }
    } else {
      setTrips([]);
      setTripsError(res.error || 'Failed to retrieve evaluation trips from backend.');
    }

    setLoadingTrips(false);
  }, [selectedTripId]);

  // Fetch the selectable decoding models (GET /models)
  const loadModels = useCallback(async () => {
    const res = await fetchModels();
    if (!res.success) {
      setModels([]);
      setSelectedModel(null);
      // 404 = backend predates model selection: run its default model, show no error
      setModelsError(res.status === 404 ? null : (res.error || 'Failed to retrieve models from backend.'));
      return;
    }
    setModels(res.models);
    setClassifierName(res.classifier);
    setModelsError(null);
    setSelectedModel((current) => {
      const stillAvailable = res.models.some((m) => m.id === current && m.available);
      return stillAvailable ? current : res.defaultModel;
    });
  }, []);

  useEffect(() => {
    if (isBackendOnline) {
      loadTrips();
      loadModels();
    }
  }, [isBackendOnline, loadTrips, loadModels]);

  // Execute route reconstruction
  const runReconstruction = useCallback(async (tripIdToRun = null) => {
    const targetTripId = tripIdToRun || selectedTripId;
    if (!targetTripId || isReconstructing) return;

    setIsReconstructing(true);
    setPredictionError(null);
    setPredictionResult(null);
    setMapHtml(null);
    setMapUrl(null);
    setMapError(null);
    setReconstructionProgress('Transmitting trip to backend...');

    try {
      setReconstructionProgress('ML Feature Scoring & Viterbi Decoding in progress...');
      const res = await runRouteReconstruction(targetTripId, selectedModel);

      if (!res.success) {
        setPredictionError(res.error || 'Reconstruction request failed on backend.');
        setIsReconstructing(false);
        setReconstructionProgress(null);
        return;
      }

      setPredictionResult(res);
      setRunHistory((prev) => [
        {
          timestamp: new Date().toLocaleTimeString(),
          tripId: targetTripId,
          model: res.modelLabel || res.model,
          metrics: res.metrics,
          roundTripTimeMs: res.roundTripTimeMs,
        },
        ...prev.slice(0, 9), // Keep latest 10 runs
      ]);

      // Handle map loading:
      // If the predict response included HTML or map URL, use it directly
      const directMapUrl = getMapUrl(targetTripId, res.model || selectedModel);
      setMapUrl(directMapUrl);

      if (typeof res.mapData === 'string' && res.mapData.includes('<html')) {
        setMapHtml(res.mapData);
      } else {
        // Fetch map HTML from /maps/{trip_id}
        setLoadingMap(true);
        const mapRes = await fetchMapHtml(targetTripId, res.model || selectedModel);
        if (mapRes.success && mapRes.html) {
          setMapHtml(mapRes.html);
        } else {
          // If fetch failed, the iframe will attempt to load direct mapUrl
          if (!directMapUrl) {
            setMapError(mapRes.error || 'Map visualization unavailable');
          }
        }
        setLoadingMap(false);
      }
    } catch (err) {
      setPredictionError(err.message || 'An unexpected error occurred during reconstruction.');
    } finally {
      setIsReconstructing(false);
      setReconstructionProgress(null);
    }
  }, [selectedTripId, selectedModel, isReconstructing]);

  const selectTrip = useCallback((tripId) => {
    setSelectedTripId(tripId);
  }, []);

  const clearResults = useCallback(() => {
    setPredictionResult(null);
    setPredictionError(null);
    setMapHtml(null);
    setMapUrl(null);
  }, []);

  return {
    models,
    selectedModel,
    selectModel: setSelectedModel,
    classifierName,
    modelsError,
    loadModels,
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
    runHistory,
    clearResults,
  };
}
