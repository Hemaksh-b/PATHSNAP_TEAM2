import { useState, useEffect, useCallback } from 'react';
import { checkBackendHealth, getApiBaseUrl, setApiBaseUrl, resetApiBaseUrl } from '../services/api';

export function useBackendStatus(pollIntervalMs = 15000) {
  const [status, setStatus] = useState('checking'); // 'online' | 'offline' | 'checking'
  const [latencyMs, setLatencyMs] = useState(null);
  const [lastCheckTime, setLastCheckTime] = useState(null);
  const [error, setError] = useState(null);
  const [baseUrl, setBaseUrlState] = useState(getApiBaseUrl());

  const checkStatus = useCallback(async () => {
    setStatus('checking');
    try {
      const res = await checkBackendHealth();
      setLastCheckTime(new Date());
      setBaseUrlState(res.baseUrl);

      if (res.success) {
        setStatus('online');
        setLatencyMs(res.latencyMs);
        setError(null);
      } else {
        setStatus('offline');
        setLatencyMs(null);
        setError(res.error || 'Backend unreachable');
      }
    } catch (err) {
      setStatus('offline');
      setLatencyMs(null);
      setLastCheckTime(new Date());
      setError(err.message || 'Health check failed');
    }
  }, []);

  useEffect(() => {
    checkStatus();

    if (pollIntervalMs > 0) {
      const interval = setInterval(checkStatus, pollIntervalMs);
      return () => clearInterval(interval);
    }
  }, [checkStatus, pollIntervalMs]);

  const updateUrl = useCallback((newUrl) => {
    setApiBaseUrl(newUrl);
    setBaseUrlState(getApiBaseUrl());
    checkStatus();
  }, [checkStatus]);

  const resetUrl = useCallback(() => {
    resetApiBaseUrl();
    setBaseUrlState(getApiBaseUrl());
    checkStatus();
  }, [checkStatus]);

  return {
    status,
    isOnline: status === 'online',
    isOffline: status === 'offline',
    isChecking: status === 'checking',
    latencyMs,
    lastCheckTime,
    error,
    baseUrl,
    checkNow: checkStatus,
    updateUrl,
    resetUrl,
  };
}
