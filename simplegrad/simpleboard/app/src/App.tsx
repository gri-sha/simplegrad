import { useState, useEffect, useCallback, useRef } from 'react';
import './App.css';
import { TopBar } from './components/TopBar';
import { Sidebar } from './components/Sidebar';
import { MainContent } from './components/MainContent';
import { SettingsModal } from './components/SettingsModal';
import { api } from './api';
import type { RunInfo, RecordInfo, CompGraphData, RunMeta } from './types';

interface GraphInfo {
  id: number;
  graph: CompGraphData;
  created_at: number;
}

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [activeTab, setActiveTab] = useState<'metrics' | 'graphs' | 'hparams'>('metrics');
  const [runMeta, setRunMeta] = useState<Record<number, RunMeta>>(() => {
    const saved = localStorage.getItem('sb_run_meta');
    return saved ? JSON.parse(saved) : {};
  });

  const updateRunMeta = (runId: number, meta: Partial<RunMeta>) => {
    setRunMeta(prev => {
      const next = { ...prev, [runId]: { ...(prev[runId] || {}), ...meta } };
      localStorage.setItem('sb_run_meta', JSON.stringify(next));
      return next;
    });
  };

  // Data state
  const [databases, setDatabases] = useState<string[]>([]);
  const [selectedDb, setSelectedDb] = useState<string | null>(null);
  const [runs, setRuns] = useState<RunInfo[]>([]);
  const [selectedRunIds, setSelectedRunIds] = useState<number[]>([]);
  const [runNames, setRunNames] = useState<Record<number, string>>({});
  const [metricsByRun, setMetricsByRun] = useState<Record<number, Record<string, RecordInfo[]>>>({});
  const [graphsByRun, setGraphsByRun] = useState<Record<number, GraphInfo[]>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // One WebSocket per selected run
  const wsMap = useRef<Map<number, WebSocket>>(new Map());

  const fetchDatabases = useCallback(async () => {
    try {
      setError(null);
      const data = await api.getDatabases();
      setDatabases(data.available_databases);
      if (data.current_database) {
        setSelectedDb(data.current_database);
      }
    } catch (err) {
      setError('Failed to fetch databases');
      console.error(err);
    }
  }, []);

  const fetchRuns = useCallback(async () => {
    if (!selectedDb) {
      setRuns([]);
      return;
    }
    try {
      setLoading(true);
      const data = await api.getRuns();
      setRuns(data || []);
    } catch (err) {
      setError('Failed to fetch runs');
      console.error(err);
    } finally {
      setLoading(false);
    }
  }, [selectedDb]);

  const fetchRunData = useCallback(async (runId: number, withSpinner = true) => {
    if (withSpinner) setLoading(true);
    setError(null);

    // Use allSettled so a single failed endpoint (e.g. histograms on an old DB)
    // doesn't tank the whole load — scalars and graphs should still display.
    const [runInfoR, metricsR, graphsR] = await Promise.allSettled([
      api.getRun(runId),
      api.getRecords(runId),
      api.getGraphs(runId),
    ]);

    if (runInfoR.status === 'fulfilled') {
      setRunNames((prev) => ({ ...prev, [runId]: runInfoR.value.name }));
    } else {
      setError('Failed to fetch run');
      console.error(runInfoR.reason);
    }
    if (metricsR.status === 'fulfilled') {
      setMetricsByRun((prev) => ({ ...prev, [runId]: metricsR.value.metrics || {} }));
    } else console.error('records fetch failed', metricsR.reason);
    if (graphsR.status === 'fulfilled') {
      setGraphsByRun((prev) => ({ ...prev, [runId]: graphsR.value.graphs || [] }));
    } else console.error('graphs fetch failed', graphsR.reason);

    if (withSpinner) setLoading(false);
  }, []);

  // Database selection clears all selected runs
  const handleDbSelect = async (dbName: string) => {
    if (!dbName) return;
    try {
      await api.selectDatabase(dbName);
      setSelectedDb(dbName);
      setSelectedRunIds([]);
      setRunNames({});
      setMetricsByRun({});
      setGraphsByRun({});
      // Close all sockets
      wsMap.current.forEach((ws) => ws.close());
      wsMap.current.clear();
    } catch (err) {
      setError('Failed to select database');
      console.error(err);
    }
  };

  const openSocket = useCallback((runId: number) => {
    if (wsMap.current.has(runId)) return;
    const ws = api.createWebSocket(runId);
    if (!ws) return;

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        if (message.type === 'metric_update') {
          setMetricsByRun((prev) => {
            const runMetrics = { ...(prev[runId] || {}) };
            const metricName = message.data.metric_name;
            const list = runMetrics[metricName] ? [...runMetrics[metricName]] : [];
            list.push(message.data);
            runMetrics[metricName] = list;
            return { ...prev, [runId]: runMetrics };
          });
        } else if (message.type === 'graph_update') {
          setGraphsByRun((prev) => {
            const list = prev[runId] ? [...prev[runId]] : [];
            list.push(message.data);
            return { ...prev, [runId]: list };
          });
        }
      } catch (err) {
        console.error('WebSocket message error:', err);
      }
    };

    ws.onerror = (err) => {
      console.error('WebSocket error:', err);
    };

    wsMap.current.set(runId, ws);
  }, []);

  const closeSocket = useCallback((runId: number) => {
    const ws = wsMap.current.get(runId);
    if (ws) {
      ws.close();
      wsMap.current.delete(runId);
    }
  }, []);

  const handleToggleRun = (runId: number) => {
    setSelectedRunIds((prev) => {
      if (prev.includes(runId)) {
        closeSocket(runId);
        // Drop cached data so it doesn't leak into future selections
        const dropOne = <T,>(m: Record<number, T>) => {
          const next = { ...m };
          delete next[runId];
          return next;
        };
        setMetricsByRun(dropOne);
        setGraphsByRun(dropOne);
        return prev.filter((id) => id !== runId);
      }
      fetchRunData(runId);
      openSocket(runId);
      return [...prev, runId];
    });
  };

  const handleClearRuns = () => {
    selectedRunIds.forEach(closeSocket);
    setSelectedRunIds([]);
    setMetricsByRun({});
    setGraphsByRun({});
    setRunNames({});
  };

  const handleSettingsSave = () => {
    wsMap.current.forEach((ws) => ws.close());
    wsMap.current.clear();
    setDatabases([]);
    setSelectedDb(null);
    setRuns([]);
    setSelectedRunIds([]);
    setRunNames({});
    setMetricsByRun({});
    setGraphsByRun({});
    fetchDatabases();
  };

  const handleRefresh = () => {
    fetchDatabases();
    if (selectedDb) fetchRuns();
    selectedRunIds.forEach((id) => fetchRunData(id));
  };

  // Initial fetch
  useEffect(() => {
    fetchDatabases();
  }, [fetchDatabases]);

  useEffect(() => {
    fetchRuns();
  }, [selectedDb, fetchRuns]);

  // Background poll for databases + runs
  useEffect(() => {
    const pollInterval = setInterval(() => {
      api
        .getDatabases()
        .then((data) => {
          setDatabases(data.available_databases);
          if (data.current_database && !selectedDb) {
            setSelectedDb(data.current_database);
          }
        })
        .catch((err) => console.error('Polling databases error:', err));

      if (selectedDb) {
        api
          .getRuns()
          .then((data) => setRuns(data || []))
          .catch((err) => console.error('Polling runs error:', err));
      }
    }, 3000);

    return () => clearInterval(pollInterval);
  }, [selectedDb]);

  // Background poll for metrics/graphs/histograms/images of every selected run
  useEffect(() => {
    if (selectedRunIds.length === 0) return;

    const pollInterval = setInterval(() => {
      selectedRunIds.forEach((runId) => {
        api
          .getRecords(runId)
          .then((metricsData) => {
            setMetricsByRun((prev) => ({ ...prev, [runId]: metricsData.metrics || {} }));
          })
          .catch((err) => console.error('Polling records error:', err));

        api
          .getGraphs(runId)
          .then((graphsData) => {
            setGraphsByRun((prev) => ({ ...prev, [runId]: graphsData.graphs || [] }));
          })
          .catch((err) => console.error('Polling graphs error:', err));
      });
    }, 2000);

    return () => clearInterval(pollInterval);
  }, [selectedRunIds]);

  // Cleanup all WebSockets on unmount
  useEffect(() => {
    const map = wsMap.current;
    return () => {
      map.forEach((ws) => ws.close());
      map.clear();
    };
  }, []);

  return (
    <div className="app">
      <TopBar
        onRefresh={handleRefresh}
        onOpenSettings={() => setSettingsOpen(true)}
        isLoading={loading}
      />

      <div className="app-body">
        <Sidebar
          runs={runs}
          selectedRunIds={selectedRunIds}
          onToggleRun={handleToggleRun}
          onClearRuns={handleClearRuns}
          isOpen={sidebarOpen}
          onToggle={() => setSidebarOpen(!sidebarOpen)}
          databases={databases}
          currentDatabase={selectedDb}
          onSelectDatabase={handleDbSelect}
          runMeta={runMeta}
          updateRunMeta={updateRunMeta}
        />

        <MainContent
          selectedRunIds={selectedRunIds}
          runNames={Object.fromEntries(
            Object.entries(runNames).map(([id, name]) => [id, runMeta[Number(id)]?.rename || name])
          )}
          rawRunNames={runNames}
          runMeta={runMeta}
          runs={runs}
          metricsByRun={metricsByRun}
          graphsByRun={graphsByRun}
          isLoading={loading}
          error={error}
          activeTab={activeTab}
          onTabChange={setActiveTab}
        />
      </div>

      <SettingsModal
        isOpen={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        onSave={handleSettingsSave}
      />
    </div>
  );
}

export default App;
