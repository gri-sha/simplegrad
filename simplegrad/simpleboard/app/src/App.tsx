import { useState, useEffect, useCallback, useRef } from 'react';
import './App.css';
import { TopBar } from './components/TopBar';
import { Sidebar } from './components/Sidebar';
import { MainContent } from './components/MainContent';
import { SettingsModal } from './components/SettingsModal';
import { api } from './api';
import type { RunInfo, RecordInfo, CompGraphData } from './types';

interface GraphInfo {
  id: number;
  graph: CompGraphData;
  created_at: number;
}

function App() {
  // UI state
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [activeTab, setActiveTab] = useState<'metrics' | 'graphs'>('metrics');

  // Data state
  const [databases, setDatabases] = useState<string[]>([]);
  const [selectedDb, setSelectedDb] = useState<string | null>(null);
  const [runs, setRuns] = useState<RunInfo[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<number | null>(null);
  const [selectedRunName, setSelectedRunName] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<Record<string, RecordInfo[]>>({});
  const [graphs, setGraphs] = useState<GraphInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // WebSocket ref
  const wsRef = useRef<WebSocket | null>(null);

  // Fetch databases
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

  // Fetch runs for selected database
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

  // Fetch metrics and graphs for selected run
  const fetchRunData = useCallback(async (runId: number) => {
    try {
      setLoading(true);
      setError(null);
      
      // Fetch run info
      const runInfo = await api.getRun(runId);
      setSelectedRunName(runInfo.name);
      
      // Fetch metrics
      const metricsData = await api.getRecords(runId);
      setMetrics(metricsData.metrics || {});

      // Fetch graphs
      const graphsData = await api.getGraphs(runId);
      setGraphs(graphsData.graphs || []);
    } catch (err) {
      setError('Failed to fetch run data');
      console.error(err);
    } finally {
      setLoading(false);
    }
  }, []);

  // Handle database selection
  const handleDbSelect = async (dbName: string) => {
    if (!dbName) {
      return;
    }
    try {
      await api.selectDatabase(dbName);
      setSelectedDb(dbName);
      setSelectedRunId(null);
      setSelectedRunName(null);
      setMetrics({});
      setGraphs([]);
    } catch (err) {
      setError('Failed to select database');
      console.error(err);
    }
  };

  // Handle run selection
  const handleRunSelect = (runId: number) => {
    setSelectedRunId(runId);
    fetchRunData(runId);
    connectWebSocket(runId);
  };

  // WebSocket connection for real-time updates
  const connectWebSocket = useCallback((runId: number) => {
    // Close existing connection
    if (wsRef.current) {
      wsRef.current.close();
    }

    const ws = api.createWebSocket(runId);

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        if (message.type === 'metric_update') {
          setMetrics(prev => {
            const updated = { ...prev };
            const metricName = message.data.metric_name;
            if (!updated[metricName]) {
              updated[metricName] = [];
            }
            updated[metricName] = [...updated[metricName], message.data];
            return updated;
          });
        } else if (message.type === 'graph_update') {
          setGraphs(prev => [...prev, message.data]);
        }
      } catch (err) {
        console.error('WebSocket message error:', err);
      }
    };

    ws.onerror = (err) => {
      console.error('WebSocket error:', err);
    };

    wsRef.current = ws;
  }, []);

  // Handle settings save
  const handleSettingsSave = () => {
    // Reset state and refetch
    setDatabases([]);
    setSelectedDb(null);
    setRuns([]);
    setSelectedRunId(null);
    setSelectedRunName(null);
    setMetrics({});
    setGraphs([]);
    fetchDatabases();
  };

  // Refresh all data
  const handleRefresh = () => {
    fetchDatabases();
    if (selectedDb) {
      fetchRuns();
    }
    if (selectedRunId) {
      fetchRunData(selectedRunId);
    }
  };

  // Initial fetch
  useEffect(() => {
    fetchDatabases();
  }, [fetchDatabases]);

  // Fetch runs when database changes
  useEffect(() => {
    fetchRuns();
  }, [selectedDb, fetchRuns]);

  // Poll for updates when a run is selected (every 2 seconds)
  useEffect(() => {
    if (!selectedRunId) return;

    const pollInterval = setInterval(() => {
      // Silently fetch updated metrics without setting loading state
      api.getRecords(selectedRunId)
        .then(metricsData => {
          setMetrics(metricsData.metrics || {});
        })
        .catch(err => {
          console.error('Polling error:', err);
        });
      
      // Also fetch graphs
      api.getGraphs(selectedRunId)
        .then(graphsData => {
          setGraphs(graphsData.graphs || []);
        })
        .catch(err => {
          console.error('Polling graphs error:', err);
        });
    }, 2000);

    return () => clearInterval(pollInterval);
  }, [selectedRunId]);

  // Cleanup WebSocket on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
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
          selectedRunId={selectedRunId}
          onSelectRun={handleRunSelect}
          isOpen={sidebarOpen}
          onToggle={() => setSidebarOpen(!sidebarOpen)}
          databases={databases}
          currentDatabase={selectedDb}
          onSelectDatabase={handleDbSelect}
        />

        <MainContent
          selectedRunId={selectedRunId}
          runName={selectedRunName}
          metrics={metrics}
          graphs={graphs}
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
