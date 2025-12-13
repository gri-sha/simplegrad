import { useState, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { Graph } from './components/Graph';
import { SettingsModal } from './components/SettingsModal';
import { api } from './api';
import type { RunInfo } from './api';
import { Settings, RefreshCw, Layout } from 'lucide-react';
import simpleGradLogo from '/simplegrad.svg';

function App() {
  const [runs, setRuns] = useState<RunInfo[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<number | null>(null);
  const [metrics, setMetrics] = useState<Record<string, any[]>>({});
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [databases, setDatabases] = useState<string[]>([]);
  const [selectedDatabase, setSelectedDatabase] = useState<string | null>(null);

  const fetchRuns = async () => {
    try {
      setLoading(true);
      const data = await api.listRuns();
      setRuns(data);
      setError(null);
    } catch (err) {
      setError('Failed to connect to server. Check settings.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const fetchRunData = async (runId: number) => {
    try {
      setLoading(true);
      const metricList = await api.getMetrics(runId);
      
      const newMetrics: Record<string, any[]> = {};
      
      // Fetch all metrics in parallel
      await Promise.all(metricList.metrics.map(async (metricName) => {
        const records = await api.getRecords(runId, metricName);
        if (records.metrics && records.metrics[metricName]) {
          newMetrics[metricName] = records.metrics[metricName];
        }
      }));
      
      setMetrics(newMetrics);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const fetchDatabases = async () => {
    try {
      const data = await api.getDatabases();
      setDatabases(data.available_databases);
      setSelectedDatabase(data.current_database);
    } catch (err) {
      console.error('Failed to fetch databases:', err);
    }
  };

  const handleSelectDatabase = async (dbName: string) => {
    try {
      await api.selectDatabase(dbName);
      setSelectedDatabase(dbName);
      await fetchRuns();
    } catch (err) {
      console.error('Failed to select database:', err);
    }
  };

  useEffect(() => {
    fetchDatabases();
    fetchRuns();
  }, []);

  useEffect(() => {
    if (selectedRunId) {
      fetchRunData(selectedRunId);
    } else {
      setMetrics({});
    }
  }, [selectedRunId]);

  const handleRunSelect = (runId: number) => {
    setSelectedRunId(runId);
  };

  return (
    <div style={{ display: 'flex', height: '100vh', overflow: 'hidden' }}>
      <Sidebar 
        runs={runs} 
        selectedRunId={selectedRunId} 
        onSelectRun={handleRunSelect} 
        isOpen={isSidebarOpen}
        databases={databases}
        selectedDatabase={selectedDatabase}
        onSelectDatabase={handleSelectDatabase}
      />
      
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        {/* Header */}
        <header style={{ 
          padding: '16px', 
          borderBottom: '3px solid #1B1E20', 
          display: 'flex', 
          justifyContent: 'space-between', 
          alignItems: 'center',
          background: 'white',
          zIndex: 5
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <button 
              className="nb-button ghost" 
              onClick={() => setIsSidebarOpen(!isSidebarOpen)}
              title="Toggle Sidebar"
            >
              <Layout size={20} />
            </button>
            
            <div style={{ display: 'flex', gap: '12px' }}>
              <button className="nb-button secondary" onClick={fetchRuns} disabled={loading}>
                <RefreshCw size={16} className={loading ? 'spin' : ''} /> 
                <span className="button-text">{loading ? 'Loading...' : 'Refresh'}</span>
              </button>
              <button className="nb-button" onClick={() => setIsSettingsOpen(true)}>
                <Settings size={16} /> <span className="button-text">Config</span>
              </button>
            </div>
          </div>
          
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
            <img 
              src={simpleGradLogo} 
              alt="SimpleGrad" 
              style={{ height: '32px', width: 'auto' }}
              className="logo-img"
            />
          </div>
        </header>

        {/* Main Content */}
        <main style={{ flex: 1, overflowY: 'auto', padding: '24px', background: '#FAFAFA' }}>
          {error && (
            <div className="nb-box" style={{ padding: '16px', background: '#F7733C', color: 'white', marginBottom: '24px' }}>
              <strong>Error:</strong> {error}
            </div>
          )}

          {!selectedRunId ? (
            <div style={{ 
              height: '100%', 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'center', 
              flexDirection: 'column',
              color: '#666'
            }}>
              <div style={{ fontSize: '4rem', marginBottom: '16px', opacity: 0.2 }}>
                <Layout />
              </div>
              <p>Select a run from the sidebar to view metrics.</p>
            </div>
          ) : (
            <div>
              <div style={{ marginBottom: '24px' }}>
                <h2 style={{ margin: 0, fontSize: '2rem' }}>
                  Run #{selectedRunId}
                </h2>
                <p style={{ color: '#666', marginTop: '4px' }}>
                  {runs.find(r => r.run_id === selectedRunId)?.name}
                </p>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '24px' }}>
                {Object.entries(metrics).map(([name, data]) => (
                  <Graph 
                    key={name} 
                    metricName={name} 
                    data={data} 
                    color={['#528AC5', '#F7733C', '#3EC291', '#FFC515', '#F7B7CF'][Math.abs(name.hashCode() || 0) % 5]}
                  />
                ))}
              </div>
              
              {Object.keys(metrics).length === 0 && !loading && (
                <div className="nb-box" style={{ padding: '32px', textAlign: 'center', color: '#666' }}>
                  No metrics recorded for this run yet.
                </div>
              )}
            </div>
          )}
        </main>
      </div>

      <SettingsModal 
        isOpen={isSettingsOpen} 
        onClose={() => setIsSettingsOpen(false)} 
        onSave={() => {
          fetchRuns();
          if (selectedRunId) fetchRunData(selectedRunId);
        }}
      />
      
      <style>{`
        .spin { animation: spin 1s linear infinite; }
        @keyframes spin { 100% { transform: rotate(360deg); } }
        
        @media (max-width: 768px) {
          .button-text { display: none; }
          .logo-img { display: none; }
        }
      `}</style>
    </div>
  );
}

// Simple hash function for color generation
declare global {
  interface String {
    hashCode(): number;
  }
}

String.prototype.hashCode = function() {
  var hash = 0, i, chr;
  if (this.length === 0) return hash;
  for (i = 0; i < this.length; i++) {
    chr   = this.charCodeAt(i);
    hash  = ((hash << 5) - hash) + chr;
    hash |= 0; // Convert to 32bit integer
  }
  return hash;
};

export default App;
