/**
 * Main content area component
 */

import { MetricGraph } from './MetricGraph';
import { CompGraph } from './CompGraph';
import type { RecordInfo, CompGraphData } from '../types';
import { BarChart3, GitBranch, AlertCircle } from 'lucide-react';

interface MainContentProps {
  selectedRunId: number | null;
  runName: string | null;
  metrics: Record<string, RecordInfo[]>;
  graphs: Array<{ id: number; graph: CompGraphData; created_at: number }>;
  isLoading: boolean;
  error: string | null;
  activeTab: 'metrics' | 'graphs';
  onTabChange: (tab: 'metrics' | 'graphs') => void;
}

export function MainContent({
  selectedRunId,
  runName,
  metrics,
  graphs,
  isLoading,
  error,
  activeTab,
  onTabChange
}: MainContentProps) {
  const metricNames = Object.keys(metrics);
  const hasMetrics = metricNames.length > 0;
  const hasGraphs = graphs.length > 0;

  if (error) {
    return (
      <main className="main-content">
        <div className="main-error">
          <AlertCircle size={48} />
          <h2>Connection Error</h2>
          <p>{error}</p>
        </div>
      </main>
    );
  }

  if (!selectedRunId) {
    return (
      <main className="main-content">
        <div className="main-placeholder">
          <img 
            src="/simpleboard_v2.svg" 
            alt="simpleboard" 
            className="main-placeholder-logo" 
          />
          <h2>Select a Run</h2>
          <p>Choose an experiment and run from the sidebar to view metrics and graphs.</p>
        </div>
      </main>
    );
  }

  return (
    <main className="main-content">
      {/* Run header */}
      <div className="main-header">
        <div className="main-header-info">
          <h1>{runName || `Run #${selectedRunId}`}</h1>
          <span className="main-header-id">ID: {selectedRunId}</span>
        </div>

        {/* Tabs */}
        <div className="main-tabs">
          <button
            className={`main-tab ${activeTab === 'metrics' ? 'main-tab-active' : ''}`}
            onClick={() => onTabChange('metrics')}
          >
            <BarChart3 size={16} />
            Metrics
            {hasMetrics && <span className="main-tab-count">{metricNames.length}</span>}
          </button>
          <button
            className={`main-tab ${activeTab === 'graphs' ? 'main-tab-active' : ''}`}
            onClick={() => onTabChange('graphs')}
          >
            <GitBranch size={16} />
            Graphs
            {hasGraphs && <span className="main-tab-count">{graphs.length}</span>}
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="main-body">
        {isLoading ? (
          <div className="main-loading">
            <div className="spinner" />
            <p>Loading data...</p>
          </div>
        ) : activeTab === 'metrics' ? (
          <div className="metrics-grid">
            {hasMetrics ? (
              metricNames.map((name) => (
                <MetricGraph
                  key={name}
                  metricName={name}
                  data={metrics[name]}
                />
              ))
            ) : (
              <div className="main-empty">
                <BarChart3 size={48} strokeWidth={1} />
                <p>No metrics recorded for this run.</p>
              </div>
            )}
          </div>
        ) : (
          <div className="graphs-container">
            {hasGraphs ? (
              graphs.map((g, index) => (
                <CompGraph
                  key={g.id}
                  data={g.graph}
                  title={`Computation Graph ${index + 1}`}
                />
              ))
            ) : (
              <div className="main-empty">
                <GitBranch size={48} strokeWidth={1} />
                <p>No computation graphs saved for this run.</p>
              </div>
            )}
          </div>
        )}
      </div>
    </main>
  );
}
