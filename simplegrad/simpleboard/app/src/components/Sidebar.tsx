/**
 * Sidebar component showing runs list
 */

import {
  ChevronLeft,
  ChevronRight,
  Play,
  CheckCircle,
  XCircle,
  Clock,
  Database,
} from 'lucide-react';
import type { RunInfo } from '../types';

interface SidebarProps {
  runs: RunInfo[];
  selectedRunId: number | null;
  onSelectRun: (runId: number) => void;
  isOpen: boolean;
  onToggle: () => void;
  databases: string[];
  currentDatabase: string | null;
  onSelectDatabase: (dbName: string) => void;
}

export function Sidebar({
  runs,
  selectedRunId,
  onSelectRun,
  isOpen,
  onToggle,
  databases,
  currentDatabase,
  onSelectDatabase,
}: SidebarProps) {
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle size={14} className="status-icon status-completed" />;
      case 'failed':
        return <XCircle size={14} className="status-icon status-failed" />;
      case 'running':
        return <Play size={14} className="status-icon status-running" />;
      default:
        return null;
    }
  };

  const hasDatabase = currentDatabase !== null;
  const hasRuns = runs.length > 0;

  return (
    <>
      <aside className={`sidebar ${isOpen ? 'sidebar-open' : 'sidebar-closed'}`}>
        <div className="sidebar-header">
          <div className="sidebar-db-selector">
            <Database size={16} />
            <select
              className="sidebar-select"
              value={currentDatabase || ''}
              onChange={(e) => onSelectDatabase(e.target.value)}
              disabled={databases.length === 0}
            >
              <option value="" disabled>
                {databases.length === 0 ? 'No experiments' : 'Select experiment'}
              </option>
              {databases.map((db) => (
                <option key={db} value={db}>
                  {db.replace('.db', '')}
                </option>
              ))}
            </select>
          </div>
          <button className="sidebar-toggle" onClick={onToggle}>
            <ChevronLeft size={18} />
          </button>
        </div>

        <div className="sidebar-content">
          {!hasDatabase || !hasRuns ? (
            <div className="sidebar-empty-centered">
              <Database size={32} strokeWidth={1.5} />
              <p>
                {!hasDatabase
                  ? 'Select an experiment to view runs'
                  : 'No runs available in this experiment'}
              </p>
            </div>
          ) : (
            <ul className="runs-list">
              {runs.map((run) => (
                <li
                  key={run.run_id}
                  className={`run-item ${selectedRunId === run.run_id ? 'run-item-selected' : ''}`}
                  onClick={() => onSelectRun(run.run_id)}
                >
                  <div className="run-item-header">
                    {getStatusIcon(run.status)}
                    <span className="run-item-name">{run.name}</span>
                  </div>
                  <div className="run-item-meta">
                    <Clock size={12} />
                    <span>{run.created_at}</span>
                  </div>
                  {run.metrics && run.metrics.length > 0 && (
                    <div className="run-item-metrics">
                      {run.metrics.slice(0, 3).map((m) => (
                        <span key={m} className="run-item-metric-tag">
                          {`'${m}' `}
                        </span>
                      ))}
                      {run.metrics.length > 3 && (
                        <span className="run-item-metric-tag">+{run.metrics.length - 3}</span>
                      )}
                    </div>
                  )}
                </li>
              ))}
            </ul>
          )}
        </div>
      </aside>

      {/* Collapsed toggle button */}
      {!isOpen && (
        <button className="sidebar-expand" onClick={onToggle}>
          <ChevronRight size={18} />
        </button>
      )}
    </>
  );
}
