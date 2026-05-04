/**
 * Sidebar component showing runs list with multi-select.
 */

import {
  ChevronLeft,
  ChevronRight,
  Play,
  CheckCircle,
  XCircle,
  Clock,
  Database,
  Pin,
  EyeOff,
  Eye,
  Star,
  Edit2
} from 'lucide-react';
import type { RunInfo, RunMeta } from '../types';
import { colorForRun } from '../colors';
import { useState } from 'react';

interface SidebarProps {
  runs: RunInfo[];
  selectedRunIds: number[];
  onToggleRun: (runId: number) => void;
  onClearRuns: () => void;
  isOpen: boolean;
  onToggle: () => void;
  databases: string[];
  currentDatabase: string | null;
  onSelectDatabase: (dbName: string) => void;
  runMeta: Record<number, RunMeta>;
  updateRunMeta: (runId: number, meta: Partial<RunMeta>) => void;
}

export function Sidebar({
  runs,
  selectedRunIds,
  onToggleRun,
  onClearRuns,
  isOpen,
  onToggle,
  databases,
  currentDatabase,
  onSelectDatabase,
  runMeta,
  updateRunMeta,
}: SidebarProps) {
  const [showHidden, setShowHidden] = useState(false);
  const [editingRunId, setEditingRunId] = useState<number | null>(null);
  const [editName, setEditName] = useState('');

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
  
  // Filter and sort runs
  const visibleRuns = runs
    .filter((r) => showHidden || !runMeta[r.run_id]?.hidden)
    .sort((a, b) => {
      const aPin = runMeta[a.run_id]?.pinned ? 1 : 0;
      const bPin = runMeta[b.run_id]?.pinned ? 1 : 0;
      if (aPin !== bPin) return bPin - aPin;
      // then sort by id descending
      return b.run_id - a.run_id;
    });

  const hasRuns = visibleRuns.length > 0 || runs.length > 0;
  const selected = new Set(selectedRunIds);

  const startEdit = (e: React.MouseEvent, runId: number, currentName: string) => {
    e.stopPropagation();
    setEditingRunId(runId);
    setEditName(runMeta[runId]?.rename || currentName);
  };

  const saveEdit = (e: React.FormEvent | React.FocusEvent, runId: number) => {
    e.preventDefault();
    e.stopPropagation();
    updateRunMeta(runId, { rename: editName.trim() || undefined });
    setEditingRunId(null);
  };

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
            <>
              <div className="sidebar-section-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span className="sidebar-section-label">
                  Runs {selected.size > 0 && <span className="sidebar-section-count">({selected.size} selected)</span>}
                </span>
                <div style={{ display: 'flex', gap: '4px' }}>
                  <button 
                    className="sidebar-clear-btn" 
                    onClick={() => setShowHidden(!showHidden)}
                    title={showHidden ? "Hide hidden runs" : "Show hidden runs"}
                    style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--muted)', padding: '2px' }}
                  >
                    {showHidden ? <Eye size={14} /> : <EyeOff size={14} />}
                  </button>
                  {selected.size > 0 && (
                    <button className="sidebar-clear-btn" onClick={onClearRuns} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--muted)', fontSize: '0.75rem' }}>
                      Clear
                    </button>
                  )}
                </div>
              </div>
              <ul className="runs-list">
                {visibleRuns.map((run) => {
                  const meta = runMeta[run.run_id] || {};
                  const isSelected = selected.has(run.run_id);
                  const swatchColor = colorForRun(run.run_id);
                  return (
                    <li
                      key={run.run_id}
                      className={`run-item ${isSelected ? 'run-item-selected' : ''}`}
                      onClick={() => onToggleRun(run.run_id)}
                    >
                      <div className="run-item-header" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flex: 1, minWidth: 0 }}>
                          <span
                            className="run-item-swatch"
                            style={{
                              backgroundColor: isSelected ? swatchColor : 'transparent',
                              borderColor: swatchColor,
                              width: '12px', height: '12px', borderRadius: '50%', borderStyle: 'solid', borderWidth: '2px', flexShrink: 0
                            }}
                          />
                          {getStatusIcon(run.status)}
                          
                          {editingRunId === run.run_id ? (
                            <form onSubmit={(e) => saveEdit(e, run.run_id)} style={{ flex: 1, minWidth: 0 }}>
                              <input
                                type="text"
                                value={editName}
                                onChange={(e) => setEditName(e.target.value)}
                                onBlur={(e) => saveEdit(e, run.run_id)}
                                onClick={(e) => e.stopPropagation()}
                                autoFocus
                                style={{ width: '100%', fontSize: '0.9rem', padding: '0 2px' }}
                              />
                            </form>
                          ) : (
                            <span className="run-item-name" style={{ color: meta.hidden ? 'var(--muted)' : undefined }}>
                              {meta.starred && <Star size={12} fill="currentColor" style={{ marginRight: '4px', color: 'var(--color-yellow)', display: 'inline' }} />}
                              {meta.rename || run.name}
                            </span>
                          )}
                        </div>
                        <div className="run-item-actions">
                          <button onClick={(e) => { e.stopPropagation(); updateRunMeta(run.run_id, { pinned: !meta.pinned }); }} style={{ background: 'none', border: 'none', cursor: 'pointer', color: meta.pinned ? 'var(--color-fg)' : 'inherit' }}>
                            <Pin size={12} fill={meta.pinned ? 'currentColor' : 'none'} />
                          </button>
                          <button onClick={(e) => { e.stopPropagation(); updateRunMeta(run.run_id, { hidden: !meta.hidden }); }} style={{ background: 'none', border: 'none', cursor: 'pointer', color: meta.hidden ? 'var(--color-fg)' : 'inherit' }}>
                            <EyeOff size={12} />
                          </button>
                          <button onClick={(e) => { e.stopPropagation(); updateRunMeta(run.run_id, { starred: !meta.starred }); }} style={{ background: 'none', border: 'none', cursor: 'pointer', color: meta.starred ? 'var(--color-fg)' : 'inherit' }}>
                            <Star size={12} />
                          </button>
                          <button onClick={(e) => startEdit(e, run.run_id, run.name)} style={{ background: 'none', border: 'none', cursor: 'pointer' }}>
                            <Edit2 size={12} />
                          </button>
                        </div>
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
                  );
                })}
              </ul>
            </>
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
