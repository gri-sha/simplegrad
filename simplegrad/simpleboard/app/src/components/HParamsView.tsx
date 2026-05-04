import { useState, useMemo } from 'react';
import type { RunInfo, RecordInfo } from '../types';
import { colorForRun } from '../colors';
import { ArrowDown, ArrowUp, ArrowUpDown } from 'lucide-react';

interface HParamsViewProps {
  selectedRunIds: number[];
  runNames: Record<number, string>;
  runs: RunInfo[];
  metricsByRun: Record<number, Record<string, RecordInfo[]>>;
}

export function HParamsView({ selectedRunIds, runNames, runs, metricsByRun }: HParamsViewProps) {
  const [sortCol, setSortCol] = useState<{ type: 'name' | 'config' | 'metric', key: string } | null>(null);
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('asc');

  const configKeys = useMemo(() => {
    const keys = new Set<string>();
    for (const id of selectedRunIds) {
      const r = runs.find(x => x.run_id === id);
      if (r?.config) Object.keys(r.config).forEach(k => keys.add(k));
    }
    return Array.from(keys).sort();
  }, [selectedRunIds, runs]);

  const metricKeys = useMemo(() => {
    const keys = new Set<string>();
    for (const id of selectedRunIds) {
      const m = metricsByRun[id];
      if (m) Object.keys(m).forEach(k => keys.add(k));
    }
    return Array.from(keys).sort();
  }, [selectedRunIds, metricsByRun]);

  const rows = useMemo(() => {
    return selectedRunIds.map(id => {
      const r = runs.find(x => x.run_id === id);
      const m = metricsByRun[id] || {};
      const config = r?.config || {};
      const finalMetrics: Record<string, number> = {};
      for (const k of metricKeys) {
        if (m[k] && m[k].length > 0) {
          finalMetrics[k] = m[k][m[k].length - 1].value;
        }
      }
      return {
        id,
        name: runNames[id] || `Run #${id}`,
        config,
        metrics: finalMetrics
      };
    });
  }, [selectedRunIds, runs, metricsByRun, runNames, metricKeys]);

  const sortedRows = useMemo(() => {
    if (!sortCol) return rows;
    return [...rows].sort((a, b) => {
      let valA: any, valB: any;
      if (sortCol.type === 'name') {
        valA = a.name; valB = b.name;
      } else if (sortCol.type === 'config') {
        valA = a.config[sortCol.key] ?? -Infinity;
        valB = b.config[sortCol.key] ?? -Infinity;
      } else {
        valA = a.metrics[sortCol.key] ?? -Infinity;
        valB = b.metrics[sortCol.key] ?? -Infinity;
      }
      if (valA < valB) return sortDir === 'asc' ? -1 : 1;
      if (valA > valB) return sortDir === 'asc' ? 1 : -1;
      return 0;
    });
  }, [rows, sortCol, sortDir]);

  const handleSort = (type: 'name' | 'config' | 'metric', key: string) => {
    if (sortCol?.type === type && sortCol.key === key) {
      if (sortDir === 'asc') setSortDir('desc');
      else setSortCol(null); // toggle off
    } else {
      setSortCol({ type, key });
      setSortDir('asc');
    }
  };

  const renderSortIcon = (type: 'name' | 'config' | 'metric', key: string) => {
    if (sortCol?.type !== type || sortCol.key !== key) return <ArrowUpDown size={12} className="sort-icon inactive" />;
    return sortDir === 'asc' ? <ArrowUp size={12} className="sort-icon" /> : <ArrowDown size={12} className="sort-icon" />;
  };

  // Per-metric extremes so we can highlight the best and worst values in each column.
  // We don't know whether a metric is "higher is better" so we mark both — green = max, orange = min.
  const metricExtremes = useMemo(() => {
    const out: Record<string, { min: number; max: number }> = {};
    for (const k of metricKeys) {
      let min = Infinity;
      let max = -Infinity;
      let count = 0;
      for (const row of rows) {
        const v = row.metrics[k];
        if (typeof v === 'number' && Number.isFinite(v)) {
          if (v < min) min = v;
          if (v > max) max = v;
          count += 1;
        }
      }
      if (count >= 2 && min !== max) out[k] = { min, max };
    }
    return out;
  }, [rows, metricKeys]);

  return (
    <div className="hparams-container" style={{ overflow: 'auto', width: '100%', height: '100%', padding: '0 4px' }}>
      <table className="hparams-table" style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.875rem' }}>
        <thead style={{ position: 'sticky', top: 0, backgroundColor: 'var(--color-bg)', zIndex: 1, borderBottom: '1px solid var(--border)' }}>
          <tr>
            <th onClick={() => handleSort('name', 'name')} style={{ cursor: 'pointer', textAlign: 'left', padding: '8px', borderRight: '1px solid var(--border)', whiteSpace: 'nowrap' }}>
              Run Name {renderSortIcon('name', 'name')}
            </th>
            {configKeys.map(k => (
              <th key={`cfg-${k}`} onClick={() => handleSort('config', k)} style={{ cursor: 'pointer', textAlign: 'left', padding: '8px', color: 'var(--color-blue)', whiteSpace: 'nowrap' }}>
                {k} {renderSortIcon('config', k)}
              </th>
            ))}
            {metricKeys.map(k => (
              <th key={`met-${k}`} onClick={() => handleSort('metric', k)} style={{ cursor: 'pointer', textAlign: 'left', padding: '8px', color: 'var(--color-green)', whiteSpace: 'nowrap', borderLeft: '1px solid var(--border)' }}>
                {k} {renderSortIcon('metric', k)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sortedRows.map(row => (
            <tr key={row.id} style={{ borderBottom: '1px solid var(--border)' }}>
              <td style={{ padding: '8px', borderRight: '1px solid var(--border)', fontWeight: 600 }}>
                <span className="run-item-swatch" style={{ display: 'inline-block', width: 10, height: 10, borderRadius: '50%', backgroundColor: colorForRun(row.id), marginRight: 6 }} />
                {row.name}
              </td>
              {configKeys.map(k => (
                <td key={`cfg-${row.id}-${k}`} style={{ padding: '8px', fontFamily: 'var(--font-mono)', fontSize: '0.8rem', color: 'var(--color-fg)' }}>
                  {row.config[k] !== undefined ? String(row.config[k]) : '-'}
                </td>
              ))}
              {metricKeys.map(k => {
                const v = row.metrics[k];
                const ext = metricExtremes[k];
                let extra: { color?: string; fontWeight?: number } = {};
                if (typeof v === 'number' && ext) {
                  if (v === ext.max) extra = { color: 'var(--color-green)', fontWeight: 700 };
                  else if (v === ext.min) extra = { color: 'var(--color-orange)', fontWeight: 700 };
                }
                return (
                  <td
                    key={`met-${row.id}-${k}`}
                    style={{
                      padding: '8px',
                      fontFamily: 'var(--font-mono)',
                      fontSize: '0.8rem',
                      color: 'var(--color-fg)',
                      borderLeft: '1px solid var(--border)',
                      ...extra,
                    }}
                  >
                    {v !== undefined ? v.toFixed(4) : '-'}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
