import { useState, useMemo, useEffect } from 'react';
import type { RunInfo } from '../types';
import { colorForRun } from '../colors';
import { GitCompare } from 'lucide-react';

interface RunDiffViewProps {
  selectedRunIds: number[];
  runNames: Record<number, string>;
  runs: RunInfo[];
}

export function RunDiffView({ selectedRunIds, runNames, runs }: RunDiffViewProps) {
  const [leftId, setLeftId] = useState<number | null>(null);
  const [rightId, setRightId] = useState<number | null>(null);

  useEffect(() => {
    setLeftId((prev) =>
      prev !== null && selectedRunIds.includes(prev) ? prev : (selectedRunIds[0] ?? null),
    );
    setRightId((prev) => {
      if (selectedRunIds.length < 2) return null;
      const last = selectedRunIds[selectedRunIds.length - 1];
      return prev !== null && selectedRunIds.includes(prev) ? prev : last;
    });
  }, [selectedRunIds]);

  const diffs = useMemo(() => {
    if (!leftId || !rightId) return [];
    const r1 = runs.find(x => x.run_id === leftId);
    const r2 = runs.find(x => x.run_id === rightId);
    if (!r1 || !r2) return [];

    const c1 = r1.config || {};
    const c2 = r2.config || {};

    const keys = Array.from(new Set([...Object.keys(c1), ...Object.keys(c2)])).sort();
    return keys.map(k => ({
      key: k,
      val1: c1[k],
      val2: c2[k],
      isDiff: c1[k] !== c2[k]
    }));
  }, [leftId, rightId, runs]);

  if (selectedRunIds.length < 2) {
    return (
      <div className="main-empty">
        <GitCompare size={48} strokeWidth={1} />
        <p>Select at least 2 runs to compare their configurations.</p>
      </div>
    );
  }

  return (
    <div className="run-diff-container" style={{ padding: '0 1rem' }}>
      <div style={{ display: 'flex', gap: '1rem', marginBottom: '1rem', alignItems: 'center' }}>
        <select value={leftId || ''} onChange={e => setLeftId(Number(e.target.value))} style={{ padding: '4px', borderRadius: '4px', border: '1px solid var(--border)' }}>
          {selectedRunIds.map(id => (
            <option key={id} value={id}>{runNames[id]} (ID: {id})</option>
          ))}
        </select>
        <GitCompare size={16} color="var(--muted)" />
        <select value={rightId || ''} onChange={e => setRightId(Number(e.target.value))} style={{ padding: '4px', borderRadius: '4px', border: '1px solid var(--border)' }}>
          {selectedRunIds.map(id => (
            <option key={id} value={id}>{runNames[id]} (ID: {id})</option>
          ))}
        </select>
      </div>

      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.875rem' }}>
        <thead style={{ borderBottom: '1px solid var(--border)' }}>
          <tr>
            <th style={{ textAlign: 'left', padding: '8px', width: '30%' }}>Hyperparameter</th>
            <th style={{ textAlign: 'left', padding: '8px', width: '35%' }}>
              <span className="run-item-swatch" style={{ display: 'inline-block', width: 10, height: 10, borderRadius: '50%', backgroundColor: leftId ? colorForRun(leftId) : 'transparent', marginRight: 6 }} />
              {leftId ? runNames[leftId] : '-'}
            </th>
            <th style={{ textAlign: 'left', padding: '8px', width: '35%' }}>
              <span className="run-item-swatch" style={{ display: 'inline-block', width: 10, height: 10, borderRadius: '50%', backgroundColor: rightId ? colorForRun(rightId) : 'transparent', marginRight: 6 }} />
              {rightId ? runNames[rightId] : '-'}
            </th>
          </tr>
        </thead>
        <tbody>
          {diffs.map(d => (
            <tr key={d.key} style={{ borderBottom: '1px solid var(--border)', backgroundColor: d.isDiff ? 'var(--surface-hover)' : 'transparent' }}>
              <td style={{ padding: '8px', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{d.key}</td>
              <td style={{ padding: '8px', fontFamily: 'var(--font-mono)', color: d.isDiff ? 'var(--color-orange)' : 'var(--color-fg)' }}>
                {d.val1 !== undefined ? String(d.val1) : '-'}
              </td>
              <td style={{ padding: '8px', fontFamily: 'var(--font-mono)', color: d.isDiff ? 'var(--color-orange)' : 'var(--color-fg)' }}>
                {d.val2 !== undefined ? String(d.val2) : '-'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
