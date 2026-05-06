/**
 * Toolbar above the metrics grid: smoothing, x-axis selector,
 * y-scale toggle, overlay toggle, metric name filter.
 */

import { Layers, LayoutList } from 'lucide-react';
import type { XAxisMode, YScaleMode } from '../types';

interface ScalarToolbarProps {
  smoothing: number;
  onSmoothingChange: (v: number) => void;
  xAxisMode: XAxisMode;
  onXAxisChange: (m: XAxisMode) => void;
  yScale: YScaleMode;
  onYScaleChange: (m: YScaleMode) => void;
  filter: string;
  onFilterChange: (v: string) => void;
  ignoreOutliers: boolean;
  onIgnoreOutliersChange: (v: boolean) => void;
  overlayMode: boolean;
  onOverlayModeChange: (v: boolean) => void;
}

export function ScalarToolbar({
  smoothing,
  onSmoothingChange,
  xAxisMode,
  onXAxisChange,
  yScale,
  onYScaleChange,
  filter,
  onFilterChange,
  ignoreOutliers,
  onIgnoreOutliersChange,
  overlayMode,
  onOverlayModeChange,
}: ScalarToolbarProps) {
  return (
    <div className="scalar-toolbar">

      {/* Smoothing */}
      <div className="scalar-toolbar-group">
        <label className="scalar-toolbar-label">Smoothing</label>
        <input
          type="range"
          min={0}
          max={0.99}
          step={0.01}
          value={smoothing}
          onChange={(e) => onSmoothingChange(Number(e.target.value))}
          className="scalar-toolbar-slider"
        />
        <span className="scalar-toolbar-readout">{smoothing.toFixed(2)}</span>
      </div>

      {/* X-axis */}
      <div className="scalar-toolbar-group">
        <label className="scalar-toolbar-label">X-axis</label>
        <div className="scalar-toolbar-segments">
          {(['step', 'relative', 'wall'] as XAxisMode[]).map((m) => (
            <button
              key={m}
              className={`scalar-toolbar-segment ${xAxisMode === m ? 'active' : ''}`}
              onClick={() => onXAxisChange(m)}
            >
              {m === 'step' ? 'Step' : m === 'relative' ? 'Relative' : 'Wall'}
            </button>
          ))}
        </div>
      </div>

      {/* Y-scale */}
      <div className="scalar-toolbar-group">
        <label className="scalar-toolbar-label">Y-scale</label>
        <div className="scalar-toolbar-segments">
          {(['linear', 'log'] as YScaleMode[]).map((m) => (
            <button
              key={m}
              className={`scalar-toolbar-segment ${yScale === m ? 'active' : ''}`}
              onClick={() => onYScaleChange(m)}
            >
              {m === 'linear' ? 'Linear' : 'Log'}
            </button>
          ))}
        </div>
      </div>

      {/* Overlay / separate toggle */}
      <div className="scalar-toolbar-group">
        <label className="scalar-toolbar-label">Runs</label>
        <div className="scalar-toolbar-segments">
          <button
            className={`scalar-toolbar-segment ${overlayMode ? 'active' : ''}`}
            onClick={() => onOverlayModeChange(true)}
            title="Overlay all runs on the same chart"
          >
            <Layers size={13} style={{ marginRight: '4px', verticalAlign: 'middle' }} />
            Overlay
          </button>
          <button
            className={`scalar-toolbar-segment ${!overlayMode ? 'active' : ''}`}
            onClick={() => onOverlayModeChange(false)}
            title="Show each run on a separate chart"
          >
            <LayoutList size={13} style={{ marginRight: '4px', verticalAlign: 'middle' }} />
            Separate
          </button>
        </div>
      </div>

      {/* Filter */}
      <div className="scalar-toolbar-group scalar-toolbar-filter-group">
        <label className="scalar-toolbar-label">Filter</label>
        <input
          type="text"
          value={filter}
          placeholder="metric name (regex)"
          onChange={(e) => onFilterChange(e.target.value)}
          className="scalar-toolbar-filter"
        />
      </div>

      {/* Ignore outliers */}
      <div className="scalar-toolbar-group">
        <label className="scalar-toolbar-label" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
          <input
            type="checkbox"
            checked={ignoreOutliers}
            onChange={(e) => onIgnoreOutliersChange(e.target.checked)}
          />
          Ignore outliers
        </label>
      </div>

    </div>
  );
}
