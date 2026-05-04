/**
 * Main content area component.
 *
 * Renders the metrics grid (one chart per metric, overlaying every selected run)
 * or the graphs view depending on the active tab. Hosts the scalar toolbar
 * controlling smoothing / x-axis / y-scale / filter.
 */

import { useMemo, useState, useEffect } from 'react';
import { MetricGraph } from './MetricGraph';
import { CompGraph } from './CompGraph';
import { ScalarToolbar } from './ScalarToolbar';
import { HParamsView } from './HParamsView';
import { RunDiffView } from './RunDiffView';
import { HistogramView } from './HistogramView';
import { ImageView } from './ImageView';
import type { RecordInfo, CompGraphData, MetricSeries, XAxisMode, YScaleMode, RunInfo, RunMeta, HistogramInfo, ImageInfo } from '../types';
import { BarChart3, GitBranch, AlertCircle, Image as ImageIcon, BarChart } from 'lucide-react';
import { colorForRun } from '../colors';

interface GraphInfo {
  id: number;
  graph: CompGraphData;
  created_at: number;
}

interface MainContentProps {
  selectedRunIds: number[];
  runNames: Record<number, string>;
  rawRunNames?: Record<number, string>;
  runMeta?: Record<number, RunMeta>;
  metricsByRun: Record<number, Record<string, RecordInfo[]>>;
  graphsByRun: Record<number, GraphInfo[]>;
  histogramsByRun: Record<number, Record<string, HistogramInfo[]>>;
  imagesByRun: Record<number, Record<string, ImageInfo[]>>;
  isLoading: boolean;
  error: string | null;
  activeTab: 'metrics' | 'graphs' | 'hparams' | 'diff' | 'histograms' | 'images';
  onTabChange: (tab: 'metrics' | 'graphs' | 'hparams' | 'diff' | 'histograms' | 'images') => void;
  runs: RunInfo[];
  theme?: 'light' | 'dark';
}

export function MainContent({
  selectedRunIds,
  runNames,
  metricsByRun,
  graphsByRun,
  histogramsByRun,
  imagesByRun,
  isLoading,
  error,
  activeTab,
  onTabChange,
  runs,
  theme,
}: MainContentProps) {
  const [smoothing, setSmoothing] = useState(() => {
    const saved = localStorage.getItem('sb_smoothing');
    return saved !== null ? Number(saved) : 0;
  });
  const [xAxisMode, setXAxisMode] = useState<XAxisMode>(() => {
    return (localStorage.getItem('sb_xAxisMode') as XAxisMode) || 'step';
  });
  const [yScale, setYScale] = useState<YScaleMode>(() => {
    return (localStorage.getItem('sb_yScale') as YScaleMode) || 'linear';
  });
  const [filter, setFilter] = useState(() => {
    return localStorage.getItem('sb_filter') || '';
  });
  const [ignoreOutliers, setIgnoreOutliers] = useState(() => {
    return localStorage.getItem('sb_ignoreOutliers') === 'true';
  });

  // Persist toolbar settings
  useEffect(() => { localStorage.setItem('sb_smoothing', String(smoothing)); }, [smoothing]);
  useEffect(() => { localStorage.setItem('sb_xAxisMode', xAxisMode); }, [xAxisMode]);
  useEffect(() => { localStorage.setItem('sb_yScale', yScale); }, [yScale]);
  useEffect(() => { localStorage.setItem('sb_filter', filter); }, [filter]);
  useEffect(() => { localStorage.setItem('sb_ignoreOutliers', String(ignoreOutliers)); }, [ignoreOutliers]);

  // Union of metric names across the currently selected runs.
  const allMetricNames = useMemo(() => {
    const set = new Set<string>();
    for (const id of selectedRunIds) {
      const m = metricsByRun[id];
      if (m) for (const name of Object.keys(m)) set.add(name);
    }
    return [...set].sort();
  }, [selectedRunIds, metricsByRun]);

  const filterRegex = useMemo(() => {
    if (!filter.trim()) return null;
    try {
      return new RegExp(filter, 'i');
    } catch {
      return null;
    }
  }, [filter]);

  const visibleMetricNames = useMemo(() => {
    if (!filterRegex) return allMetricNames;
    return allMetricNames.filter((n) => filterRegex.test(n));
  }, [allMetricNames, filterRegex]);

  // Build per-metric series (one entry per run that has data for that metric).
  const seriesByMetric = useMemo(() => {
    const out: Record<string, MetricSeries[]> = {};
    for (const name of visibleMetricNames) {
      const series: MetricSeries[] = [];
      for (const runId of selectedRunIds) {
        const data = metricsByRun[runId]?.[name];
        if (!data || data.length === 0) continue;
        series.push({
          runId,
          runName: runNames[runId] || `Run #${runId}`,
          color: colorForRun(runId),
          data,
        });
      }
      out[name] = series;
    }
    return out;
  }, [visibleMetricNames, selectedRunIds, metricsByRun, runNames]);

  // Aggregate counts for tab badges
  const totalGraphs = useMemo(
    () => selectedRunIds.reduce((acc, id) => acc + (graphsByRun[id]?.length || 0), 0),
    [selectedRunIds, graphsByRun],
  );

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

  if (selectedRunIds.length === 0) {
    return (
      <main className="main-content">
        <div className="main-placeholder">
          <img src="/simpleboard_v2.svg" alt="simpleboard" className="main-placeholder-logo" />
          <h2>Select a Run</h2>
          <p>
            Choose one or more runs from the sidebar to view metrics and graphs. Click again to
            deselect.
          </p>
        </div>
      </main>
    );
  }

  const headerTitle =
    selectedRunIds.length === 1
      ? runNames[selectedRunIds[0]] || `Run #${selectedRunIds[0]}`
      : `${selectedRunIds.length} runs selected`;

  return (
    <main className="main-content">
      <div className="main-header">
        <div className="main-header-info">
          <h1>{headerTitle}</h1>
          {selectedRunIds.length === 1 && (
            <span className="main-header-id">ID: {selectedRunIds[0]}</span>
          )}
        </div>

        <div className="main-tabs">
          <button
            className={`main-tab ${activeTab === 'metrics' ? 'main-tab-active' : ''}`}
            onClick={() => onTabChange('metrics')}
          >
            <BarChart3 size={16} />
            Metrics
            {allMetricNames.length > 0 && (
              <span className="main-tab-count">{allMetricNames.length}</span>
            )}
          </button>
          <button
            className={`main-tab ${activeTab === 'graphs' ? 'main-tab-active' : ''}`}
            onClick={() => onTabChange('graphs')}
          >
            <GitBranch size={16} />
            Graphs
            {totalGraphs > 0 && <span className="main-tab-count">{totalGraphs}</span>}
          </button>
          <button
            className={`main-tab ${activeTab === 'hparams' ? 'main-tab-active' : ''}`}
            onClick={() => onTabChange('hparams')}
          >
            HParams
          </button>
          <button
            className={`main-tab ${activeTab === 'diff' ? 'main-tab-active' : ''}`}
            onClick={() => onTabChange('diff')}
          >
            Diff
          </button>
          <button
            className={`main-tab ${activeTab === 'histograms' ? 'main-tab-active' : ''}`}
            onClick={() => onTabChange('histograms')}
          >
            <BarChart size={16} />
            Histograms
          </button>
          <button
            className={`main-tab ${activeTab === 'images' ? 'main-tab-active' : ''}`}
            onClick={() => onTabChange('images')}
          >
            <ImageIcon size={16} />
            Images
          </button>
        </div>
      </div>

      <div className="main-body">
        {isLoading && allMetricNames.length === 0 ? (
          <div className="main-loading">
            <div className="spinner" />
            <p>Loading data...</p>
          </div>
        ) : activeTab === 'metrics' ? (
          <>
            <ScalarToolbar
              smoothing={smoothing}
              onSmoothingChange={setSmoothing}
              xAxisMode={xAxisMode}
              onXAxisChange={setXAxisMode}
              yScale={yScale}
              onYScaleChange={setYScale}
              filter={filter}
              onFilterChange={setFilter}
              ignoreOutliers={ignoreOutliers}
              onIgnoreOutliersChange={setIgnoreOutliers}
            />
            {visibleMetricNames.length > 0 ? (
              <div className="metrics-grid">
                {visibleMetricNames.map((name) => (
                  <MetricGraph
                    key={name}
                    metricName={name}
                    series={seriesByMetric[name] || []}
                    smoothing={smoothing}
                    xAxisMode={xAxisMode}
                    yScale={yScale}
                    ignoreOutliers={ignoreOutliers}
                    theme={theme}
                  />
                ))}
              </div>
            ) : (
              <div className="main-empty">
                <BarChart3 size={48} strokeWidth={1} />
                <p>
                  {allMetricNames.length === 0
                    ? 'No metrics recorded for the selected run(s).'
                    : 'No metrics match this filter.'}
                </p>
              </div>
            )}
          </>
        ) : activeTab === 'graphs' ? (
          <div className="graphs-container">
            {totalGraphs > 0 ? (
              selectedRunIds.flatMap((runId) => {
                const list = graphsByRun[runId] || [];
                return list.map((g, index) => (
                  <CompGraph
                    key={`${runId}-${g.id}`}
                    data={g.graph}
                    title={`${runNames[runId] || `Run #${runId}`} — Computation Graph ${index + 1}`}
                    theme={theme}
                  />
                ));
              })
            ) : (
              <div className="main-empty">
                <GitBranch size={48} strokeWidth={1} />
                <p>No computation graphs saved for the selected run(s).</p>
              </div>
            )}
          </div>
        ) : activeTab === 'hparams' ? (
          <HParamsView
            selectedRunIds={selectedRunIds}
            runNames={runNames}
            runs={runs}
            metricsByRun={metricsByRun}
          />
        ) : activeTab === 'diff' ? (
          <RunDiffView
            selectedRunIds={selectedRunIds}
            runNames={runNames}
            runs={runs}
          />
        ) : activeTab === 'histograms' ? (
          <HistogramView
            selectedRunIds={selectedRunIds}
            runNames={runNames}
            histogramsByRun={histogramsByRun}
          />
        ) : activeTab === 'images' ? (
          <ImageView
            selectedRunIds={selectedRunIds}
            runNames={runNames}
            imagesByRun={imagesByRun}
          />
        ) : null}
      </div>
    </main>
  );
}
