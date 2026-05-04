import { useMemo, useState, useEffect, useRef } from 'react';
import type { HistogramInfo } from '../types';
import * as d3 from 'd3';
import { colorForRun } from '../colors';
import { BarChart } from 'lucide-react';

interface HistogramViewProps {
  selectedRunIds: number[];
  runNames: Record<number, string>;
  histogramsByRun: Record<number, Record<string, HistogramInfo[]>>;
}

export function HistogramView({ selectedRunIds, runNames, histogramsByRun }: HistogramViewProps) {
  const [selectedMetric, setSelectedMetric] = useState<string | null>(null);
  const [stepIndex, setStepIndex] = useState<number>(0);

  const availableMetrics = useMemo(() => {
    const keys = new Set<string>();
    for (const id of selectedRunIds) {
      const hists = histogramsByRun[id];
      if (hists) Object.keys(hists).forEach(k => keys.add(k));
    }
    return Array.from(keys).sort();
  }, [selectedRunIds, histogramsByRun]);

  useEffect(() => {
    if (availableMetrics.length > 0 && (!selectedMetric || !availableMetrics.includes(selectedMetric))) {
      setSelectedMetric(availableMetrics[0]);
    }
  }, [availableMetrics, selectedMetric]);

  const allSteps = useMemo(() => {
    if (!selectedMetric) return [];
    const steps = new Set<number>();
    for (const id of selectedRunIds) {
      const hists = histogramsByRun[id]?.[selectedMetric] || [];
      hists.forEach(h => steps.add(h.step));
    }
    return Array.from(steps).sort((a, b) => a - b);
  }, [selectedRunIds, histogramsByRun, selectedMetric]);

  useEffect(() => {
    if (allSteps.length > 0) {
      setStepIndex(allSteps.length - 1); // default to latest
    }
  }, [allSteps.length]);

  const currentStep = allSteps[stepIndex];

  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!svgRef.current || !containerRef.current || !selectedMetric || currentStep === undefined) return;
    const container = containerRef.current;
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const margin = { top: 20, right: 20, bottom: 40, left: 50 };
    const width = container.clientWidth - margin.left - margin.right;
    const height = 300 - margin.top - margin.bottom;

    svg.attr('width', container.clientWidth).attr('height', 300);

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    let xMin = Infinity;
    let xMax = -Infinity;
    let yMax = -Infinity;

    const dataToDraw: { runId: number, color: string, h: HistogramInfo }[] = [];

    for (const id of selectedRunIds) {
      const hists = histogramsByRun[id]?.[selectedMetric] || [];
      // Find the closest step <= currentStep
      let closest = hists[0];
      for (const h of hists) {
        if (h.step <= currentStep) closest = h;
        else break;
      }
      if (closest) {
        dataToDraw.push({ runId: id, color: colorForRun(id), h: closest });
        const edges = closest.bucket_edges;
        if (edges[0] < xMin) xMin = edges[0];
        if (edges[edges.length - 1] > xMax) xMax = edges[edges.length - 1];
        const mY = Math.max(...closest.bucket_counts);
        if (mY > yMax) yMax = mY;
      }
    }

    if (dataToDraw.length === 0 || !isFinite(xMin)) return;

    const x = d3.scaleLinear().domain([xMin, xMax]).range([0, width]);
    const y = d3.scaleLinear().domain([0, yMax]).range([height, 0]).nice();

    const styles = getComputedStyle(document.documentElement);
    const borderColor = styles.getPropertyValue('--border').trim() || 'rgba(21,23,24,0.11)';
    const mutedColor = styles.getPropertyValue('--muted').trim() || 'rgba(21,23,24,0.48)';

    g.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(x))
      .selectAll('text')
      .attr('fill', mutedColor);

    g.append('g')
      .call(d3.axisLeft(y).ticks(5))
      .selectAll('text')
      .attr('fill', mutedColor);

    g.selectAll('.domain').attr('stroke', borderColor);
    g.selectAll('.tick line').attr('stroke', borderColor);

    // Draw lines for histograms instead of bars to overlay nicely
    const line = (d3 as any).line()
      .x((d: any) => x(d.x))
      .y((d: any) => y(d.y))
      .curve(d3.curveStepAfter);

    for (const d of dataToDraw) {
      const pts = [];
      const edges = d.h.bucket_edges;
      const counts = d.h.bucket_counts;
      for (let i = 0; i < counts.length; i++) {
        pts.push({ x: edges[i], y: counts[i] });
      }
      pts.push({ x: edges[edges.length - 1], y: counts[counts.length - 1] }); // repeat last for step curve

      g.append('path')
        .datum(pts)
        .attr('fill', 'none')
        .attr('stroke', d.color)
        .attr('stroke-width', 2)
        .attr('opacity', 0.8)
        .attr('d', line);
      
      // Also fill with low opacity
      const area = (d3 as any).area()
        .x((p: any) => x(p.x))
        .y0(height)
        .y1((p: any) => y(p.y))
        .curve(d3.curveStepAfter);
      
      g.append('path')
        .datum(pts)
        .attr('fill', d.color)
        .attr('opacity', 0.1)
        .attr('d', area);
    }

  }, [selectedRunIds, histogramsByRun, selectedMetric, currentStep]);

  if (selectedRunIds.length === 0 || availableMetrics.length === 0) {
    return (
      <div className="main-empty">
        <BarChart size={48} strokeWidth={1} />
        <p>No histograms available for the selected run(s).</p>
      </div>
    );
  }

  return (
    <div className="hparams-container" style={{ padding: '1rem', width: '100%', maxWidth: '800px', margin: '0 auto' }}>
      <div style={{ display: 'flex', gap: '1rem', marginBottom: '1rem', alignItems: 'center' }}>
        <select 
          value={selectedMetric || ''} 
          onChange={e => setSelectedMetric(e.target.value)}
          style={{ padding: '6px', borderRadius: '4px', border: '1px solid var(--border)' }}
        >
          {availableMetrics.map(m => <option key={m} value={m}>{m}</option>)}
        </select>
      </div>

      {allSteps.length > 0 && (
        <div style={{ marginBottom: '2rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px', fontSize: '0.875rem', color: 'var(--muted)' }}>
            <span>Step {allSteps[0]}</span>
            <span style={{ fontWeight: 600, color: 'var(--color-fg)' }}>Current Step: {currentStep}</span>
            <span>Step {allSteps[allSteps.length - 1]}</span>
          </div>
          <input 
            type="range" 
            min={0} 
            max={allSteps.length - 1} 
            value={stepIndex} 
            onChange={e => setStepIndex(Number(e.target.value))}
            style={{ width: '100%' }}
          />
        </div>
      )}

      <div ref={containerRef} style={{ width: '100%', height: '300px', border: '1px solid var(--border)', borderRadius: '8px', background: 'var(--surface)' }}>
        <svg ref={svgRef} />
      </div>

      <div className="metric-graph-legend" style={{ marginTop: '1rem', justifyContent: 'center' }}>
        {selectedRunIds.map(id => (
          <span key={id} className="metric-graph-legend-item">
            <span className="metric-graph-legend-swatch" style={{ background: colorForRun(id) }} />
            {runNames[id]}
          </span>
        ))}
      </div>
    </div>
  );
}
