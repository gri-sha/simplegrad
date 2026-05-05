/**
 * D3.js metric chart that overlays one line per selected run.
 *
 * Features: smoothing (raw faded + smoothed bold), x-axis mode (step / relative
 * time / wall time), y-scale (linear / log), zoomable x-axis, hover crosshair
 * with per-run readouts, CSV export.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';
import { Download, RotateCcw } from 'lucide-react';
import type { MetricSeries, RecordInfo, XAxisMode, YScaleMode } from '../types';
import { smoothEMA } from '../smoothing';

interface MetricGraphProps {
  metricName: string;
  series: MetricSeries[];
  smoothing: number;
  xAxisMode: XAxisMode;
  yScale: YScaleMode;
  ignoreOutliers?: boolean;
}

interface PreparedPoint {
  x: number;
  raw: number;
  smoothed: number;
  step: number;
  log_time: number;
}

interface PreparedSeries {
  runId: number;
  runName: string;
  color: string;
  points: PreparedPoint[];
}

// log_time may be either seconds-since-epoch or millis-since-epoch; normalize.
function logTimeToMillis(t: number): number {
  return t > 1e12 ? t : t * 1000;
}

function xAccessor(mode: XAxisMode, baseLogTime: number): (r: RecordInfo) => number {
  if (mode === 'step') return (r) => r.step;
  if (mode === 'wall') return (r) => logTimeToMillis(r.log_time);
  // relative seconds since the first sample for this run
  return (r) => r.log_time - baseLogTime;
}

function prepareSeries(
  series: MetricSeries[],
  smoothing: number,
  xAxisMode: XAxisMode,
  yScale: YScaleMode,
): PreparedSeries[] {
  return series
    .filter((s) => s.data && s.data.length > 0)
    .map((s) => {
      const sorted = [...s.data].sort((a, b) => a.step - b.step);
      const base = sorted[0]?.log_time ?? 0;
      const xs = xAccessor(xAxisMode, base);
      const raw = sorted.map((r) => r.value);
      const smoothed = smoothEMA(raw, smoothing);

      const points: PreparedPoint[] = [];
      for (let i = 0; i < sorted.length; i++) {
        const r = sorted[i];
        const v = raw[i];
        if (yScale === 'log' && !(v > 0)) continue;
        points.push({
          x: xs(r),
          raw: v,
          smoothed: smoothed[i],
          step: r.step,
          log_time: r.log_time,
        });
      }
      return { runId: s.runId, runName: s.runName, color: s.color, points };
    })
    .filter((s) => s.points.length > 0);
}

function formatTick(mode: XAxisMode): (v: number) => string {
  if (mode === 'wall') {
    const fmt = d3.timeFormat('%H:%M:%S');
    return (v: number) => fmt(new Date(v));
  }
  if (mode === 'relative') {
    return (v: number) => {
      if (v < 60) return `${v.toFixed(0)}s`;
      if (v < 3600) return `${(v / 60).toFixed(1)}m`;
      return `${(v / 3600).toFixed(2)}h`;
    };
  }
  return d3.format('~s');
}

export function MetricGraph({
  metricName,
  series,
  smoothing,
  xAxisMode,
  yScale,
  ignoreOutliers,
}: MetricGraphProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const zoomRef = useRef<any>(null);
  const [sizeTick, setSizeTick] = useState(0);

  const prepared = useMemo(
    () => prepareSeries(series, smoothing, xAxisMode, yScale),
    [series, smoothing, xAxisMode, yScale],
  );

  // Count points dropped by log-scale so we can warn.
  const droppedForLog = useMemo(() => {
    if (yScale !== 'log') return 0;
    let dropped = 0;
    for (const s of series) for (const r of s.data) if (!(r.value > 0)) dropped += 1;
    return dropped;
  }, [series, yScale]);

  useEffect(() => {
    if (!svgRef.current || !containerRef.current) return;
    const container = containerRef.current;
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const margin = { top: 12, right: 16, bottom: 36, left: 52 };
    const headerHeight = 44;
    const legendHeight = prepared.length > 0 ? 22 : 0;
    const availableHeight = Math.max(120, container.clientHeight - headerHeight - legendHeight);
    const width = Math.max(40, container.clientWidth - margin.left - margin.right);
    const height = Math.max(40, availableHeight - margin.top - margin.bottom);

    svg.attr('width', container.clientWidth).attr('height', availableHeight);

    if (prepared.length === 0) return;

    // Capture pointer events for panning + hover
    const overlay = svg
      .append('rect')
      .attr('width', '100%')
      .attr('height', '100%')
      .attr('fill', 'transparent')
      .style('cursor', 'crosshair');

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const clipId = `clip-${metricName.replace(/[^a-zA-Z0-9]/g, '-')}`;
    svg
      .append('defs')
      .append('clipPath')
      .attr('id', clipId)
      .append('rect')
      .attr('width', width)
      .attr('height', height);

    // Domain across all series
    let xMin = Infinity;
    let xMax = -Infinity;
    let yMin = Infinity;
    let yMax = -Infinity;
    const allYValues: number[] = [];

    for (const s of prepared) {
      for (const p of s.points) {
        if (p.x < xMin) xMin = p.x;
        if (p.x > xMax) xMax = p.x;
        const v = smoothing > 0 ? p.smoothed : p.raw;
        if (v < yMin) yMin = v;
        if (v > yMax) yMax = v;
        if (p.raw < yMin) yMin = p.raw;
        if (p.raw > yMax) yMax = p.raw;
        
        if (ignoreOutliers) {
          allYValues.push(p.raw);
          if (smoothing > 0) allYValues.push(p.smoothed);
        }
      }
    }
    if (!isFinite(xMin) || !isFinite(yMin)) return;

    if (ignoreOutliers && allYValues.length > 20) {
      allYValues.sort((a, b) => a - b);
      // Clip to 1st and 99th percentile
      const q1 = allYValues[Math.floor(allYValues.length * 0.01)];
      const q99 = allYValues[Math.floor(allYValues.length * 0.99)];
      // Ensure we don't invert the domain
      if (q1 < q99) {
        yMin = Math.max(yMin, q1);
        yMax = Math.min(yMax, q99);
      }
    }

    const xRange = xMax - xMin;
    const xPad = xRange * 0.02 || 1;
    const yPad = yScale === 'log' ? 0 : (yMax - yMin) * 0.1 || 1;

    // Use scaleTime for wall mode so axis ticks come out as clock times.
    const x =
      xAxisMode === 'wall'
        ? d3.scaleTime().domain([new Date(xMin), new Date(xMax + xPad)]).range([0, width])
        : d3.scaleLinear().domain([xMin, xMax + xPad]).range([0, width]);

    const y =
      yScale === 'log'
        ? d3
            .scaleLog()
            .domain([Math.max(yMin, 1e-12), Math.max(yMax, yMin * 10)])
            .range([height, 0])
            .nice()
        : d3
            .scaleLinear()
            .domain([yMin - yPad, yMax + yPad])
            .range([height, 0]);

    const styles = getComputedStyle(document.documentElement);
    const borderColor = styles.getPropertyValue('--border').trim() || 'rgba(21,23,24,0.11)';
    const borderStrong = styles.getPropertyValue('--border-strong').trim() || 'rgba(21,23,24,0.35)';
    const mutedColor = styles.getPropertyValue('--muted').trim() || 'rgba(21,23,24,0.48)';

    // Grid
    g.append('g')
      .attr('class', 'grid')
      .attr('transform', `translate(0,${height})`)
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      .call((d3.axisBottom(x as any) as any).ticks(5).tickSize(-height).tickFormat(() => ''))
      .selectAll('line')
      .attr('stroke', borderColor)
      .attr('stroke-dasharray', '2,2');

    g.append('g')
      .attr('class', 'grid')
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      .call((d3.axisLeft(y as any) as any).ticks(5).tickSize(-width).tickFormat(() => ''))
      .selectAll('line')
      .attr('stroke', borderColor)
      .attr('stroke-dasharray', '2,2');

    g.selectAll('.grid .domain').remove();

    // X axis
    const xAxisFmt = xAxisMode === 'wall' ? null : formatTick(xAxisMode);
    const xAxis =
      xAxisMode === 'wall'
        ? // eslint-disable-next-line @typescript-eslint/no-explicit-any
          (d3.axisBottom(x as any).ticks(5) as any)
        : // eslint-disable-next-line @typescript-eslint/no-explicit-any
          (d3.axisBottom(x as any).ticks(5).tickFormat(xAxisFmt as any) as any);

    const xAxisGroup = g
      .append('g')
      .attr('class', 'axis axis-x')
      .attr('transform', `translate(0,${height})`)
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      .call(xAxis as any);
    xAxisGroup
      .selectAll('text')
      .attr('fill', mutedColor)
      .attr('font-family', 'var(--font-body)')
      .attr('font-size', '11px');

    // Y axis
    const yAxis =
      yScale === 'log'
        ? // eslint-disable-next-line @typescript-eslint/no-explicit-any
          (d3.axisLeft(y as any).ticks(5, '~s') as any)
        : // eslint-disable-next-line @typescript-eslint/no-explicit-any
          (d3.axisLeft(y as any).ticks(5).tickFormat(d3.format('~s')) as any);
    g.append('g')
      .attr('class', 'axis axis-y')
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      .call(yAxis as any)
      .selectAll('text')
      .attr('fill', mutedColor)
      .attr('font-family', 'var(--font-body)')
      .attr('font-size', '11px');

    g.selectAll('.axis .domain').attr('stroke', borderStrong);
    g.selectAll('.axis .tick line').attr('stroke', borderStrong);

    // x-axis label
    g.append('text')
      .attr('x', width / 2)
      .attr('y', height + 30)
      .attr('text-anchor', 'middle')
      .attr('fill', mutedColor)
      .attr('font-family', 'var(--font-body)')
      .attr('font-size', '11px')
      .text(
        xAxisMode === 'step' ? 'Step' : xAxisMode === 'relative' ? 'Time (relative)' : 'Wall time',
      );

    const chartArea = g.append('g').attr('clip-path', `url(#${clipId})`);

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const buildLine = (xs: any, accessor: (p: PreparedPoint) => number) =>
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (d3 as any)
        .line()
        .defined((p: PreparedPoint) => Number.isFinite(accessor(p)))
        .x((p: PreparedPoint) => xs(p.x))
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        .y((p: PreparedPoint) => (y as any)(accessor(p)))
        .curve(d3.curveMonotoneX);

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const rawPaths: { sel: any; series: PreparedSeries }[] = [];
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const smoothPaths: { sel: any; series: PreparedSeries }[] = [];

    for (const s of prepared) {
      // raw (faded if smoothing on, full otherwise)
      const rawPath = chartArea
        .append('path')
        .datum(s.points)
        .attr('fill', 'none')
        .attr('stroke', s.color)
        .attr('stroke-width', smoothing > 0 ? 1 : 1.8)
        .attr('opacity', smoothing > 0 ? 0.35 : 1)
        .attr('d', buildLine(x, (p) => p.raw));
      rawPaths.push({ sel: rawPath, series: s });

      if (smoothing > 0) {
        const smoothPath = chartArea
          .append('path')
          .datum(s.points)
          .attr('fill', 'none')
          .attr('stroke', s.color)
          .attr('stroke-width', 2)
          .attr('opacity', 1)
          .attr('d', buildLine(x, (p) => p.smoothed));
        smoothPaths.push({ sel: smoothPath, series: s });
      }
    }

    // Crosshair group
    const cross = g.append('g').attr('class', 'crosshair').style('display', 'none');
    const crossLine = cross
      .append('line')
      .attr('y1', 0)
      .attr('y2', height)
      .attr('stroke', borderStrong)
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '3,3');

    const crossDots = prepared.map((s) =>
      cross.append('circle').attr('r', 4).attr('fill', s.color).attr('stroke', '#fff').attr('stroke-width', 1),
    );

    const tooltip = d3
      .select(container)
      .append('div')
      .attr('class', 'metric-graph-tooltip')
      .style('display', 'none');

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    let currentX: any = x;

    function nearestPoint(s: PreparedSeries, xValue: number): PreparedPoint | null {
      if (s.points.length === 0) return null;
      // binary search since points are sorted by step (and x is monotone in step for step/wall/relative)
      let lo = 0;
      let hi = s.points.length - 1;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (s.points[mid].x < xValue) lo = mid + 1;
        else hi = mid;
      }
      const p = s.points[lo];
      if (lo === 0) return p;
      const prev = s.points[lo - 1];
      return Math.abs(prev.x - xValue) < Math.abs(p.x - xValue) ? prev : p;
    }

    function moveCrosshair(event: MouseEvent) {
      const [mx, my] = d3.pointer(event, g.node());
      if (mx < 0 || mx > width || my < 0 || my > height) {
        cross.style('display', 'none');
        tooltip.style('display', 'none');
        return;
      }
      const xVal = currentX.invert(mx);
      const xValNum = xAxisMode === 'wall' ? (xVal as Date).getTime() : (xVal as number);

      cross.style('display', null);
      crossLine.attr('x1', mx).attr('x2', mx);

      const rows: { name: string; color: string; value: number }[] = [];
      prepared.forEach((s, i) => {
        const p = nearestPoint(s, xValNum);
        const dot = crossDots[i];
        if (!p) {
          dot.style('display', 'none');
          return;
        }
        const yVal = smoothing > 0 ? p.smoothed : p.raw;
        dot
          .style('display', null)
          .attr('cx', currentX(p.x))
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          .attr('cy', (y as any)(yVal));
        rows.push({ name: s.runName, color: s.color, value: yVal });
      });

      rows.sort((a, b) => b.value - a.value);

      const xLabel = xAxisMode === 'wall'
        ? d3.timeFormat('%H:%M:%S')(new Date(xValNum))
        : xAxisMode === 'relative'
          ? formatTick('relative')(xValNum)
          : `${Math.round(xValNum)}`;

      tooltip
        .style('display', null)
        .html(
          `<div class="mg-tt-x">${xLabel}</div>` +
            rows
              .map(
                (r) =>
                  `<div class="mg-tt-row"><span class="mg-tt-swatch" style="background:${r.color}"></span><span class="mg-tt-name">${escapeHtml(r.name)}</span><span class="mg-tt-val">${formatNum(r.value)}</span></div>`,
              )
              .join(''),
        );

      // Position tooltip near cursor but inside container
      const containerRect = container.getBoundingClientRect();
      const ttNode = tooltip.node() as HTMLDivElement;
      const ttW = ttNode.offsetWidth;
      const ttH = ttNode.offsetHeight;
      let left = event.clientX - containerRect.left + 12;
      let top = event.clientY - containerRect.top + 12;
      if (left + ttW > containerRect.width - 4) left = event.clientX - containerRect.left - ttW - 12;
      if (top + ttH > containerRect.height - 4) top = event.clientY - containerRect.top - ttH - 12;
      tooltip.style('left', `${left}px`).style('top', `${top}px`);
    }

    overlay
      .on('mousemove', moveCrosshair)
      .on('mouseleave', () => {
        cross.style('display', 'none');
        tooltip.style('display', 'none');
      });

    // Zoom on x-axis
    zoomRef.current = d3
      .zoom()
      .scaleExtent([1, 50])
      .extent([[0, 0], [width, height]])
      .translateExtent([[0, 0], [width, height]])
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      .on('zoom', (event: any) => {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const newX = event.transform.rescaleX(x as any);
        currentX = newX;

        // Update x axis ticks
        const updatedAxis =
          xAxisMode === 'wall'
            ? // eslint-disable-next-line @typescript-eslint/no-explicit-any
              (d3.axisBottom(newX as any).ticks(5) as any)
            : // eslint-disable-next-line @typescript-eslint/no-explicit-any
              (d3.axisBottom(newX as any).ticks(5).tickFormat(formatTick(xAxisMode) as any) as any);
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        xAxisGroup.call(updatedAxis as any);
        xAxisGroup
          .selectAll('text')
          .attr('fill', mutedColor)
          .attr('font-family', 'var(--font-body)')
          .attr('font-size', '11px');
        xAxisGroup.selectAll('.domain').attr('stroke', borderStrong);
        xAxisGroup.selectAll('.tick line').attr('stroke', borderStrong);

        for (const { sel } of rawPaths) sel.attr('d', buildLine(newX, (p) => p.raw));
        for (const { sel } of smoothPaths) sel.attr('d', buildLine(newX, (p) => p.smoothed));
      });

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    svg.call(zoomRef.current as any);

    return () => {
      tooltip.remove();
    };
  }, [prepared, metricName, smoothing, xAxisMode, yScale, sizeTick, ignoreOutliers]);

  // Resize handling — bump a counter so the chart effect re-runs at the new size.
  useEffect(() => {
    if (!containerRef.current) return;
    const ro = new ResizeObserver(() => {
      setSizeTick((t) => t + 1);
    });
    ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, []);

  // Latest values for header
  const latestSummary = useMemo(() => {
    if (prepared.length === 0) return null;
    if (prepared.length === 1) {
      const p = prepared[0].points[prepared[0].points.length - 1];
      return p ? `Latest: ${formatNum(smoothing > 0 ? p.smoothed : p.raw)}` : null;
    }
    return `${prepared.length} runs`;
  }, [prepared, smoothing]);

  function downloadCsv() {
    const rows = ['run_name,run_id,step,log_time,value'];
    for (const s of series) {
      const escaped = csvEscape(s.runName);
      for (const r of s.data) {
        rows.push(`${escaped},${s.runId},${r.step},${r.log_time},${r.value}`);
      }
    }
    const blob = new Blob([rows.join('\n')], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${metricName.replace(/[^a-zA-Z0-9_-]/g, '_')}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  function resetZoom() {
    if (svgRef.current && zoomRef.current) {
      const svg = d3.select(svgRef.current);
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      svg.transition().duration(500).call(zoomRef.current.transform as any, d3.zoomIdentity);
    }
  }

  const isEmpty = prepared.length === 0;

  return (
    <div className="metric-graph" ref={containerRef}>
      <div className="metric-graph-header">
        <h3 className="metric-graph-title">{metricName}</h3>
        <div className="metric-graph-header-right">
          {latestSummary && <span className="metric-graph-value">{latestSummary}</span>}
          <button
            className="metric-graph-iconbtn"
            title="Download CSV"
            onClick={downloadCsv}
            disabled={isEmpty}
          >
            <Download size={13} />
          </button>
          <button
            className="metric-graph-iconbtn"
            title="Reset zoom"
            onClick={resetZoom}
            disabled={isEmpty}
          >
            <RotateCcw size={13} />
          </button>
        </div>
      </div>

      {prepared.length > 0 && (
        <div className="metric-graph-legend">
          {prepared.map((s) => (
            <span key={s.runId} className="metric-graph-legend-item">
              <span className="metric-graph-legend-swatch" style={{ background: s.color }} />
              {s.runName}
            </span>
          ))}
        </div>
      )}

      <svg ref={svgRef} />
      {isEmpty && (
        <div className="metric-graph-empty">
          {droppedForLog > 0
            ? 'No positive values to plot on log scale.'
            : 'No data available'}
        </div>
      )}
      {!isEmpty && droppedForLog > 0 && (
        <div className="metric-graph-note">
          {droppedForLog} non-positive value{droppedForLog === 1 ? '' : 's'} hidden for log scale
        </div>
      )}
    </div>
  );
}

function formatNum(v: number): string {
  if (!Number.isFinite(v)) return '-';
  const abs = Math.abs(v);
  if (abs !== 0 && (abs < 1e-3 || abs >= 1e6)) return v.toExponential(3);
  return v.toFixed(4);
}

function csvEscape(v: string): string {
  if (/[",\n]/.test(v)) return `"${v.replace(/"/g, '""')}"`;
  return v;
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) => {
    switch (c) {
      case '&': return '&amp;';
      case '<': return '&lt;';
      case '>': return '&gt;';
      case '"': return '&quot;';
      default: return '&#39;';
    }
  });
}
