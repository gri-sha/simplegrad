/**
 * D3.js Metric Graph component
 */

import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import type { RecordInfo } from '../types';

interface MetricGraphProps {
  metricName: string;
  data: RecordInfo[];
  color?: string;
}

const COLORS = [
  '#2563eb', // blue
  '#dc2626', // red
  '#16a34a', // green
  '#ca8a04', // yellow
  '#9333ea', // purple
  '#0891b2', // cyan
];

export function MetricGraph({ metricName, data, color }: MetricGraphProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  // Generate consistent color from metric name
  const chartColor = color || COLORS[Math.abs(hashCode(metricName)) % COLORS.length];

  useEffect(() => {
    if (!data || data.length === 0 || !svgRef.current || !containerRef.current) return;

    const container = containerRef.current;
    const svg = d3.select(svgRef.current);

    // Clear previous content
    svg.selectAll('*').remove();

    const margin = { top: 20, right: 30, bottom: 40, left: 50 };
    const width = container.clientWidth - margin.left - margin.right;
    const height = 200 - margin.top - margin.bottom;

    svg.attr('width', container.clientWidth).attr('height', 200);

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    // Scales
    const xExtent = d3.extent(data, (d: RecordInfo) => d.step) as [number, number];
    const yExtent = d3.extent(data, (d: RecordInfo) => d.value) as [number, number];

    // Add padding to y extent
    const yPadding = (yExtent[1] - yExtent[0]) * 0.1 || 1;
    
    // Add padding to x extent to keep line away from edges
    const xRange = xExtent[1] - xExtent[0];
    const xPadding = xRange * 0.02 || 1;

    const x = d3.scaleLinear().domain([xExtent[0], xExtent[1] + xPadding]).range([0, width]);

    const y = d3
      .scaleLinear()
      .domain([yExtent[0] - yPadding, yExtent[1] + yPadding])
      .range([height, 0]);

    // Grid lines
    g.append('g')
      .attr('class', 'grid')
      .attr('transform', `translate(0,${height})`)
      .call(
        d3
          .axisBottom(x)
          .ticks(5)
          .tickSize(-height)
          .tickFormat(() => '')
      )
      .selectAll('line')
      .attr('stroke', '#e5e7eb')
      .attr('stroke-dasharray', '2,2');

    g.append('g')
      .attr('class', 'grid')
      .call(
        d3
          .axisLeft(y)
          .ticks(5)
          .tickSize(-width)
          .tickFormat(() => '')
      )
      .selectAll('line')
      .attr('stroke', '#e5e7eb')
      .attr('stroke-dasharray', '2,2');

    // Remove domain lines from grid
    g.selectAll('.grid .domain').remove();

    // Axes
    g.append('g')
      .attr('class', 'axis axis-x')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(x).ticks(5))
      .selectAll('text')
      .attr('fill', '#6b7280')
      .attr('font-size', '11px');

    g.append('g')
      .attr('class', 'axis axis-y')
      .call(d3.axisLeft(y).ticks(5).tickFormat(d3.format('.3s')))
      .selectAll('text')
      .attr('fill', '#6b7280')
      .attr('font-size', '11px');

    g.selectAll('.axis .domain').attr('stroke', '#d1d5db');
    g.selectAll('.axis .tick line').attr('stroke', '#d1d5db');

    // Line
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const line = (d3 as any)
      .line()
      .x((d: RecordInfo) => x(d.step))
      .y((d: RecordInfo) => y(d.value))
      .curve(d3.curveMonotoneX);

    g.append('path')
      .datum(data)
      .attr('fill', 'none')
      .attr('stroke', chartColor)
      .attr('stroke-width', 2)
      .attr('d', line);

    // Dots (only show if not too many points)
    if (data.length <= 50) {
      g.selectAll('.dot')
        .data(data)
        .enter()
        .append('circle')
        .attr('class', 'dot')
        .attr('cx', (d: RecordInfo) => x(d.step))
        .attr('cy', (d: RecordInfo) => y(d.value))
        .attr('r', 3)
        .attr('fill', chartColor);
    }

    // X-axis label
    g.append('text')
      .attr('x', width / 2)
      .attr('y', height + 35)
      .attr('text-anchor', 'middle')
      .attr('fill', '#9ca3af')
      .attr('font-size', '11px')
      .text('Step');
  }, [data, chartColor]);

  // Handle resize
  useEffect(() => {
    const handleResize = () => {
      if (containerRef.current && svgRef.current && data.length > 0) {
        // Trigger re-render by forcing update
        const event = new Event('resize');
        window.dispatchEvent(event);
      }
    };

    const resizeObserver = new ResizeObserver(handleResize);
    if (containerRef.current) {
      resizeObserver.observe(containerRef.current);
    }

    return () => resizeObserver.disconnect();
  }, [data]);

  return (
    <div className="metric-graph" ref={containerRef}>
      <div className="metric-graph-header">
        <h3 className="metric-graph-title">{metricName}</h3>
        {data.length > 0 && (
          <span className="metric-graph-value">
            Latest: {data[data.length - 1].value.toFixed(4)}
          </span>
        )}
      </div>
      <svg ref={svgRef} />
      {data.length === 0 && <div className="metric-graph-empty">No data available</div>}
    </div>
  );
}

// Simple hash function for consistent colors
function hashCode(str: string): number {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = (hash << 5) - hash + char;
    hash = hash & hash;
  }
  return hash;
}
