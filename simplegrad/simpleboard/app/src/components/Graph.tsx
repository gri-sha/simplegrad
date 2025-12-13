import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

interface DataPoint {
  step: number;
  value: number;
  wall_time: number;
}

interface GraphProps {
  data: DataPoint[];
  metricName: string;
  color?: string;
}

export const Graph: React.FC<GraphProps> = ({ data, metricName, color = '#528AC5' }) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!data || data.length === 0 || !svgRef.current || !containerRef.current) return;

    const containerWidth = containerRef.current.clientWidth;
    const containerHeight = 300;
    
    const margin = { top: 20, right: 30, bottom: 30, left: 40 };
    const width = containerWidth - margin.left - margin.right;
    const height = containerHeight - margin.top - margin.bottom;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove(); // Clear previous render

    const g = svg
      .attr("width", containerWidth)
      .attr("height", containerHeight)
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // X Axis
    const x = d3.scaleLinear()
      .domain(d3.extent(data, d => d.step) as [number, number])
      .range([0, width]);

    g.append("g")
      .attr("transform", `translate(0,${height})`)
      .call(d3.axisBottom(x).ticks(5).tickSize(0).tickPadding(10))
      .attr("font-family", "Open Sans")
      .attr("font-size", "10px")
      .select(".domain").attr("stroke-width", 2);

    // Y Axis
    const y = d3.scaleLinear()
      .domain([d3.min(data, d => d.value) as number, d3.max(data, d => d.value) as number])
      .nice()
      .range([height, 0]);

    g.append("g")
      .call(d3.axisLeft(y).ticks(5).tickSize(-width).tickPadding(10))
      .attr("font-family", "Open Sans")
      .attr("font-size", "10px")
      .call(g => g.select(".domain").remove())
      .call(g => g.selectAll(".tick line").attr("stroke", "#ddd").attr("stroke-dasharray", "2,2"));

    // Line
    const line = d3.line<DataPoint>()
      .x(d => x(d.step))
      .y(d => y(d.value))
      .curve(d3.curveMonotoneX);

    g.append("path")
      .datum(data)
      .attr("fill", "none")
      .attr("stroke", color)
      .attr("stroke-width", 3)
      .attr("d", line);

    // Dots
    g.selectAll(".dot")
      .data(data)
      .enter()
      .append("circle")
      .attr("class", "dot")
      .attr("cx", d => x(d.step))
      .attr("cy", d => y(d.value))
      .attr("r", 3)
      .attr("fill", "#1B1E20")
      .attr("stroke", "white")
      .attr("stroke-width", 1);

  }, [data, metricName, color]);

  return (
    <div className="nb-box" style={{ padding: '16px', marginBottom: '20px', background: 'white' }} ref={containerRef}>
      <h3 style={{ marginTop: 0, marginBottom: '16px', fontSize: '1rem', textTransform: 'uppercase' }}>
        {metricName}
      </h3>
      <svg ref={svgRef} style={{ width: '100%', height: '300px' }} />
    </div>
  );
};
