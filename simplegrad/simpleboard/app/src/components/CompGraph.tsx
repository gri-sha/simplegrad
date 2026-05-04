/**
 * D3.js Computation Graph visualization component
 * Renders DAG in a left-to-right hierarchical layout similar to graphviz
 * Styled to match the inline_comp_graph.py output
 */

import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import type { CompGraphData, CompGraphNode, CompGraphEdge } from '../types';

interface CompGraphProps {
  data: CompGraphData;
  title?: string;
  theme?: 'light' | 'dark';
}

interface LayoutNode {
  id: string;
  label: string;
  type: string;
  shape?: number[];
  comp_grad?: boolean;
  is_leaf?: boolean;
  x: number;
  y: number;
  layer: number;
  width: number;
  height: number;
}

interface LayoutLink {
  source: LayoutNode;
  target: LayoutNode;
}

// Resolve node/edge colors from CSS variables so dark mode picks them up.
function resolveGraphColors() {
  const styles = getComputedStyle(document.documentElement);
  const v = (name: string, fallback: string) =>
    styles.getPropertyValue(name).trim() || fallback;
  return {
    leaf: v('--graph-node-leaf', '#FFA07A'),
    tensor: v('--graph-node-tensor', '#B0C4DE'),
    op: v('--graph-node-op', '#FAFAD2'),
    edge: v('--graph-edge', '#333'),
    arrow: v('--graph-arrow', '#666'),
    nodeStroke: v('--graph-node-stroke', '#666'),
    nodeText: v('--graph-node-text', '#333'),
  };
}

/**
 * Format node label similar to inline version
 */
function formatNodeLabel(node: CompGraphNode): string[] {
  const lines: string[] = [];

  if (node.type === 'operation' || node.type === 'op') {
    // Operation nodes just show the operation name
    lines.push(node.label);
  } else {
    // Tensor nodes show label, shape, comp_grad (like inline_comp_graph.py)
    if (node.label) {
      lines.push(`'${node.label}'`);
    }
    if (node.shape) {
      lines.push(`shape: [${node.shape.join(', ')}]`);
    }
    if (node.comp_grad !== undefined) {
      lines.push(`comp_grad: ${node.comp_grad}`);
    }
  }

  return lines;
}

/**
 * Calculate node dimensions based on content
 */
function calculateNodeSize(node: CompGraphNode): { width: number; height: number } {
  const lines = formatNodeLabel(node);
  const isOp = node.type === 'operation' || node.type === 'op';

  // Estimate width based on longest line (tight sizing)
  const maxLineLen = Math.max(...lines.map((l) => l.length), 5);
  const width = Math.max(isOp ? 50 : 100, maxLineLen * 6 + 12);
  const height = isOp ? 22 : Math.max(36, lines.length * 12 + 10);

  return { width, height };
}

/**
 * Compute left-to-right DAG layout
 * Parameters are placed near their consuming operations, not all at the start
 */
function computeDAGLayout(
  nodes: CompGraphNode[],
  edges: CompGraphEdge[]
): { nodes: LayoutNode[]; links: LayoutLink[]; width: number; height: number } {
  // Build adjacency info
  const nodeMap = new Map<string, CompGraphNode>();
  const inEdges = new Map<string, string[]>();
  const outEdges = new Map<string, string[]>();

  nodes.forEach((n) => {
    nodeMap.set(n.id, n);
    inEdges.set(n.id, []);
    outEdges.set(n.id, []);
  });

  edges.forEach((e) => {
    outEdges.get(e.source)?.push(e.target);
    inEdges.get(e.target)?.push(e.source);
  });

  // Assign layers using longest path from each node to output
  // This places parameters closer to where they're used
  const nodeLayer = new Map<string, number>();
  // const visited = new Set<string>();

  // Find the output node (no outgoing edges)
  // const outputNodes = nodes.filter((n) => (outEdges.get(n.id)?.length || 0) === 0);

  // BFS from outputs backwards to assign layers
  function assignLayer(nodeId: string): number {
    if (nodeLayer.has(nodeId)) return nodeLayer.get(nodeId)!;

    const children = outEdges.get(nodeId) || [];
    if (children.length === 0) {
      // Output node - rightmost layer
      nodeLayer.set(nodeId, 0);
      return 0;
    }

    // This node's layer is max of children + 1
    let maxChildLayer = 0;
    children.forEach((childId) => {
      const childLayer = assignLayer(childId);
      maxChildLayer = Math.max(maxChildLayer, childLayer);
    });

    const layer = maxChildLayer + 1;
    nodeLayer.set(nodeId, layer);
    return layer;
  }

  // Assign layers to all nodes
  nodes.forEach((n) => assignLayer(n.id));

  // Invert layers so inputs are on left, outputs on right
  const maxLayer = Math.max(...Array.from(nodeLayer.values()));
  nodeLayer.forEach((layer, id) => {
    nodeLayer.set(id, maxLayer - layer);
  });

  // Group nodes by layer
  const layers: string[][] = [];
  for (let i = 0; i <= maxLayer; i++) {
    layers.push([]);
  }
  nodeLayer.forEach((layer, id) => {
    layers[layer].push(id);
  });

  // Calculate positions - left to right layout (tight spacing)
  const layerGap = 30;
  const nodeGap = 6;
  const paddingLeft = 10;
  const paddingRight = 40; // Extra padding on right so graph doesn't touch edge
  const paddingY = 10;

  // First pass: calculate node sizes
  const nodeSizes = new Map<string, { width: number; height: number }>();
  nodes.forEach((n) => {
    nodeSizes.set(n.id, calculateNodeSize(n));
  });

  // Calculate layer widths and max height per layer
  const layerHeights: number[] = layers.map((layer) => {
    return layer.reduce((sum, id) => sum + nodeSizes.get(id)!.height + nodeGap, -nodeGap);
  });
  const maxLayerHeight = Math.max(...layerHeights, 100);

  // Position nodes
  const layoutNodes: LayoutNode[] = [];
  const layoutNodeMap = new Map<string, LayoutNode>();

  let currentX = paddingLeft;
  layers.forEach((layer, layerIndex) => {
    // Find max width in this layer
    const layerMaxWidth = Math.max(...layer.map((id) => nodeSizes.get(id)!.width));

    // Calculate starting Y to center this layer
    const layerHeight = layerHeights[layerIndex];
    let currentY = paddingY + (maxLayerHeight - layerHeight) / 2;

    layer.forEach((nodeId) => {
      const node = nodeMap.get(nodeId)!;
      const size = nodeSizes.get(nodeId)!;

      const layoutNode: LayoutNode = {
        id: node.id,
        label: node.label,
        type: node.type,
        shape: node.shape,
        comp_grad: node.comp_grad,
        is_leaf: node.is_leaf,
        x: currentX + layerMaxWidth / 2,
        y: currentY + size.height / 2,
        layer: layerIndex,
        width: size.width,
        height: size.height,
      };

      layoutNodes.push(layoutNode);
      layoutNodeMap.set(nodeId, layoutNode);

      currentY += size.height + nodeGap;
    });

    currentX += layerMaxWidth + layerGap;
  });

  // Create links
  const layoutLinks: LayoutLink[] = edges
    .filter((e) => layoutNodeMap.has(e.source) && layoutNodeMap.has(e.target))
    .map((e) => ({
      source: layoutNodeMap.get(e.source)!,
      target: layoutNodeMap.get(e.target)!,
    }));

  const totalWidth = currentX + paddingRight;
  const totalHeight = maxLayerHeight + paddingY * 2;

  return { nodes: layoutNodes, links: layoutLinks, width: totalWidth, height: totalHeight };
}

export function CompGraph({ data, title, theme }: CompGraphProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const zoomRef = useRef<any>(null);
  const fitTransformRef = useRef<any>(null);

  useEffect(() => {
    if (
      !data ||
      !data.nodes ||
      data.nodes.length === 0 ||
      !svgRef.current ||
      !containerRef.current
    ) {
      return;
    }

    const svg = d3.select(svgRef.current);
    const colors = resolveGraphColors();

    // Clear previous content
    svg.selectAll('*').remove();

    // Compute DAG layout
    const { nodes, links, width, height } = computeDAGLayout(data.nodes, data.edges);

    // No viewBox: d3.zoom transforms work in pixel space so the mouse wheel
    // zooms toward the cursor at a predictable rate. Size the <svg> to its
    // container and translate the inner <g> ourselves.
    const containerEl = containerRef.current!;
    const containerW = Math.max(200, containerEl.clientWidth);
    const containerH = Math.max(200, containerEl.clientHeight - 56);
    svg.attr('width', containerW).attr('height', containerH);

    // Compute the fit-to-view transform once so we can use it for the initial
    // render and for "Reset Zoom".
    const padding = 24;
    const fitScale = Math.min(
      (containerW - padding * 2) / Math.max(width, 1),
      (containerH - padding * 2) / Math.max(height, 1),
      1.5,
    );
    const fitTx = (containerW - width * fitScale) / 2;
    const fitTy = (containerH - height * fitScale) / 2;
    const fitTransform = d3.zoomIdentity.translate(fitTx, fitTy).scale(fitScale);

    // Create defs for arrow markers
    const defs = svg.append('defs');

    // Arrow marker (small)
    defs
      .append('marker')
      .attr('id', 'arrow-vee')
      .attr('viewBox', '0 -3 6 6')
      .attr('refX', 6)
      .attr('refY', 0)
      .attr('markerWidth', 5)
      .attr('markerHeight', 5)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-2L6,0L0,2')
      .attr('fill', colors.arrow);

    // Add invisible background rect to capture pan/drag events anywhere
    svg.append('rect')
      .attr('width', '100%')
      .attr('height', '100%')
      .attr('fill', 'transparent')
      .attr('style', 'cursor: grab');

    // Container group with zoom support
    const g = svg.append('g');

    zoomRef.current = d3
      .zoom()
      .scaleExtent([0.05, 10])
      // Smoother wheel-zoom — d3's default doubles per scroll; this is gentler.
      .wheelDelta((event: WheelEvent) => -event.deltaY * (event.deltaMode ? 0.05 : 0.002))
      .on('zoom', (event: any) => {
        g.attr('transform', event.transform);
      });

    svg.call(zoomRef.current as any);
    // Apply the fit-to-view transform on first render so the graph is centered
    // and fully visible before any user interaction.
    svg.call(zoomRef.current.transform as any, fitTransform);
    fitTransformRef.current = fitTransform;
    
    // Draw edges with curved paths
    g.append('g')
      .attr('class', 'edges')
      .selectAll('path')
      .data(links)
      .enter()
      .append('path')
      .attr('d', (d: LayoutLink) => {
        const sourceX = d.source.x + d.source.width / 2;
        const sourceY = d.source.y;
        const targetX = d.target.x - d.target.width / 2;
        const targetY = d.target.y;

        // Bezier curve for smooth connection
        const midX = (sourceX + targetX) / 2;

        return `M${sourceX},${sourceY} C${midX},${sourceY} ${midX},${targetY} ${targetX},${targetY}`;
      })
      .attr('fill', 'none')
      .attr('stroke', colors.edge)
      .attr('stroke-width', 1)
      .attr('marker-end', 'url(#arrow-vee)');

    // Draw nodes
    const nodeGroups = g
      .append('g')
      .attr('class', 'nodes')
      .selectAll('g')
      .data(nodes)
      .enter()
      .append('g')
      .attr('transform', (d: LayoutNode) => `translate(${d.x},${d.y})`);

    // Node shapes based on type
    nodeGroups.each(function (this: SVGGElement, d: LayoutNode) {
      const nodeG = d3.select(this);
      const isOp = d.type === 'operation' || d.type === 'op';
      // Use is_leaf from backend data for accurate coloring
      const isLeaf = d.is_leaf === true;

      const fillColor = isOp ? colors.op : isLeaf ? colors.leaf : colors.tensor;

      // All nodes are rounded rectangles (like graphviz record shape)
      nodeG
        .append('rect')
        .attr('x', -d.width / 2)
        .attr('y', -d.height / 2)
        .attr('width', d.width)
        .attr('height', d.height)
        .attr('rx', 6)
        .attr('ry', 6)
        .attr('fill', fillColor)
        .attr('stroke', colors.nodeStroke)
        .attr('stroke-width', 1);
    });

    // Node labels - multiline support
    nodeGroups.each(function (this: SVGGElement, d: LayoutNode) {
      const nodeG = d3.select(this);
      const lines = formatNodeLabel(d);
      const isOp = d.type === 'operation' || d.type === 'op';

      const lineHeight = 12;
      const startY = (-(lines.length - 1) * lineHeight) / 2;

      lines.forEach((line, i) => {
        nodeG
          .append('text')
          .text(line)
          .attr('x', 0)
          .attr('y', startY + i * lineHeight)
          .attr('dy', '0.35em')
          .attr('text-anchor', 'middle')
          .attr('fill', colors.nodeText)
          .attr('font-size', isOp ? '10px' : '9px')
          .attr('font-weight', isOp ? '500' : '400')
          .attr('font-family', 'monospace');
      });
    });

    // Tooltip on hover
    nodeGroups.append('title').text((d: LayoutNode) => {
      let tooltip = d.label;
      if (d.type) tooltip += `\nType: ${d.type}`;
      if (d.shape) tooltip += `\nShape: [${d.shape.join(', ')}]`;
      return tooltip;
    });
  }, [data, theme]);

  return (
    <div className="comp-graph" ref={containerRef} style={{ position: 'relative', resize: 'both', overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
      <div className="comp-graph-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px', flexShrink: 0 }}>
        {title && <h3 className="comp-graph-title" style={{ margin: 0 }}>{title}</h3>}
        <button 
          className="topbar-button"
          style={{ padding: '4px 8px', fontSize: '12px', borderRadius: '4px', background: 'var(--bg-secondary)', border: '1px solid var(--border)', cursor: 'pointer' }}
          onClick={() => {
            if (svgRef.current && zoomRef.current) {
              const svg = d3.select(svgRef.current);
              const target = fitTransformRef.current ?? d3.zoomIdentity;
              svg.transition().duration(500).call(zoomRef.current.transform as any, target);
            }
          }}
        >
          Reset Zoom
        </button>
      </div>
      <div className="comp-graph-container" style={{ overflow: 'hidden', border: '1px solid var(--border)', borderRadius: '8px', flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
        <svg ref={svgRef} style={{ width: '100%', height: '100%', flexGrow: 1, display: 'block' }} />
      </div>
      {(!data || !data.nodes || data.nodes.length === 0) && (
        <div className="comp-graph-empty">No computation graph available</div>
      )}
    </div>
  );
}
