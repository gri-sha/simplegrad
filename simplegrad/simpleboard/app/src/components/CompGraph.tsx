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

// Node colors matching inline_comp_graph.py style
const NODE_COLORS: Record<string, string> = {
  leaf: '#FFA07A', // lightsalmon - for leaf tensors (inputs/params)
  tensor: '#B0C4DE', // lightsteelblue - for intermediate tensors
  operation: '#FAFAD2', // lightgoldenrodyellow - for operations
  op: '#FAFAD2',
  default: '#B0C4DE',
};

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

export function CompGraph({ data, title }: CompGraphProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);

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

    // Clear previous content
    svg.selectAll('*').remove();

    // Compute DAG layout
    const { nodes, links, width, height } = computeDAGLayout(data.nodes, data.edges);

    svg.attr('width', Math.max(width, 400)).attr('height', Math.max(height, 200));

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
      .attr('fill', '#666');

    // Container group (no zoom - use native scrolling)
    const g = svg.append('g');

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
      .attr('stroke', '#333')
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

      let fillColor: string;
      if (isOp) {
        fillColor = NODE_COLORS.operation;
      } else if (isLeaf) {
        fillColor = NODE_COLORS.leaf;
      } else {
        fillColor = NODE_COLORS.tensor;
      }

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
        .attr('stroke', '#666')
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
          .attr('fill', '#333')
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
  }, [data]);

  return (
    <div className="comp-graph" ref={containerRef}>
      {title && <h3 className="comp-graph-title">{title}</h3>}
      <div className="comp-graph-container">
        <svg ref={svgRef} />
      </div>
      {(!data || !data.nodes || data.nodes.length === 0) && (
        <div className="comp-graph-empty">No computation graph available</div>
      )}
    </div>
  );
}
