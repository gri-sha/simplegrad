/**
 * TypeScript types for simpleboard
 */

export interface RunInfo {
  run_id: number;
  name: string;
  created_at: string;
  status: 'running' | 'completed' | 'failed';
  config: Record<string, unknown>;
  num_records?: number[];
  metrics?: string[];
}

export interface RecordInfo {
  step: number;
  value: number;
  log_time: number;
}

export interface DatabaseInfo {
  available_databases: string[];
  current_database: string | null;
}

export interface MetricsResponse {
  run_id: number;
  metrics: Record<string, RecordInfo[]>;
}

export interface MetricNamesResponse {
  run_id: number;
  metrics: string[];
}

export interface CompGraphNode {
  id: string;
  label: string;
  type: string;
  shape?: number[];
  comp_grad?: boolean;
  is_leaf?: boolean;
}

export interface CompGraphEdge {
  source: string;
  target: string;
}

export interface CompGraphData {
  nodes: CompGraphNode[];
  edges: CompGraphEdge[];
}

export interface CompGraphsResponse {
  run_id: number;
  graphs: Array<{
    id: number;
    graph: CompGraphData;
    created_at: number;
  }>;
}
