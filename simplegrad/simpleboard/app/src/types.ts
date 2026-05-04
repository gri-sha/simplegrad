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

export interface RunMeta {
  rename?: string;
  pinned?: boolean;
  hidden?: boolean;
  starred?: boolean;
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

export interface HistogramInfo {
  step: number;
  bucket_edges: number[];
  bucket_counts: number[];
  log_time: number;
}

export interface HistogramsResponse {
  run_id: number;
  histograms: Record<string, HistogramInfo[]>;
}

export interface ImageInfo {
  step: number;
  width: number;
  height: number;
  channels: number;
  data_b64: string;
  log_time: number;
}

export interface ImagesResponse {
  run_id: number;
  images: Record<string, ImageInfo[]>;
}

export type XAxisMode = 'step' | 'relative' | 'wall';
export type YScaleMode = 'linear' | 'log';

export interface MetricSeries {
  runId: number;
  runName: string;
  color: string;
  data: RecordInfo[];
}
