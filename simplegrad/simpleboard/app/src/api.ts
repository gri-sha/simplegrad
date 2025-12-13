export const DEFAULT_API_URL = 'http://localhost:8000';

export const getApiUrl = () => {
  return localStorage.getItem('simplegrad_api_url') || DEFAULT_API_URL;
};

export const setApiUrl = (url: string) => {
  localStorage.setItem('simplegrad_api_url', url);
};

export interface RunInfo {
  run_id: number;
  name: string;
  created_at: number;
  status: string;
  config: Record<string, any>;
}

export interface RunRecords {
  run_id: number;
  metrics: Record<string, Array<{ step: number; value: number; wall_time: number }>>;
}

export interface MetricListResponse {
  run_id: number;
  metrics: string[];
}

export interface DatabaseInfo {
  available_databases: string[];
  current_database: string | null;
}

export const api = {
  getDatabases: async (): Promise<DatabaseInfo> => {
    const res = await fetch(`${getApiUrl()}/api/databases`);
    if (!res.ok) throw new Error('Failed to fetch databases');
    return res.json();
  },

  selectDatabase: async (dbName: string): Promise<{ message: string }> => {
    const res = await fetch(`${getApiUrl()}/api/databases/select`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ db_name: dbName })
    });
    if (!res.ok) throw new Error('Failed to select database');
    return res.json();
  },

  listRuns: async (): Promise<RunInfo[]> => {
    const res = await fetch(`${getApiUrl()}/api/runs`);
    if (!res.ok) throw new Error('Failed to fetch runs');
    return res.json();
  },

  getRun: async (runId: number): Promise<RunInfo> => {
    const res = await fetch(`${getApiUrl()}/api/runs/${runId}`);
    if (!res.ok) throw new Error('Failed to fetch run');
    return res.json();
  },

  getMetrics: async (runId: number): Promise<MetricListResponse> => {
    const res = await fetch(`${getApiUrl()}/api/runs/${runId}/metrics`);
    if (!res.ok) throw new Error('Failed to fetch metrics');
    return res.json();
  },

  getRecords: async (runId: number, metricName?: string): Promise<RunRecords> => {
    const url = new URL(`${getApiUrl()}/api/runs/${runId}/records`);
    if (metricName) {
      url.searchParams.append('metric_name', metricName);
    }
    const res = await fetch(url.toString());
    if (!res.ok) throw new Error('Failed to fetch records');
    return res.json();
  }
};
