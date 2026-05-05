/**
 * API client for simpleboard
 */

import type {
  RunInfo,
  DatabaseInfo,
  MetricsResponse,
  MetricNamesResponse,
  CompGraphsResponse,
  HistogramsResponse,
  ImagesResponse,
} from './types';

const DEFAULT_API_URL = '';

export const getApiUrl = (): string => {
  return localStorage.getItem('simpleboard_api_url') || DEFAULT_API_URL;
};

export const setApiUrl = (url: string): void => {
  if (url.trim() === '') {
    localStorage.removeItem('simpleboard_api_url');
  } else {
    localStorage.setItem('simpleboard_api_url', url.trim());
  }
};

export const clearApiUrl = (): void => {
  localStorage.removeItem('simpleboard_api_url');
};

class ApiClient {
  private getBaseUrl(): string {
    return getApiUrl();
  }

  async getDatabases(): Promise<DatabaseInfo> {
    const storedBase = this.getBaseUrl();
    // Try the stored/configured base URL first. If it fails (stale port, wrong
    // host, etc.) and a non-empty URL was stored, automatically fall back to
    // relative URLs (same origin as the page) and clear the stale value so the
    // problem doesn't repeat next time.
    try {
      const res = await fetch(`${storedBase}/api/databases`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.json();
    } catch (err) {
      if (storedBase !== DEFAULT_API_URL) {
        console.warn(
          `simpleboard: could not reach ${storedBase}/api/databases (${err}). ` +
          `Falling back to same-origin and clearing the stored URL.`
        );
        clearApiUrl();
        const res = await fetch(`/api/databases`);
        if (!res.ok) throw new Error('Failed to fetch databases');
        return res.json();
      }
      throw err;
    }
  }

  async selectDatabase(dbName: string): Promise<{ message: string }> {
    const res = await fetch(`${this.getBaseUrl()}/api/databases/select`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ db_name: dbName }),
    });
    if (!res.ok) throw new Error('Failed to select database');
    return res.json();
  }

  async getRuns(): Promise<RunInfo[]> {
    const res = await fetch(`${this.getBaseUrl()}/api/runs`);
    if (!res.ok) throw new Error('Failed to fetch runs');
    return res.json();
  }

  async getRun(runId: number): Promise<RunInfo> {
    const res = await fetch(`${this.getBaseUrl()}/api/runs/${runId}`);
    if (!res.ok) throw new Error('Failed to fetch run');
    return res.json();
  }

  async getMetricNames(runId: number): Promise<MetricNamesResponse> {
    const res = await fetch(`${this.getBaseUrl()}/api/runs/${runId}/metrics`);
    if (!res.ok) throw new Error('Failed to fetch metric names');
    return res.json();
  }

  async getRecords(runId: number, metricName?: string): Promise<MetricsResponse> {
    const base = this.getBaseUrl() || window.location.origin;
    const url = new URL(`${base}/api/runs/${runId}/records`);
    if (metricName) {
      url.searchParams.append('metric_name', metricName);
    }
    const res = await fetch(url.toString());
    if (!res.ok) throw new Error('Failed to fetch records');
    return res.json();
  }

  async getGraphs(runId: number): Promise<CompGraphsResponse> {
    const res = await fetch(`${this.getBaseUrl()}/api/runs/${runId}/graphs`);
    if (!res.ok) throw new Error('Failed to fetch graphs');
    return res.json();
  }

  async getHistograms(runId: number): Promise<HistogramsResponse> {
    const res = await fetch(`${this.getBaseUrl()}/api/runs/${runId}/histograms`);
    if (!res.ok) throw new Error('Failed to fetch histograms');
    return res.json();
  }

  async getImages(runId: number): Promise<ImagesResponse> {
    const res = await fetch(`${this.getBaseUrl()}/api/runs/${runId}/images`);
    if (!res.ok) throw new Error('Failed to fetch images');
    return res.json();
  }

  async getConfig(): Promise<{ exp_dir: string }> {
    const res = await fetch(`${this.getBaseUrl()}/api/config`);
    if (!res.ok) throw new Error('Failed to fetch config');
    return res.json();
  }

  async updateExpDir(path: string): Promise<{ exp_dir: string }> {
    const res = await fetch(`${this.getBaseUrl()}/api/config/exp-dir`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path }),
    });
    if (!res.ok) throw new Error('Failed to update experiments directory');
    return res.json();
  }

  createWebSocket(runId: number): WebSocket | null {
    try {
      const base = this.getBaseUrl();
      const wsUrl = base 
        ? base.replace('http', 'ws')
        : (window.location.protocol === 'https:' ? 'wss:' : 'ws:') + '//' + window.location.host;
      return new WebSocket(`${wsUrl}/ws/${runId}`);
    } catch (e) {
      console.warn("WebSocket not supported by backend", e);
      return null;
    }
  }
}

export const api = new ApiClient();
