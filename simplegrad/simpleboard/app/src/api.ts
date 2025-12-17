/**
 * API client for simpleboard
 */

import type { 
  RunInfo, 
  DatabaseInfo, 
  MetricsResponse, 
  MetricNamesResponse,
  CompGraphsResponse 
} from './types';

const DEFAULT_API_URL = 'http://localhost:8000';

export const getApiUrl = (): string => {
  return localStorage.getItem('simpleboard_api_url') || DEFAULT_API_URL;
};

export const setApiUrl = (url: string): void => {
  localStorage.setItem('simpleboard_api_url', url);
};

class ApiClient {
  private getBaseUrl(): string {
    return getApiUrl();
  }

  async getDatabases(): Promise<DatabaseInfo> {
    console.log('Fetching databases from', this.getBaseUrl());
    const res = await fetch(`${this.getBaseUrl()}/api/databases`);
    if (!res.ok) throw new Error('Failed to fetch databases');
    return res.json();
  }

  async selectDatabase(dbName: string): Promise<{ message: string }> {
    const res = await fetch(`${this.getBaseUrl()}/api/databases/select`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ db_name: dbName })
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
    const url = new URL(`${this.getBaseUrl()}/api/runs/${runId}/records`);
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

  createWebSocket(runId: number): WebSocket {
    const wsUrl = this.getBaseUrl().replace('http', 'ws');
    return new WebSocket(`${wsUrl}/ws/${runId}`);
  }
}

export const api = new ApiClient();
