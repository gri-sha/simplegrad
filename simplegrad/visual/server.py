"""
FastAPI server for the visualization dashboard.
Provides REST API endpoints for accessing training runs and metrics.
"""

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .storage import RunStorage
from .types import (
    RunInfo,
    RunCreate,
    RunRecords,
    MetricListResponse,
    GraphResponse,
    StatusUpdateRequest,
    MessageResponse,
)


app = FastAPI(
    title="SimpleGrad Dashboard",
    description="Training visualization dashboard for SimpleGrad",
    version="0.1.0",
)

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

store: Optional[RunStorage] = None


def get_store() -> RunStorage:
    """Get the run store, initializing if needed."""
    global store
    if store is None:
        logdir = os.environ.get("SIMPLEGRAD_LOGDIR", "./runs")
        store = RunStorage(logdir=logdir)
    return store


@app.get("/api/runs", response_model=list[RunInfo])
async def list_runs():
    """List all training runs."""
    return get_store().list_runs()


@app.post("/api/runs", response_model=RunInfo)
async def create_run(run_data: RunCreate):
    """Create a new training run."""
    run_id = get_store().create_run(name=run_data.name, config=run_data.config)
    run = get_store().get_run(run_id)
    if run is None:
        raise HTTPException(status_code=500, detail="Failed to create run")
    return run


@app.get("/api/runs/{run_id}", response_model=RunInfo)
async def get_run(run_id: int):
    """Get details for a specific run."""
    run = get_store().get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return run


@app.delete("/api/runs/{run_id}", response_model=MessageResponse)
async def delete_run(run_id: int):
    """Delete a training run and all its data."""
    run = get_store().get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    get_store().delete_run(run_id)
    return MessageResponse(message=f"Run {run_id} deleted")


@app.patch("/api/runs/{run_id}/status", response_model=MessageResponse)
async def update_run_status(run_id: int, request: StatusUpdateRequest):
    """Update the status of a run."""
    run = get_store().get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    get_store().update_run_status(run_id, request.status)
    return MessageResponse(message=f"Run {run_id} status updated to {request.status}")


@app.get("/api/runs/{run_id}/records", response_model=RunRecords)
async def get_records(run_id: int, metric_name: Optional[str] = None):
    """Get metric records for a specific run."""
    run = get_store().get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    metrics = get_store().get_records(run_id, metric_name)
    return RunRecords(run_id=run_id, metrics=metrics)


@app.get("/api/runs/{run_id}/metrics", response_model=MetricListResponse)
async def get_metrics(run_id: int):
    """Get list of available metrics for a run."""
    run = get_store().get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    names = get_store().get_metric_names(run_id)
    return MetricListResponse(run_id=run_id, metrics=names)


@app.get("/api/runs/{run_id}/graph", response_model=GraphResponse)
async def get_graph(run_id: int):
    """Get computation graph for a run."""
    run = get_store().get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    graph = get_store().get_graph(run_id)
    if graph is None:
        raise HTTPException(status_code=404, detail=f"No graph found for run {run_id}")

    return GraphResponse(run_id=run_id, graph=graph)


# ===== WebSocket for Real-time Updates =====


class ConnectionManager:
    """Manage WebSocket connections for real-time updates."""

    def __init__(self):
        self.active_connections: dict[int, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, run_id: int):
        await websocket.accept()
        if run_id not in self.active_connections:
            self.active_connections[run_id] = []
        self.active_connections[run_id].append(websocket)

    def disconnect(self, websocket: WebSocket, run_id: int):
        if run_id in self.active_connections:
            self.active_connections[run_id].remove(websocket)
            if not self.active_connections[run_id]:
                del self.active_connections[run_id]

    async def broadcast(self, run_id: int, message: dict):
        """Broadcast message to all connections watching a run."""
        if run_id in self.active_connections:
            for connection in self.active_connections[run_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    pass


manager = ConnectionManager()


@app.websocket("/ws/{run_id}")
async def websocket_endpoint(websocket: WebSocket, run_id: int):
    """WebSocket endpoint for real-time metric updates."""
    await manager.connect(websocket, run_id)
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        manager.disconnect(websocket, run_id)


# ===== Static File Serving =====

DASHBOARD_DIST = Path(__file__).parent.parent.parent / "dashboard" / "dist"

if DASHBOARD_DIST.exists():
    app.mount("/assets", StaticFiles(directory=DASHBOARD_DIST / "assets"), name="assets")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(DASHBOARD_DIST / "index.html")

    @app.get("/{path:path}")
    async def serve_frontend_routes(path: str):
        file_path = DASHBOARD_DIST / path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(DASHBOARD_DIST / "index.html")
