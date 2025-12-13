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

from typing import Optional
from pydantic import BaseModel

from simplegrad.track import ExperimentDBManager, RunInfo, RecordInfo


app = FastAPI(
    title="simpleboard",
    description="training visualization dashboard for simplegrad",
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

all_exp_dir: Optional[Path] = None
exp_db: Optional[ExperimentDBManager] = None


def init_exp_dir():
    """Initialize the experiments directory."""
    global all_exp_dir
    if all_exp_dir is None:
        all_exp_dir = Path(os.environ.get("SG_EXPERIMENTS_DIR", "./experiments"))
        all_exp_dir.mkdir(parents=True, exist_ok=True)


def get_exp_db() -> ExperimentDBManager:
    """Get the experiment database manager, initializing if needed."""
    global exp_db
    if exp_db is None:
        init_exp_dir()
        db_name = os.environ.get("SG_DB_NAME", "experiment.db")
        db_path = all_exp_dir / db_name
        exp_db = ExperimentDBManager(db_path=db_path)
        if not exp_db.check_connection():
            exp_db.init_exp_db()
    return exp_db


def set_exp_db(db_name: str) -> bool:
    """Switch to a different experiment database. Returns True if successful."""
    global exp_db
    init_exp_dir()
    db_path = all_exp_dir / db_name
    if not db_path.exists():
        return False
    exp_db = ExperimentDBManager(db_path=db_path)
    return exp_db.check_connection()


class CreateRunRequest(BaseModel):
    """Request body for creating a new run."""

    name: Optional[str] = None
    config: Optional[dict] = None


class MessageResponse(BaseModel):
    """Generic message response."""

    message: str


class UpdateRunStatusRequest(BaseModel):
    """Request body for updating run status."""

    status: str  # 'running', 'completed', 'failed'


class SelectDatabaseRequest(BaseModel):
    """Request body for selecting a database."""

    db_name: str


class DatabaseInfo(BaseModel):
    """Information about available databases."""

    available_databases: list[str]
    current_database: Optional[str] = None


class RunMetricsResponse(BaseModel):
    """Response containing all metric records for a run."""

    run_id: int
    metrics: dict[str, list[RecordInfo]]


class RunMetricsListResponse(BaseModel):
    """Response containing available metric names for a run."""

    run_id: int
    metrics: list[str]


class RunComputationGraphResponse(BaseModel):
    """Response containing computation graph for a run."""

    run_id: int
    graph: dict


@app.get("/api/databases", response_model=DatabaseInfo)
async def get_databases():
    """Get list of available experiment databases."""
    init_exp_dir()
    db_files = list(all_exp_dir.glob("*.db"))
    available_databases = [f.name for f in db_files]
    current_db = os.environ.get("SG_DB_NAME", "experiment.db")
    return DatabaseInfo(available_databases=available_databases, current_database=current_db)


@app.post("/api/databases/select", response_model=MessageResponse)
async def select_database(request: SelectDatabaseRequest):
    """Select an experiment database."""
    if set_exp_db(request.db_name):
        os.environ["SG_DB_NAME"] = request.db_name
        return MessageResponse(message=f"Database {request.db_name} selected")
    else:
        raise HTTPException(status_code=404, detail=f"Database {request.db_name} not found")


@app.get("/api/runs", response_model=list[RunInfo])
async def get_runs():
    """List all runs of the experiment."""
    return get_exp_db().get_all_runs()


@app.post("/api/runs", response_model=RunInfo)
async def create_run(run_data: CreateRunRequest):
    """Create a new training run."""
    run_id = get_exp_db().create_run(name=run_data.name, config=run_data.config)
    run = get_exp_db().get_run(run_id)
    if run is None:
        raise HTTPException(status_code=500, detail="Failed to create run")
    return run


@app.get("/api/runs/{run_id}", response_model=RunInfo)
async def get_run(run_id: int):
    """Get details for a specific run."""
    run = get_exp_db().get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return run


@app.delete("/api/runs/{run_id}", response_model=MessageResponse)
async def delete_run(run_id: int):
    """Delete a training run and all its data."""
    run = get_exp_db().get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    get_exp_db().delete_run(run_id)
    return MessageResponse(message=f"Run {run_id} deleted")


@app.patch("/api/runs/{run_id}/status", response_model=MessageResponse)
async def update_run_status(run_id: int, request: UpdateRunStatusRequest):
    """Update the status of a run."""
    run = get_exp_db().get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    get_exp_db().update_run_status(run_id, request.status)
    return MessageResponse(message=f"Run {run_id} status updated to {request.status}")


@app.get("/api/runs/{run_id}/records", response_model=RunMetricsResponse)
async def get_records(run_id: int, metric_name: Optional[str] = None):
    """Get metric records for a specific run."""
    run = get_exp_db().get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    if metric_name:
        records = get_exp_db().get_records(run_id, metric_name)
        metrics = {metric_name: records}
    else:
        metric_names = get_exp_db().get_metrics(run_id)
        metrics = {name: get_exp_db().get_records(run_id, name) for name in metric_names}

    return RunMetricsResponse(run_id=run_id, metrics=metrics)


@app.get("/api/runs/{run_id}/metrics", response_model=RunMetricsListResponse)
async def get_metrics(run_id: int):
    """Get list of available metrics for a run."""
    run = get_exp_db().get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    names = get_exp_db().get_metrics(run_id)
    return RunMetricsListResponse(run_id=run_id, metrics=names)


@app.get("/api/runs/{run_id}/graph", response_model=RunComputationGraphResponse)
async def get_graph(run_id: int):
    """Get computation graph for a run."""
    run = get_exp_db().get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    graph = get_exp_db().get_comp_graph(run_id)
    if graph is None:
        raise HTTPException(status_code=404, detail=f"No graph found for run {run_id}")

    return RunComputationGraphResponse(run_id=run_id, graph=graph)


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


SIMPLEBOARD_DIST = Path(__file__).parent / "app" / "dist"
print("simpleboard dist:", SIMPLEBOARD_DIST)

if SIMPLEBOARD_DIST.exists():
    app.mount("/assets", StaticFiles(directory=SIMPLEBOARD_DIST / "assets"), name="assets")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(SIMPLEBOARD_DIST / "index.html")

    @app.get("/{path:path}")
    async def serve_frontend_routes(path: str):
        file_path = SIMPLEBOARD_DIST / path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(SIMPLEBOARD_DIST / "index.html")

else:
    print("Warning: simpleboard frontend not found. Please build the web app.")
    print("Obtained path:", SIMPLEBOARD_DIST)
