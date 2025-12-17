"""REST API routes for simpleboard."""

from typing import Optional

from fastapi import APIRouter, HTTPException

from simplegrad.track import RunInfo

from . import state
from .models import (
    CreateRunRequest,
    MessageResponse,
    UpdateRunStatusRequest,
    SelectDBRequest,
    DBsResponse,
    MetricsResponse,
    MetricNamesResponse,
    CompGraphsResponse,
)

router = APIRouter(prefix="/api")


# Database Routes


@router.get("/databases", response_model=DBsResponse)
async def get_databases():
    """Get list of available experiment databases."""
    state.init_all_exp_dir()
    db_files = list(state.all_exp_dir.glob("*.db"))
    print(state.all_exp_dir)
    print(db_files)
    available_databases = [f.name for f in db_files]
    print(available_databases)
    return DBsResponse(available_databases=available_databases, current_database=state.exp_db_name)


@router.post("/databases/select", response_model=MessageResponse)
async def select_database(request: SelectDBRequest):
    """Select an experiment database."""
    if state.set_exp_db(request.db_name):
        return MessageResponse(message=f"Database {request.db_name} selected")
    else:
        raise HTTPException(status_code=404, detail=f"Database {request.db_name} not found")


# Run Routes


@router.get("/runs", response_model=list[RunInfo])
async def get_runs():
    """List all runs of the experiment."""
    if state.exp_db is None:
        return []
    return state.exp_db.get_all_runs()


@router.post("/runs", response_model=RunInfo)
async def create_run(run_data: CreateRunRequest):
    """Create a new training run."""
    if state.exp_db is None:
        raise HTTPException(status_code=400, detail="No database selected")
    run_id = state.exp_db.create_run(name=run_data.name, config=run_data.config)
    run = state.exp_db.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=500, detail="Failed to create run")
    return run


@router.get("/runs/{run_id}", response_model=RunInfo)
async def get_run(run_id: int):
    """Get details for a specific run."""
    if state.exp_db is None:
        raise HTTPException(status_code=400, detail="No database selected")
    run = state.exp_db.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return run


@router.delete("/runs/{run_id}", response_model=MessageResponse)
async def delete_run(run_id: int):
    """Delete a training run and all its data."""
    if state.exp_db is None:
        raise HTTPException(status_code=400, detail="No database selected")
    run = state.exp_db.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    state.exp_db.delete_run(run_id)
    return MessageResponse(message=f"Run {run_id} deleted")


@router.patch("/runs/{run_id}/status", response_model=MessageResponse)
async def update_run_status(run_id: int, request: UpdateRunStatusRequest):
    """Update the status of a run."""
    if state.exp_db is None:
        raise HTTPException(status_code=400, detail="No database selected")
    run = state.exp_db.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    state.exp_db.update_run_status(run_id, request.status)
    return MessageResponse(message=f"Run {run_id} status updated to {request.status}")


# Metrics Routes


@router.get("/runs/{run_id}/records", response_model=MetricsResponse)
async def get_records(run_id: int, metric_name: Optional[str] = None):
    """Get metric records for a specific run."""
    if state.exp_db is None:
        raise HTTPException(status_code=400, detail="No database selected")
    run = state.exp_db.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    if metric_name:
        records = state.exp_db.get_records(run_id, metric_name)
        metrics = {metric_name: records}
    else:
        metric_names = state.exp_db.get_metrics(run_id)
        metrics = {name: state.exp_db.get_records(run_id, name) for name in metric_names}

    return MetricsResponse(run_id=run_id, metrics=metrics)


@router.get("/runs/{run_id}/metrics", response_model=MetricNamesResponse)
async def get_metrics(run_id: int):
    """Get list of available metrics for a run."""
    if state.exp_db is None:
        raise HTTPException(status_code=400, detail="No database selected")
    run = state.exp_db.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    names = state.exp_db.get_metrics(run_id)
    return MetricNamesResponse(run_id=run_id, metrics=names)


# Graph Routes


@router.get("/runs/{run_id}/graphs", response_model=CompGraphsResponse)
async def get_graphs(run_id: int):
    """Get computation graphs of a run."""
    if state.exp_db is None:
        raise HTTPException(status_code=400, detail="No database selected")

    graphs = state.exp_db.get_comp_graphs(run_id)
    return CompGraphsResponse(run_id=run_id, graphs=graphs)
