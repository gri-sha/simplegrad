"""Pydantic models for API requests and responses."""

from pydantic import BaseModel

from simplegrad.track import RecordInfo


class CreateRunRequest(BaseModel):
    """Request body for creating a new run."""

    name: str | None = None
    config: dict | None = None


class MessageResponse(BaseModel):
    """Generic message response."""

    message: str


class UpdateRunStatusRequest(BaseModel):
    """Request body for updating run status."""

    status: str  # 'running', 'completed', 'failed'


class SelectDBRequest(BaseModel):
    """Request body for selecting a database."""

    db_name: str


class DBsResponse(BaseModel):
    """Information about available databases."""

    available_databases: list[str]
    current_database: str | None = None


class MetricsResponse(BaseModel):
    """Response containing all metric records for a run."""

    run_id: int
    metrics: dict[str, list[RecordInfo]]


class MetricNamesResponse(BaseModel):
    """Response containing available metric names for a run."""

    run_id: int
    metrics: list[str]


class CompGraphsResponse(BaseModel):
    """Response containing computation graphs for a run."""

    run_id: int
    graphs: list[dict]
