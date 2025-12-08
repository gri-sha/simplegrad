"""
Shared type definitions for the visualization system.
"""

from typing import Optional
from pydantic import BaseModel


# ===== Database Models =====


class RunInfo(BaseModel):
    """Metadata for a training run."""

    run_id: int
    name: str
    created_at: float
    status: str  # 'running', 'completed', 'failed'
    config: dict


class MetricRecord(BaseModel):
    """A single metric record (data point)."""

    step: int
    value: float
    wall_time: float


# ===== API Request/Response Models =====


class RunCreate(BaseModel):
    """Request body for creating a new run."""

    name: Optional[str] = None
    config: Optional[dict] = None


class RunRecords(BaseModel):
    """Response containing all metric records for a run."""

    run_id: int
    metrics: dict[str, list[MetricRecord]]


class MetricListResponse(BaseModel):
    """Response containing available metric names for a run."""

    run_id: int
    metrics: list[str]


class GraphResponse(BaseModel):
    """Response containing computation graph."""

    run_id: int
    graph: dict


class StatusUpdateRequest(BaseModel):
    """Request body for updating run status."""

    status: str  # 'running', 'completed', 'failed'


class MessageResponse(BaseModel):
    """Generic message response."""

    message: str
