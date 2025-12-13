"""
Storage layer for persisting training runs and metrics.
Uses SQLite for simplicity and portability.
"""

import sqlite3
import json
import time
from datetime import datetime
from typing import Optional
from contextlib import contextmanager
from pathlib import Path
from pydantic import BaseModel


def _format_timestamp(timestamp: float) -> str:
    """Convert Unix timestamp to readable datetime string."""
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


class RunInfo(BaseModel):
    """Metadata for a training run."""

    run_id: int
    name: str
    created_at: str  # Formatted datetime string for display
    status: str  # 'running', 'completed', 'failed'
    config: dict
    num_records: Optional[list[int]] = None
    metrics: Optional[list[str]] = None


class RecordInfo(BaseModel):
    """A single metric record (data point)."""

    step: int
    value: float
    log_time: float


class ExperimentDBManager:
    """SQLite-based storage for training runs and metrics."""

    def __init__(self, db_path: Path):
        self.db_path = db_path

    @contextmanager
    def _get_connection(self, readonly: bool = False):
        """Get a database connection with proper cleanup."""
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        if readonly:
            conn.isolation_level = None  # Autocommit mode
        try:
            yield conn
            if not readonly:
                conn.commit()
        except Exception:
            if not readonly:
                conn.rollback()
            raise
        finally:
            conn.close()

    def check_connection(self) -> bool:
        """Check if the database exists and is accessible."""
        if not self.db_path.exists():
            return False
        try:
            with self._get_connection() as conn:
                conn.execute("SELECT 1 FROM sqlite_master LIMIT 1")
            return True
        except (sqlite3.DatabaseError, sqlite3.OperationalError):
            return False

    def init_exp_db(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    status TEXT NOT NULL DEFAULT 'running' CHECK(status IN ('running', 'completed', 'failed')),
                    config TEXT DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE
                );
                
                CREATE TABLE IF NOT EXISTS records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    metric_id INTEGER NOT NULL,
                    step INTEGER NOT NULL,
                    value REAL NOT NULL,
                    wall_time REAL NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES runs(id),
                    FOREIGN KEY (metric_id) REFERENCES metrics(id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_records_run_step 
                ON records(run_id, step);
                
                CREATE TABLE IF NOT EXISTS graphs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL UNIQUE,
                    graph_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES runs(id)
                );
            """
            )

    def create_run(self, name: Optional[str] = None, config: Optional[dict] = None) -> int:
        """Create a new training run. Returns run_id."""
        created_at = time.time()
        config = config or {}
        name = name or f"run_{int(created_at)}"

        with self._get_connection() as conn:
            cursor = conn.execute(
                "INSERT INTO runs (name, created_at, status, config) VALUES (?, ?, ?, ?)",
                (name, created_at, "running", json.dumps(config)),
            )
            run_id = cursor.lastrowid

        return run_id

    def get_run(self, run_id: int) -> Optional[RunInfo]:
        """Get run metadata."""
        with self._get_connection(readonly=True) as conn:
            row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            if row:
                metrics = None
                num_records = None

                if row["status"] == "completed":
                    # Get metrics for this run
                    metric_rows = conn.execute(
                        """SELECT DISTINCT m.name 
                           FROM records r 
                           JOIN metrics m ON r.metric_id = m.id 
                           WHERE r.run_id = ?""",
                        (run_id,),
                    ).fetchall()
                    metrics = [m["name"] for m in metric_rows]

                    # Get record counts per metric
                    record_count_rows = conn.execute(
                        """SELECT m.name, COUNT(*) as count
                           FROM records r
                           JOIN metrics m ON r.metric_id = m.id
                           WHERE r.run_id = ?
                           GROUP BY m.name""",
                        (run_id,),
                    ).fetchall()
                    num_records = [rc["count"] for rc in record_count_rows]

                return RunInfo(
                    run_id=row["id"],
                    name=row["name"],
                    created_at=_format_timestamp(row["created_at"]),
                    status=row["status"],
                    config=json.loads(row["config"]),
                    metrics=metrics,
                    num_records=num_records,
                )
        return None

    def get_all_runs(self) -> list[RunInfo]:
        """List all runs, newest first."""
        with self._get_connection(readonly=True) as conn:
            rows = conn.execute("SELECT * FROM runs ORDER BY created_at DESC").fetchall()
            runs = []
            for row in rows:
                metrics = None
                num_records = None

                if row["status"] == "completed":
                    # Get metrics for this run
                    metric_rows = conn.execute(
                        """SELECT DISTINCT m.name 
                           FROM records r 
                           JOIN metrics m ON r.metric_id = m.id 
                           WHERE r.run_id = ?""",
                        (row["id"],),
                    ).fetchall()
                    metrics = [m["name"] for m in metric_rows]

                    # Get record counts per metric
                    record_count_rows = conn.execute(
                        """SELECT m.name, COUNT(*) as count
                           FROM records r
                           JOIN metrics m ON r.metric_id = m.id
                           WHERE r.run_id = ?
                           GROUP BY m.name""",
                        (row["id"],),
                    ).fetchall()
                    num_records = [rc["count"] for rc in record_count_rows]

                runs.append(
                    RunInfo(
                        run_id=row["id"],
                        name=row["name"],
                        created_at=_format_timestamp(row["created_at"]),
                        status=row["status"],
                        config=json.loads(row["config"]),
                        metrics=metrics,
                        num_records=num_records,
                    )
                )
            return runs

    def update_run_status(self, run_id: int, status: str):
        """Update run status."""
        with self._get_connection() as conn:
            conn.execute("UPDATE runs SET status = ? WHERE id = ?", (status, run_id))

    def delete_run(self, run_id: int):
        """Delete a run and all its data."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM records WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM graphs WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))

    def record(self, run_id: int, metric_name: str, step: int, value: float):
        """Log a single metric record."""
        log_time = time.time()

        with self._get_connection() as conn:
            conn.execute("INSERT OR IGNORE INTO metrics (name) VALUES (?)", (metric_name,))
            metric_id = conn.execute("SELECT id FROM metrics WHERE name = ?", (metric_name,)).fetchone()["id"]
            conn.execute(
                "INSERT INTO records (run_id, metric_id, step, value, wall_time) VALUES (?, ?, ?, ?, ?)",
                (run_id, metric_id, step, value, log_time),
            )

    def get_records(self, run_id: int, metric_name: str) -> list[RecordInfo]:
        """Get metric records for a run. Returns {metric_name: [MetricRecord, ...]}"""
        with self._get_connection(readonly=True) as conn:
            rows = conn.execute(
                """SELECT m.name as metric_name, r.step, r.value, r.wall_time 
                    FROM records r 
                    JOIN metrics m ON r.metric_id = m.id 
                    WHERE r.run_id = ? AND m.name = ? 
                    ORDER BY r.step""",
                (run_id, metric_name),
            ).fetchall()

            result: list[RecordInfo] = [RecordInfo(step=row["step"], value=row["value"], log_time=row["wall_time"]) for row in rows]
            return result

    def get_metrics(self, run_id: int) -> list[str]:
        """Get list of metric names for a run."""
        with self._get_connection(readonly=True) as conn:
            rows = conn.execute(
                """SELECT DISTINCT m.name 
                   FROM records r 
                   JOIN metrics m ON r.metric_id = m.id 
                   WHERE r.run_id = ?""",
                (run_id,),
            ).fetchall()
            return [row["name"] for row in rows]

    def save_comp_graph(self, run_id: int, graph_data: dict):
        """Save computation graph as JSON."""
        with self._get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO graphs (run_id, graph_json, created_at) VALUES (?, ?, ?)",
                (run_id, json.dumps(graph_data), time.time()),
            )

    def get_comp_graphs(self, graph_id: str) -> Optional[dict]:
        """Get computation graph for a run."""
        with self._get_connection(readonly=True) as conn:
            row = conn.execute("SELECT graph_json FROM graphs WHERE run_id = ?", (graph_id,)).fetchone()
            if row:
                return json.loads(row["graph_json"])
        return None

    def get_comp_graph(self, graph_id: str) -> Optional[dict]:
        """Get computation graph for a run."""
        with self._get_connection(readonly=True) as conn:
            row = conn.execute("SELECT graph_json FROM graphs WHERE run_id = ?", (graph_id,)).fetchone()
            if row:
                return json.loads(row["graph_json"])
        return None
