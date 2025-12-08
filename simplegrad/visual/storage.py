"""
Storage layer for persisting training runs and metrics.
Uses SQLite for simplicity and portability.
"""

import sqlite3
import json
import time
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

from .types import RunInfo, MetricRecord


class RunStorage:
    """SQLite-based storage for training runs and metrics."""

    def __init__(self, logdir: str = "./experiment"):
        self.logdir = Path(logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.logdir / "experiment.db"
        self._init_db()

    def _init_db(self):
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

    @contextmanager
    def _get_connection(self):
        """Get a database connection with proper cleanup."""
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

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
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            if row:
                return RunInfo(
                    run_id=row["id"],
                    name=row["name"],
                    created_at=row["created_at"],
                    status=row["status"],
                    config=json.loads(row["config"]),
                )
        return None

    def list_runs(self) -> list[RunInfo]:
        """List all runs, newest first."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM runs ORDER BY created_at DESC").fetchall()
            return [
                RunInfo(
                    run_id=row["id"],
                    name=row["name"],
                    created_at=row["created_at"],
                    status=row["status"],
                    config=json.loads(row["config"]),
                )
                for row in rows
            ]

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

    def log_record(self, run_id: int, metric_name: str, step: int, value: float):
        """Log a single metric record."""
        wall_time = time.time()

        with self._get_connection() as conn:
            cursor = conn.execute("INSERT OR IGNORE INTO metrics (name) VALUES (?)", (metric_name,))
            metric_id = conn.execute("SELECT id FROM metrics WHERE name = ?", (metric_name,)).fetchone()["id"]
            conn.execute(
                "INSERT INTO records (run_id, metric_id, step, value, wall_time) VALUES (?, ?, ?, ?, ?)",
                (run_id, metric_id, step, value, wall_time),
            )

    def log_records_batch(self, events: list[tuple[int, str, int, float]]):
        """Log multiple metric records efficiently. Each tuple: (run_id, metric_name, step, value)"""
        wall_time = time.time()

        with self._get_connection() as conn:
            unique_metrics = set(metric_name for _, metric_name, _, _ in events)
            for metric_name in unique_metrics:
                conn.execute("INSERT OR IGNORE INTO metrics (name) VALUES (?)", (metric_name,))

            metric_map = {}
            for metric_name in unique_metrics:
                metric_id = conn.execute("SELECT id FROM metrics WHERE name = ?", (metric_name,)).fetchone()["id"]
                metric_map[metric_name] = metric_id

            conn.executemany(
                "INSERT INTO records (run_id, metric_id, step, value, wall_time) VALUES (?, ?, ?, ?, ?)",
                [(run_id, metric_map[metric_name], step, value, wall_time) for run_id, metric_name, step, value in events],
            )

    def get_records(self, run_id: int, metric_name: Optional[str] = None) -> dict[str, list[MetricRecord]]:
        """Get metric records for a run. Returns {metric_name: [MetricRecord, ...]}"""
        with self._get_connection() as conn:
            if metric_name:
                rows = conn.execute(
                    """SELECT m.name as metric_name, r.step, r.value, r.wall_time 
                       FROM records r 
                       JOIN metrics m ON r.metric_id = m.id 
                       WHERE r.run_id = ? AND m.name = ? 
                       ORDER BY r.step""",
                    (run_id, metric_name),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT m.name as metric_name, r.step, r.value, r.wall_time 
                       FROM records r 
                       JOIN metrics m ON r.metric_id = m.id 
                       WHERE r.run_id = ? 
                       ORDER BY m.name, r.step""",
                    (run_id,),
                ).fetchall()

            result: dict[str, list[MetricRecord]] = {}
            for row in rows:
                name = row["metric_name"]
                if name not in result:
                    result[name] = []
                result[name].append(MetricRecord(step=row["step"], value=row["value"], wall_time=row["wall_time"]))

            return result

    def get_metric_names(self, run_id: int) -> list[str]:
        """Get list of metric names for a run."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """SELECT DISTINCT m.name 
                   FROM records r 
                   JOIN metrics m ON r.metric_id = m.id 
                   WHERE r.run_id = ?""",
                (run_id,),
            ).fetchall()
            return [row["name"] for row in rows]

    def get_latest_step(self, run_id: int) -> int:
        """Get the latest step number for a run."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT MAX(step) as max_step FROM records WHERE run_id = ?", (run_id,)).fetchone()
            return row["max_step"] if row["max_step"] is not None else 0

    def save_graph(self, run_id: int, graph_data: dict):
        """Save computation graph as JSON."""
        with self._get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO graphs (run_id, graph_json, created_at) VALUES (?, ?, ?)",
                (run_id, json.dumps(graph_data), time.time()),
            )

    def get_graph(self, run_id: int) -> Optional[dict]:
        """Get computation graph for a run."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT graph_json FROM graphs WHERE run_id = ?", (run_id,)).fetchone()
            if row:
                return json.loads(row["graph_json"])
        return None
