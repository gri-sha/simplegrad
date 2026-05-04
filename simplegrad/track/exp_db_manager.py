"""
Storage layer for persisting training runs and metrics.
Uses SQLite for simplicity and portability.
"""

import sqlite3
import json
import time
from dataclasses import dataclass
from datetime import datetime
from contextlib import contextmanager
from pathlib import Path


def _format_timestamp(timestamp: float) -> str:
    """Convert Unix timestamp to readable datetime string."""
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class RunInfo:
    """Metadata for a training run."""

    run_id: int
    name: str
    created_at: str  # Formatted datetime string for display
    status: str  # 'running', 'completed', 'failed'
    config: dict
    num_records: list[int] | None = None
    metrics: list[str] | None = None


@dataclass
class RecordInfo:
    """A single metric record (data point)."""

    step: int
    value: float
    log_time: float


def _build_run_info(conn, row) -> RunInfo:
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

    return RunInfo(
        run_id=row["id"],
        name=row["name"],
        created_at=_format_timestamp(row["created_at"]),
        status=row["status"],
        config=json.loads(row["config"]),
        metrics=metrics,
        num_records=num_records,
    )


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
            conn.executescript("""
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
                CREATE TABLE IF NOT EXISTS histograms (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    step INTEGER NOT NULL,
                    bucket_edges TEXT NOT NULL,
                    bucket_counts TEXT NOT NULL,
                    wall_time REAL NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES runs(id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_histograms_run_name ON histograms(run_id, name);

                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    step INTEGER NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    channels INTEGER NOT NULL,
                    image_data BLOB NOT NULL,
                    wall_time REAL NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES runs(id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_images_run_name ON images(run_id, name);

                CREATE TABLE IF NOT EXISTS texts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    step INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    wall_time REAL NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES runs(id)
                );

                CREATE INDEX IF NOT EXISTS idx_texts_run_name ON texts(run_id, name);
            """)

    def create_run(self, name: str | None = None, config: dict | None = None) -> int:
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

    def get_run(self, run_id: int) -> RunInfo | None:
        """Get run metadata."""
        with self._get_connection(readonly=True) as conn:
            row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            if row:
                return _build_run_info(conn, row)
        return None

    def get_all_runs(self) -> list[RunInfo]:
        """List all runs, newest first."""
        with self._get_connection(readonly=True) as conn:
            rows = conn.execute("SELECT * FROM runs ORDER BY created_at DESC").fetchall()
            return [_build_run_info(conn, row) for row in rows]

    def update_run_status(self, run_id: int, status: str):
        """Update run status."""
        with self._get_connection() as conn:
            conn.execute("UPDATE runs SET status = ? WHERE id = ?", (status, run_id))

    def delete_run(self, run_id: int):
        """Delete a run and all its data."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM records WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM graphs WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM histograms WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM images WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM texts WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))

    def record(self, run_id: int, metric_name: str, step: int, value: float):
        """Log a single metric record."""
        log_time = time.time()

        with self._get_connection() as conn:
            conn.execute("INSERT OR IGNORE INTO metrics (name) VALUES (?)", (metric_name,))
            metric_id = conn.execute(
                "SELECT id FROM metrics WHERE name = ?", (metric_name,)
            ).fetchone()["id"]
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

            result: list[RecordInfo] = [
                RecordInfo(step=row["step"], value=row["value"], log_time=row["wall_time"])
                for row in rows
            ]
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

    def get_comp_graph(self, graph_id: int) -> dict | None:
        """Get a single computation graph by its ID."""
        with self._get_connection(readonly=True) as conn:
            row = conn.execute("SELECT graph_json FROM graphs WHERE id = ?", (graph_id,)).fetchone()
            if row:
                return json.loads(row["graph_json"])
        return None

    def get_comp_graphs(self, run_id: int) -> list[dict]:
        """Get all computation graphs for a run."""
        with self._get_connection(readonly=True) as conn:
            rows = conn.execute(
                "SELECT id, graph_json, created_at FROM graphs WHERE run_id = ? ORDER BY created_at",
                (run_id,),
            ).fetchall()
            return [
                {
                    "id": row["id"],
                    "graph": json.loads(row["graph_json"]),
                    "created_at": row["created_at"],
                }
                for row in rows
            ]

    def save_histogram(
        self, run_id: int, name: str, step: int, bucket_edges: list[float], bucket_counts: list[int]
    ):
        """Save a histogram."""
        log_time = time.time()
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO histograms (run_id, name, step, bucket_edges, bucket_counts, wall_time) VALUES (?, ?, ?, ?, ?, ?)",
                (run_id, name, step, json.dumps(bucket_edges), json.dumps(bucket_counts), log_time),
            )

    def get_histograms(self, run_id: int) -> dict[str, list[dict]]:
        """Get all histograms for a run."""
        with self._get_connection(readonly=True) as conn:
            rows = conn.execute(
                "SELECT name, step, bucket_edges, bucket_counts, wall_time FROM histograms WHERE run_id = ? ORDER BY step",
                (run_id,),
            ).fetchall()
            result = {}
            for row in rows:
                name = row["name"]
                if name not in result:
                    result[name] = []
                result[name].append(
                    {
                        "step": row["step"],
                        "bucket_edges": json.loads(row["bucket_edges"]),
                        "bucket_counts": json.loads(row["bucket_counts"]),
                        "log_time": row["wall_time"],
                    }
                )
            return result

    def save_image(
        self,
        run_id: int,
        name: str,
        step: int,
        width: int,
        height: int,
        channels: int,
        image_data: bytes,
    ):
        """Save raw image data."""
        log_time = time.time()
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO images (run_id, name, step, width, height, channels, image_data, wall_time) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, name, step, width, height, channels, image_data, log_time),
            )

    def save_text(self, run_id: int, name: str, step: int, content: str):
        """Save a text entry for a run.

        Use this to record arbitrary string artifacts associated with a step:
        sampled model outputs, validation predictions, free-form notes, model
        summaries, etc. Mirrors TensorBoard's `add_text` semantics.
        """
        log_time = time.time()
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO texts (run_id, name, step, content, wall_time) VALUES (?, ?, ?, ?, ?)",
                (run_id, name, step, content, log_time),
            )

    def get_texts(self, run_id: int) -> dict[str, list[dict]]:
        """Get all text entries for a run, grouped by name."""
        with self._get_connection(readonly=True) as conn:
            rows = conn.execute(
                "SELECT name, step, content, wall_time FROM texts WHERE run_id = ? ORDER BY step",
                (run_id,),
            ).fetchall()
            result: dict[str, list[dict]] = {}
            for row in rows:
                result.setdefault(row["name"], []).append(
                    {
                        "step": row["step"],
                        "content": row["content"],
                        "log_time": row["wall_time"],
                    }
                )
            return result

    def get_metric_summary(self, run_id: int) -> dict[str, dict]:
        """Compute per-metric summary stats (min/max/mean/std/last/n) for a run.

        Returns a mapping from metric name to a dict with the standard
        TensorBoard scalar-summary fields. Cheap because it's a single SQL pass
        per metric and the records table is indexed on (run_id, step).
        """
        with self._get_connection(readonly=True) as conn:
            metrics = [
                row["name"]
                for row in conn.execute(
                    """SELECT DISTINCT m.name FROM records r
                       JOIN metrics m ON r.metric_id = m.id
                       WHERE r.run_id = ?""",
                    (run_id,),
                ).fetchall()
            ]
            summary: dict[str, dict] = {}
            for name in metrics:
                row = conn.execute(
                    """SELECT MIN(r.value) AS vmin, MAX(r.value) AS vmax,
                              AVG(r.value) AS vmean, COUNT(*) AS n,
                              MIN(r.step) AS first_step, MAX(r.step) AS last_step
                       FROM records r JOIN metrics m ON r.metric_id = m.id
                       WHERE r.run_id = ? AND m.name = ?""",
                    (run_id, name),
                ).fetchone()
                last_row = conn.execute(
                    """SELECT r.value FROM records r JOIN metrics m ON r.metric_id = m.id
                       WHERE r.run_id = ? AND m.name = ?
                       ORDER BY r.step DESC LIMIT 1""",
                    (run_id, name),
                ).fetchone()
                # SQLite has no built-in stddev — pull values once for std calc.
                # Cheap for typical metric counts (thousands), and avoids loading
                # an extension just for a single aggregate.
                vals = [
                    r["value"]
                    for r in conn.execute(
                        """SELECT r.value FROM records r JOIN metrics m ON r.metric_id = m.id
                           WHERE r.run_id = ? AND m.name = ?""",
                        (run_id, name),
                    ).fetchall()
                ]
                if vals:
                    mean = sum(vals) / len(vals)
                    var = sum((v - mean) ** 2 for v in vals) / len(vals)
                    std = var**0.5
                else:
                    std = 0.0
                summary[name] = {
                    "min": row["vmin"],
                    "max": row["vmax"],
                    "mean": row["vmean"],
                    "std": std,
                    "last": last_row["value"] if last_row else None,
                    "n": row["n"],
                    "first_step": row["first_step"],
                    "last_step": row["last_step"],
                }
            return summary

    def get_images(self, run_id: int) -> dict[str, list[dict]]:
        """Get all images for a run."""
        with self._get_connection(readonly=True) as conn:
            rows = conn.execute(
                "SELECT name, step, width, height, channels, image_data, wall_time FROM images WHERE run_id = ? ORDER BY step",
                (run_id,),
            ).fetchall()
            result = {}
            import base64

            for row in rows:
                name = row["name"]
                if name not in result:
                    result[name] = []
                result[name].append(
                    {
                        "step": row["step"],
                        "width": row["width"],
                        "height": row["height"],
                        "channels": row["channels"],
                        "data_b64": base64.b64encode(row["image_data"]).decode("ascii"),
                        "log_time": row["wall_time"],
                    }
                )
            return result
