"""HTTP server for the simpleboard visualization dashboard."""

import csv
import dataclasses
import io
import json
import mimetypes
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .api import state

DIST = Path(__file__).parent / "app" / "dist"


def _ema_smooth(values: list[float], smoothing: float) -> list[float]:
    """Exponential moving average with debiasing, matching TensorBoard's
    scalar smoothing. `smoothing` is in [0, 1); 0 means no smoothing."""
    if smoothing <= 0 or not values:
        return list(values)
    smoothing = min(smoothing, 0.999)
    smoothed: list[float] = []
    last = 0.0
    weight = 0.0
    for v in values:
        last = last * smoothing + v * (1 - smoothing)
        weight = weight * smoothing + (1 - smoothing)
        smoothed.append(last / weight if weight > 0 else v)
    return smoothed


def _parse_float(qs: dict, key: str, default: float) -> float:
    raw = qs.get(key, [None])[0]
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _parse_id_list(qs: dict, key: str) -> list[int]:
    raw = qs.get(key, [""])[0]
    out: list[int] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if chunk.isdigit():
            out.append(int(chunk))
    return out


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path == "/api/config":
            state.init_all_exp_dir()
            self._json({"exp_dir": str(state.all_exp_dir) if state.all_exp_dir else ""})

        elif path == "/api/databases":
            try:
                state.init_all_exp_dir()
            except Exception as exc:
                self._error(500, str(exc))
                return
            db_files = list(state.all_exp_dir.glob("*.db"))
            self._json(
                {
                    "available_databases": [f.name for f in db_files],
                    "current_database": state.exp_db_name,
                }
            )

        elif path == "/api/runs":
            if state.exp_db is None:
                self._json([])
            else:
                runs = state.exp_db.get_all_runs()
                self._json([dataclasses.asdict(r) for r in runs])

        elif m := re.fullmatch(r"/api/runs/(\d+)", path):
            run_id = int(m.group(1))
            if state.exp_db is None:
                self._error(400, "No database selected")
            else:
                run = state.exp_db.get_run(run_id)
                if run is None:
                    self._error(404, f"Run {run_id} not found")
                else:
                    self._json(dataclasses.asdict(run))

        elif m := re.fullmatch(r"/api/runs/(\d+)/records", path):
            run_id = int(m.group(1))
            if state.exp_db is None:
                self._error(400, "No database selected")
            elif state.exp_db.get_run(run_id) is None:
                self._error(404, f"Run {run_id} not found")
            else:
                metric_name = qs.get("metric_name", [None])[0]
                smoothing = _parse_float(qs, "smoothing", 0.0)
                if metric_name:
                    names = [metric_name]
                else:
                    names = state.exp_db.get_metrics(run_id)
                metrics = {}
                for name in names:
                    records = state.exp_db.get_records(run_id, name)
                    items = [dataclasses.asdict(r) for r in records]
                    if smoothing > 0 and items:
                        smoothed = _ema_smooth([it["value"] for it in items], smoothing)
                        for it, sv in zip(items, smoothed):
                            it["smoothed"] = sv
                    metrics[name] = items
                self._json({"run_id": run_id, "metrics": metrics, "smoothing": smoothing})

        elif m := re.fullmatch(r"/api/runs/(\d+)/metrics", path):
            run_id = int(m.group(1))
            if state.exp_db is None:
                self._error(400, "No database selected")
            elif state.exp_db.get_run(run_id) is None:
                self._error(404, f"Run {run_id} not found")
            else:
                self._json({"run_id": run_id, "metrics": state.exp_db.get_metrics(run_id)})

        elif m := re.fullmatch(r"/api/runs/(\d+)/graphs", path):
            run_id = int(m.group(1))
            if state.exp_db is None:
                self._error(400, "No database selected")
            else:
                self._json({"run_id": run_id, "graphs": state.exp_db.get_comp_graphs(run_id)})

        elif m := re.fullmatch(r"/api/runs/(\d+)/histograms", path):
            run_id = int(m.group(1))
            if state.exp_db is None:
                self._error(400, "No database selected")
            else:
                self._json({"run_id": run_id, "histograms": state.exp_db.get_histograms(run_id)})

        elif m := re.fullmatch(r"/api/runs/(\d+)/images", path):
            run_id = int(m.group(1))
            if state.exp_db is None:
                self._error(400, "No database selected")
            else:
                self._json({"run_id": run_id, "images": state.exp_db.get_images(run_id)})

        elif m := re.fullmatch(r"/api/runs/(\d+)/texts", path):
            run_id = int(m.group(1))
            if state.exp_db is None:
                self._error(400, "No database selected")
            elif state.exp_db.get_run(run_id) is None:
                self._error(404, f"Run {run_id} not found")
            else:
                self._json({"run_id": run_id, "texts": state.exp_db.get_texts(run_id)})

        elif m := re.fullmatch(r"/api/runs/(\d+)/summary", path):
            run_id = int(m.group(1))
            if state.exp_db is None:
                self._error(400, "No database selected")
            elif state.exp_db.get_run(run_id) is None:
                self._error(404, f"Run {run_id} not found")
            else:
                self._json({"run_id": run_id, "metrics": state.exp_db.get_metric_summary(run_id)})

        elif path == "/api/runs/compare":
            if state.exp_db is None:
                self._error(400, "No database selected")
            else:
                metric = qs.get("metric", [None])[0]
                if not metric:
                    self._error(400, "Query param 'metric' is required")
                else:
                    ids = _parse_id_list(qs, "ids")
                    smoothing = _parse_float(qs, "smoothing", 0.0)
                    runs_out: dict[str, dict] = {}
                    for rid in ids:
                        run = state.exp_db.get_run(rid)
                        if run is None:
                            continue
                        records = state.exp_db.get_records(rid, metric)
                        items = [dataclasses.asdict(r) for r in records]
                        if smoothing > 0 and items:
                            smoothed = _ema_smooth([it["value"] for it in items], smoothing)
                            for it, sv in zip(items, smoothed):
                                it["smoothed"] = sv
                        runs_out[str(rid)] = {"name": run.name, "records": items}
                    self._json({"metric": metric, "smoothing": smoothing, "runs": runs_out})

        elif path == "/api/hparams":
            if state.exp_db is None:
                self._error(400, "No database selected")
            else:
                runs = state.exp_db.get_all_runs()
                hparam_keys: set[str] = set()
                metric_keys: set[str] = set()
                table = []
                for r in runs:
                    summary = state.exp_db.get_metric_summary(r.run_id)
                    metric_keys.update(summary.keys())
                    if isinstance(r.config, dict):
                        hparam_keys.update(r.config.keys())
                    table.append(
                        {
                            "run_id": r.run_id,
                            "name": r.name,
                            "status": r.status,
                            "created_at": r.created_at,
                            "hparams": r.config if isinstance(r.config, dict) else {},
                            "metrics": summary,
                        }
                    )
                self._json(
                    {
                        "hparam_keys": sorted(hparam_keys),
                        "metric_keys": sorted(metric_keys),
                        "runs": table,
                    }
                )

        elif m := re.fullmatch(r"/api/runs/(\d+)/records\.csv", path):
            run_id = int(m.group(1))
            if state.exp_db is None:
                self._error(400, "No database selected")
            elif state.exp_db.get_run(run_id) is None:
                self._error(404, f"Run {run_id} not found")
            else:
                metric_name = qs.get("metric", [None])[0]
                names = [metric_name] if metric_name else state.exp_db.get_metrics(run_id)
                buf = io.StringIO()
                writer = csv.writer(buf)
                writer.writerow(["metric", "step", "value", "wall_time"])
                for name in names:
                    for r in state.exp_db.get_records(run_id, name):
                        writer.writerow([name, r.step, r.value, r.log_time])
                self._send_bytes(
                    buf.getvalue().encode("utf-8"),
                    content_type="text/csv",
                    filename=f"run_{run_id}.csv",
                )

        elif m := re.fullmatch(r"/api/runs/(\d+)/export\.json", path):
            run_id = int(m.group(1))
            if state.exp_db is None:
                self._error(400, "No database selected")
            else:
                run = state.exp_db.get_run(run_id)
                if run is None:
                    self._error(404, f"Run {run_id} not found")
                else:
                    metrics = {
                        name: [
                            dataclasses.asdict(r) for r in state.exp_db.get_records(run_id, name)
                        ]
                        for name in state.exp_db.get_metrics(run_id)
                    }
                    payload = {
                        "run": dataclasses.asdict(run),
                        "metrics": metrics,
                        "summary": state.exp_db.get_metric_summary(run_id),
                        "histograms": state.exp_db.get_histograms(run_id),
                        "texts": state.exp_db.get_texts(run_id),
                        "graphs": state.exp_db.get_comp_graphs(run_id),
                    }
                    body = json.dumps(payload, indent=2).encode("utf-8")
                    self._send_bytes(
                        body,
                        content_type="application/json",
                        filename=f"run_{run_id}.json",
                    )

        else:
            self._serve_static(path)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PATCH, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        path = urlparse(self.path).path

        if path == "/api/databases/select":
            body = self._read_body()
            db_name = body.get("db_name", "")
            if state.set_exp_db(db_name):
                self._json({"message": f"Database {db_name} selected"})
            else:
                self._error(404, f"Database {db_name} not found")

        elif path == "/api/runs":
            if state.exp_db is None:
                self._error(400, "No database selected")
            else:
                body = self._read_body()
                run_id = state.exp_db.create_run(name=body.get("name"), config=body.get("config"))
                run = state.exp_db.get_run(run_id)
                self._json(dataclasses.asdict(run), status=201)

        else:
            self._error(404, "Not found")

    def do_DELETE(self):
        path = urlparse(self.path).path

        if m := re.fullmatch(r"/api/runs/(\d+)", path):
            run_id = int(m.group(1))
            if state.exp_db is None:
                self._error(400, "No database selected")
            elif state.exp_db.get_run(run_id) is None:
                self._error(404, f"Run {run_id} not found")
            else:
                state.exp_db.delete_run(run_id)
                self._json({"message": f"Run {run_id} deleted"})
        else:
            self._error(404, "Not found")

    def do_PATCH(self):
        path = urlparse(self.path).path

        if path == "/api/config/exp-dir":
            body = self._read_body()
            new_path = body.get("path", "").strip()
            if not new_path:
                self._error(400, "'path' field is required")
                return
            try:
                state.update_exp_dir(new_path)
            except Exception as exc:
                self._error(500, str(exc))
                return
            self._json({"exp_dir": str(state.all_exp_dir)})

        elif m := re.fullmatch(r"/api/runs/(\d+)/status", path):
            run_id = int(m.group(1))
            if state.exp_db is None:
                self._error(400, "No database selected")
            elif state.exp_db.get_run(run_id) is None:
                self._error(404, f"Run {run_id} not found")
            else:
                body = self._read_body()
                state.exp_db.update_run_status(run_id, body["status"])
                self._json({"message": f"Run {run_id} status updated to {body['status']}"})
        else:
            self._error(404, "Not found")

    def _json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status, detail):
        self._json({"detail": detail}, status)

    def _send_bytes(self, body: bytes, content_type: str, filename: str | None = None):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length)) if length else {}

    def _serve_static(self, path):
        if not DIST.exists():
            self._error(503, "Frontend not built. Run: python build_web.py")
            return
        rel = path.lstrip("/") or "index.html"
        file_path = DIST / rel
        if not (file_path.exists() and file_path.is_file()):
            file_path = DIST / "index.html"
        mime, _ = mimetypes.guess_type(str(file_path))
        body = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime or "application/octet-stream")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def make_server(host: str, port: int) -> ThreadingHTTPServer:
    """Create and return a configured ThreadingHTTPServer."""
    return ThreadingHTTPServer((host, port), Handler)
