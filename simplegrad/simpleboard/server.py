"""HTTP server for the simpleboard visualization dashboard."""

import dataclasses
import json
import mimetypes
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .api import state

DIST = Path(__file__).parent / "app" / "dist"


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path == "/api/databases":
            state.init_all_exp_dir()
            db_files = list(state.all_exp_dir.glob("*.db"))
            self._json({
                "available_databases": [f.name for f in db_files],
                "current_database": state.exp_db_name,
            })

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
                if metric_name:
                    records = state.exp_db.get_records(run_id, metric_name)
                    metrics = {metric_name: [dataclasses.asdict(r) for r in records]}
                else:
                    names = state.exp_db.get_metrics(run_id)
                    metrics = {
                        name: [dataclasses.asdict(r) for r in state.exp_db.get_records(run_id, name)]
                        for name in names
                    }
                self._json({"run_id": run_id, "metrics": metrics})

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

        else:
            self._serve_static(path)

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
                run_id = state.exp_db.create_run(
                    name=body.get("name"), config=body.get("config")
                )
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

        if m := re.fullmatch(r"/api/runs/(\d+)/status", path):
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
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status, detail):
        self._json({"detail": detail}, status)

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
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def make_server(host: str, port: int) -> ThreadingHTTPServer:
    """Create and return a configured ThreadingHTTPServer."""
    return ThreadingHTTPServer((host, port), Handler)
