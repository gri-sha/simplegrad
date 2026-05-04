"""Benchmark dashboard server — stdlib only.

Usage:
    python benchmarks/dashboard.py
    python benchmarks/dashboard.py --port 8080 --logs-dir benchmarks/logs
"""

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

_DASHBOARD_HTML = Path(__file__).with_name("dashboard.html")


class _Handler(BaseHTTPRequestHandler):
    logs_dir: Path  # injected via make_server()

    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/":
            body = _DASHBOARD_HTML.read_bytes()
            self._respond(200, "text/html; charset=utf-8", body)

        elif path == "/api/runs":
            runs = []
            for f in self.logs_dir.glob("*.json"):
                try:
                    data = json.loads(f.read_text())
                    runs.append(
                        {
                            "name": f.stem,
                            "file": f.name,
                            "suite": data.get("suite", ""),
                            "timestamp": data.get("timestamp", ""),
                        }
                    )
                except Exception:
                    pass
            runs.sort(key=lambda r: r["timestamp"], reverse=True)
            self._json(runs)

        elif m := re.fullmatch(r"/api/runs/([^/]+\.json)", path):
            filename = m.group(1)
            if "/" in filename or "\\" in filename or filename.startswith("."):
                self._error(400, "Invalid filename")
                return
            target = self.logs_dir / filename
            if not target.exists():
                self._error(404, f"{filename} not found")
                return
            self._respond(200, "application/json", target.read_bytes())

        else:
            self._error(404, "Not found")

    def _respond(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data, status: int = 200) -> None:
        body = json.dumps(data).encode()
        self._respond(status, "application/json", body)

    def _error(self, status: int, detail: str) -> None:
        self._json({"detail": detail}, status)

    def log_message(self, *args):
        pass  # suppress access log


def make_server(host: str, port: int, logs_dir: Path) -> ThreadingHTTPServer:
    """Create a configured ThreadingHTTPServer for the benchmark dashboard.

    Args:
        host: Hostname or IP to bind to.
        port: Port to listen on.
        logs_dir: Directory containing ``*.json`` benchmark result files.

    Returns:
        Ready-to-serve :class:`ThreadingHTTPServer` instance.
    """

    class Handler(_Handler):
        pass

    Handler.logs_dir = logs_dir
    return ThreadingHTTPServer((host, port), Handler)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--logs-dir", type=str, default="benchmarks/logs")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    if not logs_dir.exists():
        print(f"Logs directory not found: {logs_dir.resolve()}")
        print("Run a benchmark first to generate results.")
        return

    server = make_server(args.host, args.port, logs_dir)
    url = f"http://{args.host}:{args.port}"
    print(f"Benchmark dashboard  {url}")
    print(f"Logs directory       {logs_dir.resolve()}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
