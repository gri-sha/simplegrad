"""CLI launcher for the simpleboard."""

import argparse
import os
import socket
import sys
import threading
import time
import webbrowser
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _get_version() -> str:
    try:
        return version("simplegrad")
    except PackageNotFoundError:
        return "unknown"


def _local_ips() -> list[str]:
    """Return non-loopback IPv4 addresses for LAN access hints."""
    ips: list[str] = []
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = info[4][0]
            if ip and not ip.startswith("127.") and ip not in ips:
                ips.append(ip)
    except OSError:
        pass
    return ips


def _summarize_experiments(exp_dir: Path) -> None:
    """Print a brief summary of what's in the experiments directory."""
    if not exp_dir.exists():
        print(f"  Note: {exp_dir} does not exist yet — it will be created on first run.")
        return
    db_files = sorted(exp_dir.glob("*.db"))
    if not db_files:
        print(f"  No experiment databases found in {exp_dir}.")
        print(f"  Tip: use simplegrad.Tracker(...) to record runs into this folder.")
        return
    print(f"  Found {len(db_files)} database(s):")
    for db in db_files[:10]:
        size_kb = db.stat().st_size / 1024
        print(f"    - {db.name} ({size_kb:,.1f} KB)")
    if len(db_files) > 10:
        print(f"    ... and {len(db_files) - 10} more")


def _frontend_built() -> bool:
    dist = Path(__file__).parent / "app" / "dist" / "index.html"
    return dist.exists()


def _bind_with_fallback(host: str, requested_port: int):
    """Try the requested port, falling back to alternatives if blocked."""
    from .server import make_server

    candidate_ports = [requested_port] + [
        p for p in (8001, 8080, 8765, 8888, 0) if p != requested_port
    ]
    last_err: OSError | None = None
    for port in candidate_ports:
        try:
            server = make_server(host, port)
            return server, server.server_address[1]
        except OSError as e:
            last_err = e
            print(f"  Could not bind to port {port}: {e}")
            # WinError 10013: socket forbidden — typically Hyper-V/WinNAT reserved range.
            if sys.platform == "win32" and getattr(e, "winerror", None) == 10013:
                print(
                    "  (Windows reserves port ranges for Hyper-V/WinNAT. To list them run:\n"
                    "     netsh interface ipv4 show excludedportrange protocol=tcp)"
                )
    raise SystemExit(f"Failed to bind any port. Last error: {last_err}")


def main():
    parser = argparse.ArgumentParser(
        description="Launch simpleboard, the simplegrad experiment dashboard."
    )
    parser.add_argument(
        "--port", "-p", type=int, default=8000, help="Port to run the server on (default: 8000)"
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Bind to 0.0.0.0 so the dashboard is reachable from other devices on your LAN",
    )
    parser.add_argument(
        "--all-exp-dir",
        "-e",
        type=str,
        default="./experiments",
        help="Directory containing experiment databases (default: ./experiments)",
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="Don't automatically open the browser"
    )
    parser.add_argument(
        "--version", "-V", action="version", version=f"simpleboard {_get_version()}"
    )

    args = parser.parse_args()

    host = "0.0.0.0" if args.public else args.host
    exp_dir = Path(args.all_exp_dir).resolve()
    os.environ["SG_EXPERIMENTS_DIR"] = str(exp_dir)

    print(f"simpleboard {_get_version()}")
    print(f"  Experiments directory: {exp_dir}")
    _summarize_experiments(exp_dir)

    if not _frontend_built():
        print(
            "  Warning: frontend bundle not found in simpleboard/app/dist.\n"
            "           Build it with: python scripts/build_web.py"
        )

    server, bound_port = _bind_with_fallback(host, args.port)

    # Pick a friendly URL: if bound to 0.0.0.0, use the loopback for the local link
    # and additionally print LAN URLs so the user can connect from other devices.
    local_url = f"http://127.0.0.1:{bound_port}" if host == "0.0.0.0" else f"http://{host}:{bound_port}"
    print(f"  Local:   {local_url}")
    if host == "0.0.0.0":
        for ip in _local_ips():
            print(f"  Network: http://{ip}:{bound_port}")
    print("  Press Ctrl+C to stop.")

    if not args.no_browser:

        def _open():
            time.sleep(1.0)
            webbrowser.open(local_url)

        threading.Thread(target=_open, daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down simpleboard...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
