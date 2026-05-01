"""CLI launcher for the simpleboard."""

import argparse
import os
import threading
import time
import webbrowser


def main():
    parser = argparse.ArgumentParser(description="Launch simpleboard.")
    parser.add_argument(
        "--port", "-p", type=int, default=8000, help="Port to run the server on (default: 8000)"
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--all-exp-dir",
        "-e",
        type=str,
        default="./experiments",
        help="Directory containing experiment databases (default: ./experiments)",
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="Don't automatically open browser"
    )

    args = parser.parse_args()

    os.environ["SG_EXPERIMENTS_DIR"] = args.all_exp_dir

    from .server import make_server

    url = f"http://{args.host}:{args.port}"
    print(f"Starting simpleboard...")
    print(f"Experiments directory: {args.all_exp_dir}")
    print(f"Server URL: {url}")

    if not args.no_browser:
        def _open():
            time.sleep(1.0)
            webbrowser.open(url)

        threading.Thread(target=_open, daemon=True).start()

    server = make_server(args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
