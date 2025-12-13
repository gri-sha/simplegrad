"""
CLI launcher for the simpleboard.
"""

import argparse
import os
import sys
import webbrowser
import time


def main():
    parser = argparse.ArgumentParser(description="Launch simpleboard.")
    parser.add_argument("--port", "-p", type=int, default=8000, help="Port to run the server on (default: 8000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument(
        "--all-exp-dir", "-e", type=str, default="./experiments", help="Directory containing experiment databases (default: ./experiments)"
    )
    parser.add_argument("--no-browser", action="store_true", help="Don't automatically open browser")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")

    args = parser.parse_args()

    # Set environment variables for the server
    os.environ["SG_EXPERIMENTS_DIR"] = args.all_exp_dir

    # Import uvicorn here to avoid import errors if not installed
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn is not installed.")
        print("Install it with: pip install uvicorn")
        sys.exit(1)

    # Check if FastAPI is available
    try:
        from fastapi import FastAPI
    except ImportError:
        print("Error: fastapi is not installed.")
        print("Install it with: pip install fastapi")
        sys.exit(1)

    url = f"http://{args.host}:{args.port}"
    print(f"Starting simpleboard...")
    print(f"All experiments directory: {args.all_exp_dir}")
    print(f"Server URL: {url}")

    # Open browser after a short delay
    if not args.no_browser:

        def open_browser():
            time.sleep(1.5)
            webbrowser.open(url)

        import threading

        threading.Thread(target=open_browser, daemon=True).start()

    # Run the server
    uvicorn.run("simplegrad.simpleboard.server:app", host=args.host, port=args.port, reload=args.reload, log_level="info")


if __name__ == "__main__":
    main()
