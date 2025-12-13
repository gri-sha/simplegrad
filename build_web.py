#!/usr/bin/env python3
"""
Build script for the SimpleBoard web app.
Run this before installing the Python package to bundle the web app.
"""

import subprocess
import shutil
from pathlib import Path


def build_web_app():
    """Build the React web app using npm."""
    app_dir = Path(__file__).parent / "simplegrad" / "simpleboard" / "app"

    if not app_dir.exists():
        print(f"Error: Web app directory not found at {app_dir}")
        return False

    print(f"Building web app in {app_dir}...")

    # Install dependencies
    print("Installing npm dependencies...")
    result = subprocess.run(["npm", "install"], cwd=app_dir)
    if result.returncode != 0:
        print("Error: npm install failed")
        return False

    # Build the app
    print("Building React app...")
    result = subprocess.run(["npm", "run", "build"], cwd=app_dir)
    if result.returncode != 0:
        print("Error: npm build failed")
        return False

    print("Web app built successfully!")
    return True


if __name__ == "__main__":
    import sys

    success = build_web_app()
    sys.exit(0 if success else 1)
