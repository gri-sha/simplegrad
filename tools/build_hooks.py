"""Custom setuptools hooks to build the SimpleBoard frontend during installation."""

import os
import subprocess
import sys
from pathlib import Path

from setuptools.command.build_py import build_py as _build_py


class BuildPyCommand(_build_py):
    """Custom build_py that also compiles the SimpleBoard React app."""

    def run(self):
        if os.environ.get("SIMPLEGRAD_NO_BUILD_WEB") == "1":
            print("Skipping SimpleBoard web app build (SIMPLEGRAD_NO_BUILD_WEB=1)")
        else:
            build_script = Path(__file__).parent / "build_simpleboard.py"
            if build_script.exists():
                print("Building SimpleBoard web app...")
                result = subprocess.run(
                    [sys.executable, str(build_script)],
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    print("WARNING: SimpleBoard web app build failed.")
                    if result.stderr:
                        print(result.stderr)
                    print("The package will be installed but SimpleBoard may not work.")
                    print("To skip this step, set SIMPLEGRAD_NO_BUILD_WEB=1.")
                else:
                    print("SimpleBoard web app built successfully.")
            else:
                print(f"WARNING: Build script not found at {build_script}")
        super().run()
