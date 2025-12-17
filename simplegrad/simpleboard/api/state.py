"""Global state management for the simpleboard server."""

import os
from pathlib import Path
from typing import Optional

from simplegrad.track import ExperimentDBManager


# Global state
all_exp_dir: Optional[Path] = None  # Directory containing all experiment databases
exp_db: Optional[ExperimentDBManager] = None  # Manager instance of currently selected experiment database
exp_db_name: Optional[str] = None  # Name of the currently selected experiment database


def init_all_exp_dir():
    """Initialize the experiments directory."""
    global all_exp_dir
    if all_exp_dir is None:
        all_exp_dir = Path(os.environ.get("SG_EXPERIMENTS_DIR"))
        if all_exp_dir is None:
            raise ValueError("Environment variable SG_EXPERIMENTS_DIR not set")
        all_exp_dir.mkdir(parents=True, exist_ok=True)


def set_exp_db(db_name: str) -> bool:
    """Switch to a different experiment database. Returns True if successful."""
    global exp_db, exp_db_name
    init_all_exp_dir()
    db_path = all_exp_dir / db_name
    if not db_path.exists():
        return False
    exp_db = ExperimentDBManager(db_path=db_path)
    if not exp_db.check_connection():
        exp_db = None
        exp_db_name = None
        return False
    exp_db_name = db_name
    return True
