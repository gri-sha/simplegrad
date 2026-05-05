"""Global state management for the simpleboard server."""

import os
from pathlib import Path

from simplegrad.track import ExperimentDBManager

# Global state
all_exp_dir: Path | None = None  # Directory containing all experiment databases
exp_db: ExperimentDBManager | None = (
    None  # Manager instance of currently selected experiment database
)
exp_db_name: str | None = None  # Name of the currently selected experiment database


def init_all_exp_dir():
    """Initialize the experiments directory from the SG_EXPERIMENTS_DIR env var."""
    global all_exp_dir
    if all_exp_dir is not None:
        return
    env_val = os.environ.get("SG_EXPERIMENTS_DIR")
    if not env_val:
        raise RuntimeError(
            "SG_EXPERIMENTS_DIR is not set. "
            "Launch simpleboard through the CLI: simpleboard --all-exp-dir <path>"
        )
    all_exp_dir = Path(env_val)
    all_exp_dir.mkdir(parents=True, exist_ok=True)


def update_exp_dir(new_path: str) -> None:
    """Switch the experiments directory to a new path at runtime."""
    global all_exp_dir, exp_db, exp_db_name
    p = Path(new_path).resolve()
    p.mkdir(parents=True, exist_ok=True)
    all_exp_dir = p
    os.environ["SG_EXPERIMENTS_DIR"] = str(p)
    exp_db = None
    exp_db_name = None


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
    # Run the schema creation so any tables added since this DB was first
    # written (e.g. histograms, images) exist before we start serving queries.
    # All CREATE statements use IF NOT EXISTS, so this is safe for both fresh
    # and pre-existing databases.
    exp_db.init_exp_db()
    exp_db_name = db_name
    return True
