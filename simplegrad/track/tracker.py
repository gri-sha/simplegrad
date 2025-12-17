from pathlib import Path
from typing import Optional
import subprocess
import webbrowser
import time
from .exp_db_manager import ExperimentDBManager, RunInfo, RecordInfo
from .comp_graph import _build_graph_data
from simplegrad.core import Tensor


class Tracker:
    """
    Tracker class to manage experiments and runs using ExperimentDBManager.
    It allows setting experiment, starting and ending runs.

    There is a main directory where all experiment databases are stored.
    One database corresponds to one experiment.
    Each experiment can have multiple runs, each with its own metrics and computational graphs.
    """

    def __init__(self, all_exp_dir: str = "./experiments"):
        self.all_exp_dir = Path(all_exp_dir)
        self.all_exp_dir.mkdir(parents=True, exist_ok=True)  # Create the experiments directory if it doesn't exist

        self.cur_exp_path = None
        self.db_manager = None
        self.current_run_id = None
        self.current_run_name = None

    def get_all_exp_paths(self) -> list[Path]:
        """Get all experiment database paths."""
        if not self.all_exp_dir.exists():
            return []
        return list(self.all_exp_dir.glob("*.db"))

    def set_all_exp_dir(self, directory: str):
        """Set the experiments directory"""
        self.all_experiments_dir = Path(directory)

    def set_experiment(self, exp_name: str):
        """Set the current experiment by name, initializing its database manager."""
        db_name = exp_name if exp_name.endswith(".db") else f"{exp_name}.db"
        self.cur_exp_path = self.all_exp_dir / db_name
        self.db_manager = ExperimentDBManager(db_path=self.cur_exp_path)
        if self.db_manager.check_connection():
            print(f"Connected to existing experiment database at {self.cur_exp_path}")
        else:
            self.db_manager.init_exp_db()

    def start_run(self, name: Optional[str] = None, config: Optional[dict] = None) -> int:
        """Start a new run and return the run_id"""
        self.current_run_id = self.db_manager.create_run(name=name, config=config)
        self.current_run_name = name or f"run_{self.current_run_id}"
        return self.current_run_id

    def record(self, metric_name: str, value: float, step: int):
        """Log a metric value at a given step"""
        if self.current_run_id is None:
            raise RuntimeError("No active run. Call start_run() first.")
        self.db_manager.record(self.current_run_id, metric_name, step, value)

    def end_run(self, status: str = "completed"):
        """End the current run with a given status"""
        if self.current_run_id is None:
            raise RuntimeError("No active run. Call start_run() first.")
        id = self.current_run_id
        self.db_manager.update_run_status(self.current_run_id, status)
        self.current_run_id = None
        self.current_run_name = None
        return id

    def get_all_runs(self) -> list[RunInfo]:
        """Get all runs"""
        return self.db_manager.get_all_runs()

    def get_run(self, run_id: int) -> Optional[RunInfo]:
        """Get a specific run by id"""
        return self.db_manager.get_run(run_id)

    def delete_run(self, run_id: int):
        """Delete a run and all its data"""
        self.db_manager.delete_run(run_id)

    def get_metrics(self, run_id: int) -> list[str]:
        """Get all metric names for a given run"""
        return self.db_manager.get_metrics(run_id)

    def get_records(self, run_id: int, metric_name: str) -> list[RecordInfo]:
        """Get metric records for a given run and optional metric name"""
        return self.db_manager.get_records(run_id, metric_name)

    def get_results(self, run_id: int) -> dict[str, list[RecordInfo]]:
        """Get all metric records for a given run"""
        metrics = self.get_metrics(run_id)
        results = {metric: self.get_records(run_id, metric) for metric in metrics}
        return results

    def save_comp_graph(self, tensor: Tensor, run_id: Optional[int] = None):
        """Save computation graph for the current run"""
        id = run_id
        if id is None:
            id = self.current_run_id
        if id is None:
            raise RuntimeError("No active run. Call start_run() first.")
        print(f"Saving computation graph for run {id}...")
        graph_data = _build_graph_data(tensor)
        self.db_manager.save_comp_graph(run_id=id, graph_data=graph_data)

    def get_comp_graph(self, graph_id: int) -> Optional[dict]:
        """Get computation graph for a given run"""
        return self.db_manager.get_comp_graph(graph_id)

    def get_comp_graphs(self, run_id: int) -> list[dict]:
        """Get all computation graphs for a given run"""
        return self.db_manager.get_comp_graphs(run_id)

    # def launch_simpleboard(self, port: int = 8000, host: str = "127.0.0.1", no_browser: bool = False):
    #     """Launch the SimpleBoard visualization dashboard.

    #     Args:
    #         port (int): Port to run the server on. Defaults to 8000.
    #         host (str): Host to bind to. Defaults to "127.0.0.1".
    #         no_browser (bool): Don't automatically open browser. Defaults to False.
    #     """
    #     import os

    #     # Set environment variables for the server
    #     os.environ["SG_EXPERIMENTS_DIR"] = str(self.all_exp_dir)

    #     # Open browser after a short delay if not disabled
    #     if not no_browser:

    #         def open_browser():
    #             time.sleep(1.5)
    #             webbrowser.open(f"http://{host}:{port}")

    #         import threading

    #         threading.Thread(target=open_browser, daemon=True).start()

    #     print(f"Starting SimpleBoard...")
    #     print(f"Experiments directory: {self.all_exp_dir}")
    #     print(f"Server URL: http://{host}:{port}")

    #     # Run the server using uvicorn
    #     try:
    #         import uvicorn
    #     except ImportError:
    #         raise ImportError("uvicorn is not installed. Install it with: pip install uvicorn")

    #     uvicorn.run("simplegrad.simpleboard.server:app", host=host, port=port, log_level="info")
