"""High-level experiment tracking API built on top of ExperimentDBManager."""

from pathlib import Path
from .exp_db_manager import ExperimentDBManager, RunInfo, RecordInfo
from .comp_graph import _build_graph_data
from simplegrad.core import Tensor
import numpy as np


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
        self.all_exp_dir.mkdir(
            parents=True, exist_ok=True
        )  # Create the experiments directory if it doesn't exist

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

    def start_run(self, name: str | None = None, config: dict | None = None) -> int:
        """Start a new run and return the run_id"""
        self.current_run_id = self.db_manager.create_run(name=name, config=config)
        self.current_run_name = name or f"run_{self.current_run_id}"
        return self.current_run_id

    def record(self, metric_name: str, value: float, step: int):
        """Log a metric value at a given step"""
        if self.current_run_id is None:
            raise RuntimeError("No active run. Call start_run() first.")
        self.db_manager.record(self.current_run_id, metric_name, step, value)

    def histogram(self, name: str, tensor: Tensor | np.ndarray, step: int, bins: int = 30):
        """Log a histogram of values at a given step"""
        if self.current_run_id is None:
            raise RuntimeError("No active run. Call start_run() first.")
        
        if isinstance(tensor, Tensor):
            data = tensor.realize().numpy()
        else:
            data = np.asarray(tensor)
            
        counts, edges = np.histogram(data, bins=bins)
        self.db_manager.save_histogram(
            self.current_run_id, 
            name, 
            step, 
            edges.tolist(), 
            counts.tolist()
        )

    def image(self, name: str, image_data: np.ndarray, step: int):
        """Log an image at a given step. image_data should be a numpy array of shape (H, W, C) or (H, W)"""
        if self.current_run_id is None:
            raise RuntimeError("No active run. Call start_run() first.")
        
        arr = np.asarray(image_data)
        if arr.ndim == 2:
            h, w = arr.shape
            c = 1
        elif arr.ndim == 3:
            h, w, c = arr.shape
        else:
            raise ValueError(f"Image must be 2D or 3D array, got shape {arr.shape}")
            
        # Ensure it's uint8
        if arr.dtype != np.uint8:
            if arr.dtype in [np.float32, np.float64]:
                arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
                
        self.db_manager.save_image(
            self.current_run_id,
            name,
            step,
            int(w),
            int(h),
            int(c),
            arr.tobytes()
        )

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

    def get_run(self, run_id: int) -> RunInfo | None:
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

    def save_comp_graph(self, tensor: Tensor, run_id: int | None = None):
        """Save computation graph for the current run"""
        id = run_id
        if id is None:
            id = self.current_run_id
        if id is None:
            raise RuntimeError("No active run. Call start_run() first.")
        print(f"Saving computation graph for run {id}...")
        graph_data = _build_graph_data(tensor)
        self.db_manager.save_comp_graph(run_id=id, graph_data=graph_data)

    def get_comp_graph(self, graph_id: int) -> dict | None:
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
