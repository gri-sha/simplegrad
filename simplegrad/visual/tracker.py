import matplotlib.pyplot as plt
from typing import Optional
from .visualizer import MetricHistory, Visualizer, LineVisualizer, ScatterVisualizer
from .storage import RunStorage


class TrainingTracker:
    def __init__(
        self, title: str = "Training Progress", logdir: Optional[str] = None, run_name: Optional[str] = None, config: Optional[dict] = None
    ):
        self.title = title
        self.metrics: dict[str, MetricHistory] = {}
        self.visualizers: dict[str, Visualizer] = {}
        self.step = 0

        # Storage backend (optional)
        self._store: Optional[RunStorage] = None
        self._run_id: Optional[str] = None

        if logdir is not None:
            self._store = RunStorage(logdir=logdir)
            self._run_id = self._store.create_run(name=run_name or title, config=config or {})
            print(f"📊 Logging to: {logdir} (run_id: {self._run_id})")

    @property
    def run_id(self) -> Optional[str]:
        """Get the current run ID if logging to storage."""
        return self._run_id

    def register_metric(self, name, visualizer="line"):
        self.metrics[name] = MetricHistory(name)
        self.visualizers[name] = visualizer or LineVisualizer()

    def log(self, **kwargs):
        for name, value in kwargs.items():
            if name not in self.metrics:
                self.register_metric(name)
            self.metrics[name].add(self.step, value)

            # Also log to storage if enabled
            if self._store is not None and self._run_id is not None:
                self._store.log_scalar(self._run_id, name, self.step, float(value))

        self.step += 1

    def plot(self, figsize=None, show=True, num_cols: int = 5, cell_h=4, cell_w=6, path=None):
        n_metrics = len(self.metrics)
        if n_metrics == 0:
            raise ValueError("No metrics to plot.")

        cols = min(n_metrics, num_cols)
        rows = n_metrics // cols + 1

        # Auto-adjust figsize based on grid
        if figsize is None:
            figsize = (cell_w * cols, cell_h * rows)

        fig, axes = plt.subplots(rows, cols, figsize=figsize)

        # Handle single metric case
        if n_metrics == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        for idx, (ax, (name, metric)) in enumerate(zip(axes, self.metrics.items())):
            visualizer = self.visualizers[name]
            visualizer.plot(ax, metric)
            ax.set_xlabel("Step")
            ax.set_ylabel(name.capitalize())
            ax.set_title(name)
            ax.grid(True)
            ax.legend()

        # Hide extra subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle(self.title)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        if show:
            plt.show()

        if path:
            fig.savefig(path)

        return fig, axes

    def plot_combined(self, metrics_names, figsize=(10, 6)):
        fig, ax = plt.subplots(figsize=figsize)

        for name in metrics_names:
            if name in self.metrics:
                metric = self.metrics[name]
                ax.plot(metric.steps, metric.values, marker="o", label=name)

        ax.set_xlabel("Step")
        ax.set_ylabel("Value")
        ax.set_title(self.title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return fig, ax

    def get_metric(self, name):
        """Retrieve metric history"""
        return self.metrics.get(name)

    def summary(self):
        """Print summary of all metrics"""
        for name, metric in self.metrics.items():
            if metric.values:
                print(f"{name}: min={min(metric.values):.4f}, " f"max={max(metric.values):.4f}, " f"final={metric.values[-1]:.4f}")

    def finish(self, status: str = "completed"):
        """Mark the run as finished. Call this at the end of training."""
        if self._store is not None and self._run_id is not None:
            self._store.update_run_status(self._run_id, status)
            print(f"✅ Run {self._run_id} marked as {status}")

    def log_graph(self, tensor):
        """Log computation graph from a tensor. Requires the tensor to have been computed."""
        if self._store is None or self._run_id is None:
            return

        # Build graph data structure for D3
        from .graph import _build_graph_data

        graph_data = _build_graph_data(tensor)
        self._store.save_graph(self._run_id, graph_data)
