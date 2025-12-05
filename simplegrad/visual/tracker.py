import matplotlib.pyplot as plt
from .visualizer import MetricHistory, Visualizer, LineVisualizer, ScatterVisualizer


class TrainingTracker:
    def __init__(self, title: str = "Training Progress"):
        self.title = title
        self.metrics: dict[str, MetricHistory] = {}
        self.visualizers: dict[str, Visualizer] = {}
        self.step = 0

    def register_metric(self, name, visualizer="line"):
        self.metrics[name] = MetricHistory(name)
        self.visualizers[name] = visualizer or LineVisualizer()

    def log(self, **kwargs):
        for name, value in kwargs.items():
            if name not in self.metrics:
                self.register_metric(name)
            self.metrics[name].add(self.step, value)
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
