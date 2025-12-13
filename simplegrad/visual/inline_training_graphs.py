import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
import random
from simplegrad.track import RecordInfo


def plot(
    results: dict[str, list[RecordInfo]],
    selected: Optional[list[str]] = None,
    num_cols: int = 2,
    cell_w: int = 8,
    cell_h: int = 5,
    path: Optional[Path] = None,
    color: Optional[str] = None,
):
    """Plot training metrics inline.

    Args:
        results (dict[str, list[RecordInfo]]): A dictionary where keys are metric names and values are lists of RecordInfo.
        selected (Optional[list[str]], optional): List of metric names to plot. If None, plots all metrics. Defaults to None.
        num_cols (int, optional): Number of columns in the plot grid. Defaults to 2.
        cell_w (int, optional): Width of each subplot cell. Defaults to 8.
        cell_h (int, optional): Height of each subplot cell. Defaults to 5.
        path (Optional[Path], optional): If provided, saves the plot to this path. Defaults to None.
        marker (str, optional): Marker style for the plot lines. Defaults to "o".
        color (Optional[str], optional): Color for all plot lines. If None, colors are randomly assigned. Defaults to None.
    """
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    # Determine which metrics to plot
    metrics_to_plot = selected if selected else list(results.keys())

    num_metrics = len(metrics_to_plot)
    num_rows = (num_metrics + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(cell_w * num_cols, cell_h * num_rows))
    axes = axes.flatten() if num_metrics > 1 else [axes]

    for ax in axes[num_metrics:]:
        ax.axis("off")

    for i, metric_name in enumerate(metrics_to_plot):
        if metric_name not in results:
            continue

        records = results[metric_name]
        steps = [record.step for record in records]
        values = [record.value for record in records]
        plot_color = color if color else random.choice(colors)
        marker = "o" if len(steps) < cell_w * 8 else None

        axes[i].plot(steps, values, marker=marker, color=plot_color)
        axes[i].set_title(metric_name)
        axes[i].set_xlabel("Step")
        axes[i].set_ylabel("Value")
        axes[i].grid(True)

    plt.tight_layout()
    if path:
        plt.savefig(path)
    plt.show()


def scatter(
    results: dict[str, list[RecordInfo]],
    selected: Optional[list[str]] = None,
    num_cols: int = 2,
    cell_w: int = 8,
    cell_h: int = 5,
    path: Optional[Path] = None,
    color: Optional[str] = None,
):
    """Plot training metrics as scatter plot inline.

    Args:
        results (dict[str, list[RecordInfo]]): A dictionary where keys are metric names and values are lists of RecordInfo.
        selected (Optional[list[str]], optional): List of metric names to plot. If None, plots all metrics. Defaults to None.
        num_cols (int, optional): Number of columns in the plot grid. Defaults to 2.
        cell_w (int, optional): Width of each subplot cell. Defaults to 8.
        cell_h (int, optional): Height of each subplot cell. Defaults to 5.
        path (Optional[Path], optional): If provided, saves the plot to this path. Defaults to None.
        color (Optional[str], optional): Color for all scatter points. If None, colors are randomly assigned. Defaults to None.
    """
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    # Determine which metrics to plot
    metrics_to_plot = selected if selected else list(results.keys())

    num_metrics = len(metrics_to_plot)
    num_rows = (num_metrics + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(cell_w * num_cols, cell_h * num_rows))
    axes = axes.flatten() if num_metrics > 1 else [axes]

    for ax in axes[num_metrics:]:
        ax.axis("off")

    for i, metric_name in enumerate(metrics_to_plot):
        if metric_name not in results:
            continue

        records = results[metric_name]
        steps = [record.step for record in records]
        values = [record.value for record in records]
        plot_color = color if color else random.choice(colors)

        axes[i].scatter(steps, values, color=plot_color)
        axes[i].set_title(metric_name)
        axes[i].set_xlabel("Step")
        axes[i].set_ylabel("Value")
        axes[i].grid(True)

    plt.tight_layout()
    if path:
        plt.savefig(path)
    plt.show()
