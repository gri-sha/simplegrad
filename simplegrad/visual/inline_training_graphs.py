"""Matplotlib-based training metric plots for inline (notebook) use."""

import matplotlib.pyplot as plt
from pathlib import Path
import random
from ..track import RecordInfo


def plot(
    results: dict[str, list[RecordInfo]],
    selected: list[str] | None = None,
    num_cols: int = 2,
    cell_w: int = 8,
    cell_h: int = 5,
    path: Path | None = None,
    color: str | None = None,
):
    """Plot training metrics as line charts.

    Args:
        results: Mapping of metric name to list of ``RecordInfo`` data points.
        selected: Subset of metric names to plot. Plots all if None.
        num_cols: Number of subplot columns. Defaults to 2.
        cell_w: Width of each subplot cell in inches. Defaults to 8.
        cell_h: Height of each subplot cell in inches. Defaults to 5.
        path: If provided, save the figure to this path.
        color: Fixed color for all lines. Random if None.
    """
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

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
    selected: list[str] | None = None,
    num_cols: int = 2,
    cell_w: int = 8,
    cell_h: int = 5,
    path: Path | None = None,
    color: str | None = None,
):
    """Plot training metrics as scatter charts.

    Args:
        results: Mapping of metric name to list of ``RecordInfo`` data points.
        selected: Subset of metric names to plot. Plots all if None.
        num_cols: Number of subplot columns. Defaults to 2.
        cell_w: Width of each subplot cell in inches. Defaults to 8.
        cell_h: Height of each subplot cell in inches. Defaults to 5.
        path: If provided, save the figure to this path.
        color: Fixed color for all points. Random if None.
    """
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

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
