from abc import ABC, abstractmethod
from .metric import MetricHistory


class Visualizer(ABC):
    @abstractmethod
    def plot(self, ax, metric: MetricHistory):
        pass


class LineVisualizer(Visualizer):
    def __init__(self, color: str = "blue", linestyle: str = "-", marker: str = "o"):
        self.color = color
        self.linestyle = linestyle
        self.marker = marker

    def plot(self, ax, metric: MetricHistory):
        ax.plot(
            metric.steps,
            metric.values,
            color=self.color,
            linestyle=self.linestyle,
            marker=self.marker,
            label=metric.name,
        )


class ScatterVisualizer(Visualizer):
    def __init__(self, color: str = "red", alpha: float = 0.6):
        self.color = color
        self.alpha = alpha

    def plot(self, ax, metric: MetricHistory):
        ax.scatter(
            metric.steps,
            metric.values,
            color=self.color,
            alpha=self.alpha,
            label=metric.name,
        )
