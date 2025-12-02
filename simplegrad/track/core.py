from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class MetricHistory:
    name: str
    values: list[float] = field(default_factory=list)
    steps: list[int] = field(default_factory=list)

    def add(self, step: int, value: float):
        self.steps.append(step)
        self.values.append(value)


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
