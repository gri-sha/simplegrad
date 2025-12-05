from dataclasses import dataclass, field

@dataclass
class MetricHistory:
    name: str
    values: list[float] = field(default_factory=list)
    steps: list[int] = field(default_factory=list)

    def add(self, step: int, value: float):
        self.steps.append(step)
        self.values.append(value)