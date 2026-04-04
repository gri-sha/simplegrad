"""Abstract base class for learning rate schedulers."""


class Scheduler:
    """Base class for all learning rate schedulers.

    Subclasses must implement `step()` to define the schedule update rule.
    """

    def __init__(self, optimizer):
        if not optimizer:
            raise ValueError("Optimizer must be provided.")
        self.optimizer = optimizer
        self.steps = 0

    def step(self, *args, **kwargs):
        """Advance the scheduler by one step. Must be implemented by subclasses."""
        raise NotImplementedError
