"""Abstract base class for all optimizers."""

import numpy as np


class Optimizer:
    """Base class for all optimizers.

    Subclasses must implement `step()` to define the parameter update rule.
    """

    def __init__(self):
        self.step_count = 0

    def zero_grad(self):
        """Zero gradients for all model parameters."""
        for _, param in self.model.parameters().items():
            param.grad = np.zeros_like(param.values)

    def step(self):
        """Perform a single optimization step. Must be implemented by subclasses."""
        raise NotImplementedError("step() method is not implemented.")

    def reset_step_count(self):
        """Reset the internal step counter to zero."""
        self.step_count = 0
