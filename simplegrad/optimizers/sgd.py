"""Stochastic Gradient Descent optimizer with optional momentum."""

from ..core import Optimizer, Module, Tensor
import numpy as np


class SGD(Optimizer):
    """Stochastic gradient descent with optional momentum.

    Update rule (with momentum)::

        v_t = momentum * v_{t-1} - lr * (1 - dampening) * grad
        param += v_t

    Args:
        model: The model whose parameters to optimize.
        lr: Learning rate. Defaults to 0.01.
        momentum: Momentum factor. 0 disables momentum. Defaults to 0.
        dampening: Dampening applied to the gradient. Defaults to 0.

    Raises:
        TypeError: If ``model`` is not a Module.
    """

    def __init__(
        self, model: Module, lr: float = 0.01, momentum: float = 0, dampening: float = 0
    ) -> None:
        if not isinstance(model, Module):
            raise TypeError(f"model must be a Module")

        super().__init__(lr, model)
        self.momentum = momentum
        self.dampening = dampening
        self.velocities = {
            name: np.zeros_like(param.values) for name, param in model.parameters().items()
        }
        self.step_count = 0

    def step(self):
        """Apply one SGD update step to all model parameters.

        Raises:
            ValueError: If any parameter gradient is None (forgot to call backward).
        """
        self.step_count += 1
        for name, param in self.model.parameters().items():
            if param.grad is None:
                raise ValueError(f"Gradient for {name} is None. Did you forget to call backward()?")

            self.velocities[name] = (
                self.momentum * self.velocities[name] - self.lr * (1 - self.dampening) * param.grad
            )
            param.values += self.velocities[name]
