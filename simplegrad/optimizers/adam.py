"""Adam optimizer."""

import numpy as np
from ..core import Optimizer, Module, Tensor


class Adam(Optimizer):
    """Adam optimizer with bias-corrected moment estimates.

    Update rule::

        m_t = beta_1 * m_{t-1} + (1 - beta_1) * grad
        v_t = beta_2 * v_{t-1} + (1 - beta_2) * grad^2
        m_hat = m_t / (1 - beta_1^t)
        v_hat = v_t / (1 - beta_2^t)
        param -= lr * m_hat / (sqrt(v_hat) + eps)

    Args:
        model: The model whose parameters to optimize.
        lr: Learning rate.
        beta_1: Exponential decay for the first moment. Defaults to 0.9.
        beta_2: Exponential decay for the second moment. Defaults to 0.999.
        eps: Numerical stability constant. Defaults to 1e-8.
    """

    def __init__(
        self,
        model: Module,
        lr: float,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(lr, model)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.moments1 = {
            name: np.zeros_like(param.values) for name, param in self.model.parameters().items()
        }
        self.moments2 = {
            name: np.zeros_like(param.values) for name, param in self.model.parameters().items()
        }
        self.step_count = 0

    def step(self):
        """Apply one Adam update step to all model parameters.

        Raises:
            ValueError: If any parameter gradient is None (forgot to call backward).
        """
        self.step_count += 1
        for name, param in self.model.parameters().items():
            if param.grad is None:
                raise ValueError(f"Gradient for {name} is None. Did you forget to call backward()?")

            # Update biased first moment estimate
            self.moments1[name] = self.beta_1 * self.moments1[name] + (1 - self.beta_1) * param.grad

            # Update biased second raw moment estimate
            self.moments2[name] = self.beta_2 * self.moments2[name] + (1 - self.beta_2) * (
                param.grad**2
            )

            # Compute bias-corrected first moment estimate
            m_hat = self.moments1[name] / (1 - self.beta_1**self.step_count)

            # Compute bias-corrected second raw moment estimate
            v_hat = self.moments2[name] / (1 - self.beta_2**self.step_count)

            # Update parameters
            param.values -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
