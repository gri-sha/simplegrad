"""Activation function layers (Module wrappers)."""

from ..core import Tensor, Module
from ..functions.activations import *


class ReLU(Module):
    """ReLU activation layer: max(0, x)."""

    def forward(self, x: Tensor) -> Tensor:
        """Apply ReLU element-wise."""
        return relu(x)

    def __str__(self):
        return "ReLU"


class Softmax(Module):
    """Softmax activation layer.

    Args:
        dim: Dimension to normalize over. Defaults to None (all elements).
    """

    def __init__(self, dim: int | None = None):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor):
        """Apply softmax along ``dim``."""
        return softmax(x, self.dim)

    def __str__(self):
        return f"Softmax(dim={self.dim})"


class Tanh(Module):
    """Tanh activation layer: tanh(x)."""

    def forward(self, x: Tensor):
        """Apply tanh element-wise."""
        return tanh(x)

    def __str__(self):
        return "Tanh"


class Sigmoid(Module):
    """Sigmoid activation layer: 1 / (1 + exp(-x))."""

    def forward(self, x: Tensor):
        """Apply sigmoid element-wise."""
        return sigmoid(x)

    def __str__(self):
        return "Sigmoid"
