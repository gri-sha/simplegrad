"""Activation function layers (Module wrappers)."""

from ..core import Tensor, Module
from ..functions.activations import relu, tanh, sigmoid, elu, gelu, softmax


class ReLU(Module):
    """ReLU activation layer: max(0, x)."""

    def forward(self, x: Tensor) -> Tensor:
        """Apply ReLU element-wise."""
        return relu(x)

    def __str__(self):
        return "ReLU"


class ELU(Module):
    """ELU (Exponential Linear Unit) activation layer.

    Applies ``elu(x, alpha)`` element-wise. See :func:`~simplegrad.functions.activations.elu`
    for the full definition.

    Args:
        alpha: Saturation slope for the negative region. Defaults to 1.0.
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        """Apply ELU element-wise."""
        return elu(x, alpha=self.alpha)

    def __str__(self):
        return f"ELU(alpha={self.alpha})"


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


class GELU(Module):
    """GELU (Gaussian Error Linear Unit) activation layer.

    Applies ``gelu(x, mode)`` element-wise. See :func:`~simplegrad.functions.activations.gelu`
    for the full definition and the difference between modes.

    Args:
        mode: Computation mode — ``"erf"`` (exact, default) or ``"tanh"`` (approximation).
    """

    def __init__(self, mode: str = "erf"):
        super().__init__()
        self.mode = mode

    def forward(self, x: Tensor) -> Tensor:
        """Apply GELU element-wise."""
        return gelu(x, mode=self.mode)

    def __str__(self):
        return f"GELU(mode='{self.mode}')"
