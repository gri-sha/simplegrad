"""Shape transformation layers."""

from ..core import Tensor
from .module import Module
from ..functions import flatten


class Flatten(Module):
    """Flatten a range of tensor dimensions into a single dimension.

    Args:
        start_dim: First dimension to flatten (inclusive). Defaults to 1
            (preserves the batch dimension).
        end_dim: Last dimension to flatten (inclusive). Defaults to -1
            (the last dimension).
    """

    def __init__(self, start_dim=1, end_dim=-1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x):
        """Flatten the input tensor.

        Args:
            x: Input tensor.

        Returns:
            Tensor with dimensions ``[start_dim, end_dim]`` merged into one.
        """
        return flatten(x, self.start_dim, self.end_dim)

    def __str__(self):
        return f"Flatten(start_dim={self.start_dim}, end_dim={self.end_dim})"
