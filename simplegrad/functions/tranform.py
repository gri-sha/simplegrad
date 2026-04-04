"""Tensor shape transformation functions: flatten and reshape."""

import numpy as np
from ..core import Tensor, Function, Context

class _Flatten(Function):
    oper = "Flatten"

    @staticmethod
    def output_shape(x: Tensor, start_dim: int = 0, end_dim: int = -1) -> tuple:
        ndim = len(x.shape)
        start = start_dim % ndim
        end = end_dim % ndim
        flat_size = 1
        for i in range(start, end + 1):
            flat_size *= x.shape[i]
        return x.shape[:start] + (flat_size,) + x.shape[end + 1 :]

    @staticmethod
    def forward(ctx: Context, x: Tensor, start_dim: int = 0, end_dim: int = -1) -> np.ndarray:
        ctx.x_shape = x.shape
        ndim = len(x.shape)
        start = start_dim % ndim
        end = end_dim % ndim
        flat_size = 1
        for i in range(start, end + 1):
            flat_size *= x.shape[i]
        out_shape = x.shape[:start] + (flat_size,) + x.shape[end + 1 :]
        return x.values.reshape(out_shape)

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> np.ndarray:
        return grad_output.reshape(ctx.x_shape)


class _Reshape(Function):
    @staticmethod
    def output_shape(x: Tensor, new_shape: tuple) -> tuple:
        return new_shape

    @staticmethod
    def forward(ctx: Context, x: Tensor, new_shape: tuple) -> np.ndarray:
        ctx.x_shape = x.shape
        return x.values.reshape(new_shape)

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> np.ndarray:
        return grad_output.reshape(ctx.x_shape)


def flatten(x: Tensor, start_dim: int = 0, end_dim: int = -1) -> Tensor:
    """Flatten a range of dimensions into a single dimension.

    Args:
        x: Input tensor.
        start_dim: First dimension to flatten (inclusive). Supports negative indexing.
        end_dim: Last dimension to flatten (inclusive). Supports negative indexing.

    Returns:
        Tensor with dimensions ``[start_dim, end_dim]`` merged into one.
    """
    return _Flatten.apply(x, start_dim, end_dim)


def reshape(x: Tensor, new_shape: tuple[int, ...]) -> Tensor:
    """Reshape a tensor to a new shape.

    Args:
        x: Input tensor.
        new_shape: Target shape. Total number of elements must match.

    Returns:
        Tensor with values laid out in ``new_shape``.
    """
    return _Reshape.apply(x, new_shape, oper=f"reshape({new_shape})")
