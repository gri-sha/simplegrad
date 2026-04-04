"""Activation functions with autograd support."""

import numpy as np
from .math import exp
from .reduction import sum
from ..core import Tensor, Function, Context, compound_op


class _Relu(Function):
    oper = "ReLU"

    @staticmethod
    def forward(ctx: Context, x: Tensor) -> np.ndarray:
        ctx.mask = x.values > 0
        return np.maximum(0, x.values)

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * ctx.mask


class _Tanh(Function):
    oper = "Tanh"

    @staticmethod
    def forward(ctx: Context, x: Tensor) -> np.ndarray:
        ctx.out = np.tanh(x.values)
        return ctx.out

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * (1 - ctx.out**2)


class _Sigmoid(Function):
    oper = "Sigmoid"

    @staticmethod
    def forward(ctx: Context, x: Tensor) -> np.ndarray:
        ctx.out = 1 / (1 + np.exp(-x.values))
        return ctx.out

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * ctx.out * (1 - ctx.out)


def relu(x: Tensor) -> Tensor:
    """Apply ReLU activation element-wise: max(0, x).

    Args:
        x: Input tensor.

    Returns:
        Tensor with negative values replaced by zero.
    """
    return _Relu.apply(x)


def tanh(x: Tensor) -> Tensor:
    """Apply hyperbolic tangent element-wise.

    Args:
        x: Input tensor.

    Returns:
        Tensor with values in (-1, 1).
    """
    return _Tanh.apply(x)


def sigmoid(x: Tensor) -> Tensor:
    """Apply sigmoid activation element-wise: 1 / (1 + exp(-x)).

    Args:
        x: Input tensor.

    Returns:
        Tensor with values in (0, 1).
    """
    return _Sigmoid.apply(x)


@compound_op
def softmax(x: Tensor, dim: int | None = None) -> Tensor:
    """Apply softmax along the given dimension.

    Args:
        x: Input tensor.
        dim: Dimension to normalize over. If None, normalizes over all elements.

    Returns:
        Tensor where values along ``dim`` sum to 1.
    """
    exps = exp(x)
    return exps / sum(exps, dim)
