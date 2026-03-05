"""Activation functions with autograd support."""

import numpy as np
from simplegrad.core.tensor import Tensor, _should_compute_grad
from .math import exp, log
from .reduction import sum


def relu(x: Tensor) -> Tensor:
    """Apply ReLU activation element-wise: max(0, x).

    Args:
        x: Input tensor.

    Returns:
        Tensor with negative values replaced by zero.
    """
    out = Tensor(np.maximum(0, x.values))
    out.prev = {x}
    out.oper = "ReLU"
    out.comp_grad = _should_compute_grad(x)
    out.is_leaf = False

    if out.comp_grad:
        out.backward_step = lambda: _relu_backward(x, out)
    return out


def _relu_backward(x: Tensor, out: Tensor) -> None:
    """Backward for ReLU: gradient is 1 where x > 0, else 0."""
    if x.comp_grad:
        x._init_grad_if_needed()
        x.grad = out.grad * np.where(x.values > 0, 1.0, 0.0)


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


def tanh(x: Tensor) -> Tensor:
    """Apply hyperbolic tangent element-wise.

    Args:
        x: Input tensor.

    Returns:
        Tensor with values in (-1, 1).
    """
    out = Tensor(np.tanh(x.values))
    out.prev = {x}
    out.oper = "Tanh"
    out.comp_grad = _should_compute_grad(x)
    out.is_leaf = False

    if out.comp_grad:
        out.backward_step = lambda: _tanh_backward(x, out)
    return out


def _tanh_backward(x: Tensor, out: Tensor) -> None:
    """Backward for tanh: d/dx = (1 - tanh(x)^2) * out.grad."""
    if x.comp_grad:
        x._init_grad_if_needed()
        x.grad += out.grad * (1 - np.tanh(x.values) ** 2)


def sigmoid(x: Tensor) -> Tensor:
    """Apply sigmoid activation element-wise: 1 / (1 + exp(-x)).

    Args:
        x: Input tensor.

    Returns:
        Tensor with values in (0, 1).
    """
    out = Tensor(1 / (1 + np.exp(-x.values)))
    out.prev = {x}
    out.oper = "Sigmoid"
    out.comp_grad = _should_compute_grad(x)
    out.is_leaf = False

    if out.comp_grad:
        out.backward_step = lambda: _sigmoid_backward(x, out)
    return out


def _sigmoid_backward(x: Tensor, out: Tensor) -> None:
    """Backward for sigmoid: d/dx = sigmoid(x) * (1 - sigmoid(x)) * out.grad."""
    if x.comp_grad:
        x._init_grad_if_needed()
        sigmoid_x = 1 / (1 + np.exp(-x.values))
        x.grad += out.grad * sigmoid_x * (1 - sigmoid_x)
