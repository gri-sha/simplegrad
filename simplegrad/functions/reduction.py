"""Reduction operations: sum, mean, trace, argmax, argmin."""

import numpy as np
from ..core import Tensor, Function, Context, get_dtype_class


class _Sum(Function):
    @staticmethod
    def output_shape(x: Tensor, dim: int | None = None) -> tuple:
        if dim is None:
            return (1,) * len(x.shape)
        return tuple(1 if i == dim % len(x.shape) else s for i, s in enumerate(x.shape))

    @staticmethod
    def forward(ctx: Context, x: Tensor, dim: int | None = None) -> np.ndarray:
        xp = ctx.backend
        ctx.x_shape = x.shape
        return xp.sum(x.values, axis=dim, keepdims=True)

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> np.ndarray:
        xp = ctx.backend
        return xp.ones(ctx.x_shape) * grad_output


class _Trace(Function):
    oper = "trace"

    @staticmethod
    def output_shape(x: Tensor) -> tuple:
        return (1, 1)

    @staticmethod
    def forward(ctx: Context, x: Tensor) -> np.ndarray:
        xp = ctx.backend
        ctx.x_shape = x.shape
        return xp.array([[xp.trace(x.values)]])

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> np.ndarray:
        xp = ctx.backend
        grad = xp.zeros(ctx.x_shape)
        n = min(ctx.x_shape)
        idxs = xp.arange(n)
        grad[idxs, idxs] = grad_output.flatten()[:n]
        return grad


class _Argmax(Function):
    oper = "argmax"
    differentiable = False

    @staticmethod
    def output_shape(x: Tensor, dim: int | None = None, dtype: str = "int32") -> tuple:
        if dim is None:
            return (1,)
        return tuple(1 if i == dim % len(x.shape) else s for i, s in enumerate(x.shape))

    @staticmethod
    def forward(
        ctx: Context, x: Tensor, dim: int | None = None, dtype: str = "int32"
    ) -> np.ndarray:
        xp = ctx.backend
        return xp.argmax(x.values, axis=dim)

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> np.ndarray:
        raise RuntimeError("argmax is not differentiable and does not support backpropagation")


class _Argmin(Function):
    oper = "argmin"
    differentiable = False

    @staticmethod
    def output_shape(x: Tensor, dim: int | None = None, dtype: str = "int32") -> tuple:
        if dim is None:
            return (1,)
        return tuple(1 if i == dim % len(x.shape) else s for i, s in enumerate(x.shape))

    @staticmethod
    def forward(
        ctx: Context, x: Tensor, dim: int | None = None, dtype: str = "int32"
    ) -> np.ndarray:
        xp = ctx.backend
        return xp.argmin(x.values, axis=dim)

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> np.ndarray:
        raise RuntimeError("argmin is not differentiable and does not support backpropagation")


def sum(x: Tensor, dim: int | None = None) -> Tensor:
    """Sum tensor elements along a dimension.

    Args:
        x: Input tensor.
        dim: Dimension to reduce. If None, sums all elements.

    Returns:
        Reduced tensor (keepdims=True).
    """
    return _Sum.apply(x, dim, oper=f"sum(d={dim})")


def trace(x: Tensor) -> Tensor:
    """Compute the trace (sum of diagonal elements) of a square matrix.

    Args:
        x: 2D square tensor.

    Returns:
        Scalar tensor containing the trace.

    Raises:
        ValueError: If x is not a 2D square tensor (checked in eager mode).
    """
    if x.values is not None and (x.values.ndim != 2 or x.values.shape[0] != x.values.shape[1]):
        raise ValueError("Trace is only defined for square matrices")
    return _Trace.apply(x)


def mean(x: Tensor, dim: int | None = None) -> Tensor:
    """Compute the mean of tensor elements along a dimension.

    Args:
        x: Input tensor.
        dim: Dimension to reduce. If None, averages all elements.

    Returns:
        Reduced tensor.
    """
    if dim is None:
        n = 1
        for s in x.shape:
            n *= s
        return sum(x) / n
    return sum(x, dim=dim) / x.shape[dim]


def argmax(x: Tensor, dim: int | None = None, dtype: str = "int32") -> Tensor:
    """Return indices of maximum values along a dimension.

    Not differentiable — ``comp_grad`` is always False on the output.

    Args:
        x: Input tensor.
        dim: Dimension to reduce. If None, returns the flat index.
        dtype: Integer dtype for the output indices.

    Returns:
        Integer tensor of argmax indices.
    """
    out = _Argmax.apply(x, dim, dtype, oper=f"argmax(d={dim})")
    out.dtype = get_dtype_class(dtype)
    return out


def argmin(x: Tensor, dim: int | None = None, dtype: str = "int32") -> Tensor:
    """Return indices of minimum values along a dimension.

    Not differentiable — ``comp_grad`` is always False on the output.

    Args:
        x: Input tensor.
        dim: Dimension to reduce. If None, returns the flat index.
        dtype: Integer dtype for the output indices.

    Returns:
        Integer tensor of argmin indices.
    """
    out = _Argmin.apply(x, dim, dtype, oper=f"argmin(d={dim})")
    out.dtype = get_dtype_class(dtype)
    return out
