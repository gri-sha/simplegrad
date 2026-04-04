"""Differentiable math functions (log, exp, trig)."""

import numpy as np
from ..core import Tensor, Function, Context, is_lazy


class _Log(Function):
    oper = "log"

    @staticmethod
    def forward(ctx: Context, x: Tensor) -> np.ndarray:
        ctx.x_values = x.values
        return np.log(x.values)

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> np.ndarray:
        return grad_output / ctx.x_values


class _Exp(Function):
    oper = "exp"

    @staticmethod
    def forward(ctx: Context, x: Tensor) -> np.ndarray:
        ctx.out = np.exp(x.values)
        return ctx.out

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * ctx.out


class _Sin(Function):
    oper = "sin"

    @staticmethod
    def forward(ctx: Context, x: Tensor) -> np.ndarray:
        ctx.x_values = x.values
        return np.sin(x.values)

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * np.cos(ctx.x_values)


class _Cos(Function):
    oper = "cos"

    @staticmethod
    def forward(ctx: Context, x: Tensor) -> np.ndarray:
        ctx.x_values = x.values
        return np.cos(x.values)

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> np.ndarray:
        return -grad_output * np.sin(ctx.x_values)


class _Tan(Function):
    oper = "tan"

    @staticmethod
    def forward(ctx: Context, x: Tensor) -> np.ndarray:
        ctx.x_values = x.values
        return np.tan(x.values)

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> np.ndarray:
        return grad_output / (np.cos(ctx.x_values) ** 2)


def log(x: Tensor) -> Tensor:
    """Compute element-wise natural logarithm.

    Args:
        x: Input tensor. All values must be positive.

    Returns:
        Tensor of ln(x).

    Raises:
        ValueError: If any value in x is <= 0 (checked in eager mode only).
    """
    if not is_lazy() and np.any(x.values <= 0):
        raise ValueError("Log of negative value is undefined")
    return _Log.apply(x)


def exp(x: Tensor) -> Tensor:
    """Compute element-wise exponential.

    Args:
        x: Input tensor.

    Returns:
        Tensor of e^x.
    """
    return _Exp.apply(x)


def sin(x: Tensor) -> Tensor:
    """Compute element-wise sine.

    Args:
        x: Input tensor (radians).

    Returns:
        Tensor of sin(x).
    """
    return _Sin.apply(x)


def cos(x: Tensor) -> Tensor:
    """Compute element-wise cosine.

    Args:
        x: Input tensor (radians).

    Returns:
        Tensor of cos(x).
    """
    return _Cos.apply(x)


def tan(x: Tensor) -> Tensor:
    """Compute element-wise tangent.

    Args:
        x: Input tensor (radians).

    Returns:
        Tensor of tan(x).
    """
    return _Tan.apply(x)
