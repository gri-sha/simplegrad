"""Activation functions with autograd support."""

import math as _math

import numpy as np
from .math import exp
from .reduction import sum
from ..core import Tensor, Function, Context, compound_op

# vectorised erf for numpy; cupy exposes xp.erf directly as a ufunc
_erf_np = np.vectorize(_math.erf)


class _Relu(Function):
    oper = "ReLU"

    @staticmethod
    def forward(ctx: Context, x: Tensor) -> np.ndarray:
        xp = ctx.backend
        ctx.mask = x.values > 0
        return xp.maximum(0, x.values)

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * ctx.mask

def relu(x: Tensor) -> Tensor:
    """Apply ReLU activation element-wise: max(0, x).

    ReLU is the most widely used activation function in deep learning due to
    its simplicity and effectiveness at avoiding the vanishing gradient problem
    for positive inputs. The gradient is 1 for positive inputs and 0 otherwise,
    meaning that neurons with negative pre-activations receive no gradient
    signal (the "dying ReLU" problem).

    Args:
        x: Input tensor of any shape.

    Returns:
        Tensor of the same shape with all negative values replaced by zero.

    Example:
        >>> x = Tensor([-1.0, 0.0, 2.0])
        >>> relu(x)
        Tensor([0.0, 0.0, 2.0])
    """
    return _Relu.apply(x)


def _erf(xp, arr):
    """Compute erf element-wise, supporting numpy and cupy backends."""
    if xp is np:
        return _erf_np(arr)
    return xp.erf(arr)

class _GELUErf(Function):
    oper = "GELUErf"

    @staticmethod
    def forward(ctx: Context, x: Tensor) -> np.ndarray:
        xp = ctx.backend
        ctx.x = x.values
        ctx.erf_val = _erf(xp, x.values / np.sqrt(2))
        return 0.5 * x.values * (1 + ctx.erf_val)

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> np.ndarray:
        xp = ctx.backend
        # d/dx [0.5 * x * (1 + erf(x/sqrt(2)))]
        # = 0.5*(1 + erf(x/sqrt(2))) + x*exp(-x^2/2)/sqrt(2*pi)
        deriv = 0.5 * (1 + ctx.erf_val) + ctx.x * xp.exp(-ctx.x**2 / 2) / np.sqrt(2 * np.pi)
        return grad_output * deriv

class _Tanh(Function):
    oper = "Tanh"

    @staticmethod
    def forward(ctx: Context, x: Tensor) -> np.ndarray:
        xp = ctx.backend
        ctx.out = xp.tanh(x.values)
        return ctx.out

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * (1 - ctx.out**2)


def tanh(x: Tensor) -> Tensor:
    """Apply hyperbolic tangent element-wise.

    Maps real values to the range (-1, 1). Compared to sigmoid, tanh is
    zero-centered, which often leads to faster convergence in practice.
    Like sigmoid, it saturates for large |x|, causing vanishing gradients
    in deep networks.

    Args:
        x: Input tensor of any shape.

    Returns:
        Tensor of the same shape with values in the open interval (-1, 1).

    Example:
        >>> x = Tensor([0.0, 1.0, -1.0])
        >>> tanh(x)
        Tensor([0.0, 0.762, -0.762])
    """
    return _Tanh.apply(x)

class _Sigmoid(Function):
    oper = "Sigmoid"

    @staticmethod
    def forward(ctx: Context, x: Tensor) -> np.ndarray:
        xp = ctx.backend
        ctx.out = 1 / (1 + xp.exp(-x.values))
        return ctx.out

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * ctx.out * (1 - ctx.out)


def sigmoid(x: Tensor) -> Tensor:
    """Apply sigmoid activation element-wise: 1 / (1 + exp(-x)).

    Maps real values to the range (0, 1), making it useful for binary
    classification outputs. The gradient is sigma(x) * (1 - sigma(x)),
    which approaches zero for large |x| (vanishing gradient problem).

    Args:
        x: Input tensor of any shape.

    Returns:
        Tensor of the same shape with values in the open interval (0, 1).

    Example:
        >>> x = Tensor([0.0, 1.0, -1.0])
        >>> sigmoid(x)
        Tensor([0.5, 0.731, 0.269])
    """
    return _Sigmoid.apply(x)

class _ELU(Function):
    @staticmethod
    def forward(ctx: Context, x: Tensor, alpha: float) -> np.ndarray:
        xp = ctx.backend
        ctx.mask = x.values > 0
        # clamp to 0 before exp to avoid overflow on large positive values
        ctx.exp_x = xp.exp(xp.minimum(x.values, 0))
        ctx.alpha = alpha
        return xp.where(ctx.mask, x.values, alpha * (ctx.exp_x - 1))

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> np.ndarray:
        # derivative: 1 where x>0, alpha*exp(x) where x<=0
        return grad_output * (ctx.mask + ~ctx.mask * ctx.alpha * ctx.exp_x)


def elu(x: Tensor, alpha: float = 1.0) -> Tensor:
    """Apply Exponential Linear Unit activation element-wise.

    ELU is defined as::

        elu(x) = x                    if x > 0
                 alpha * (exp(x) - 1) if x <= 0

    Unlike ReLU, ELU has a smooth, non-zero output for negative inputs,
    which allows the mean activation to be closer to zero and reduces the
    "dying neuron" problem. The ``alpha`` parameter controls the saturation
    value for strongly negative inputs (the limit as x → -∞ is -alpha).

    Args:
        x: Input tensor of any shape.
        alpha: Slope scale for the negative region. Must be > 0. Defaults to 1.0.

    Returns:
        Tensor of the same shape as x.

    Example:
        >>> x = Tensor([-1.0, 0.0, 1.0])
        >>> elu(x)
        Tensor([-0.632, 0.0, 1.0])
    """
    return _ELU.apply(x, alpha, oper=f"ELU(a={alpha})")


@compound_op
def _gelu_tanh(x: Tensor) -> Tensor:
    """GELU tanh approximation — called internally by gelu(mode='tanh')."""
    c = float(np.sqrt(2.0 / np.pi))
    return 0.5 * x * (1 + tanh(c * (x + 0.044715 * x**3)))


def gelu(x: Tensor, mode: str = "erf") -> Tensor:
    """Apply Gaussian Error Linear Unit activation element-wise.

    GELU is defined as x * Φ(x), where Φ is the standard Gaussian CDF.
    It has become the standard activation in transformer architectures
    (BERT, GPT) because it combines the properties of dropout, zoneout,
    and ReLU. Unlike ReLU it is smooth and probabilistically gates inputs
    based on their magnitude.

    Two computation modes are provided:

    **"erf"** (default, exact)::

        gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))

    **"tanh"** (fast approximation, original paper formula)::

        gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))

    The tanh approximation was proposed by Hendrycks & Gimpel (2016) and
    is accurate to within 0.02% for all x. It is the mode used by most
    production implementations when a fast path is desired.

    Args:
        x: Input tensor of any shape.
        mode: Computation mode — ``"erf"`` for the exact formula or
            ``"tanh"`` for the fast approximation. Defaults to ``"erf"``.

    Returns:
        Tensor of the same shape as x.

    Raises:
        ValueError: If ``mode`` is not ``"erf"`` or ``"tanh"``.

    Example:
        >>> x = Tensor([-1.0, 0.0, 1.0])
        >>> gelu(x)
        Tensor([-0.159, 0.0, 0.841])
    """
    if mode == "tanh":
        return _gelu_tanh(x)
    if mode == "erf":
        return _GELUErf.apply(x)
    raise ValueError(f"Unknown GELU mode '{mode}'. Expected 'erf' or 'tanh'.")


@compound_op
def softmax(x: Tensor, dim: int | None = None) -> Tensor:
    """Apply softmax along the given dimension.

    Softmax converts a vector of real values into a probability distribution:
    each output is in (0, 1) and the values along ``dim`` sum to 1. It is
    defined as exp(x_i) / sum_j(exp(x_j)).

    For numerical stability, implementations typically subtract max(x) before
    exponentiation; this implementation relies on the underlying exp/sum ops
    and their handling of large values.

    Args:
        x: Input tensor of any shape.
        dim: Dimension to normalize over. If None, normalizes over all elements.

    Returns:
        Tensor of the same shape as x. Values along ``dim`` sum to 1.

    Example:
        >>> x = Tensor([[1.0, 2.0, 3.0]])
        >>> softmax(x, dim=1)
        Tensor([[0.090, 0.245, 0.665]])
    """
    exps = exp(x)
    return exps / sum(exps, dim)
