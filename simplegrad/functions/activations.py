"""Activation functions with autograd support."""

import math as _math

import numpy as np
from ..core import Tensor, Function, Context

# vectorised erf for numpy; cupy exposes xp.erf directly as a ufunc
_erf_np = np.vectorize(_math.erf)

from ..core.devices import _CUPY, cp


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
    """Apply ReLU activation element-wise: max(0, x)."""
    return _Relu.apply(x)


def _erf(xp, arr):
    """Compute erf element-wise, supporting numpy and cupy backends."""
    if xp is np:
        return _erf_np(arr)
    import cupyx.scipy.special

    return cupyx.scipy.special.erf(arr)


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


if _CUPY:
    # fused forward: 0.5 * x * (1 + erf(x / sqrt(2)))
    _gelu_erf_fwd = cp.ElementwiseKernel(
        "T x",
        "T y",
        "y = (T)0.5 * x * ((T)1.0 + erf(x * (T)0.7071067811865476))",
        "sg_gelu_erf_fwd",
    )
    # fused backward: grad * d/dx[0.5 * x * (1 + erf(x/sqrt(2)))]
    _gelu_erf_bwd = cp.ElementwiseKernel(
        "T x, T g",
        "T y",
        """
        T e   = erf(x * (T)0.7071067811865476);
        T pdf = exp((T)-0.5 * x * x) * (T)0.3989422804014327;
        y = g * ((T)0.5 * ((T)1.0 + e) + x * pdf);
        """,
        "sg_gelu_erf_bwd",
    )

    def _gelu_erf_cuda_fwd(ctx, x):
        ctx.x = x.values
        return _gelu_erf_fwd(x.values)

    def _gelu_erf_cuda_bwd(ctx, grad):
        return _gelu_erf_bwd(ctx.x, grad)

    _GELUErf.cuda_forward = _gelu_erf_cuda_fwd
    _GELUErf.cuda_backward = _gelu_erf_cuda_bwd


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
    """Apply hyperbolic tangent element-wise, mapping inputs to (-1, 1)."""
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


if _CUPY:
    # fused forward: 1 / (1 + exp(-x))
    _sigmoid_fwd = cp.ElementwiseKernel(
        "T x",
        "T y",
        "y = (T)1.0 / ((T)1.0 + exp(-x))",
        "sg_sigmoid_fwd",
    )
    # fused backward: grad * s * (1 - s)
    _sigmoid_bwd = cp.ElementwiseKernel(
        "T s, T g",
        "T y",
        "y = g * s * ((T)1.0 - s)",
        "sg_sigmoid_bwd",
    )

    def _sigmoid_cuda_fwd(ctx, x):
        out = _sigmoid_fwd(x.values)
        ctx.out = out
        return out

    def _sigmoid_cuda_bwd(ctx, grad):
        return _sigmoid_bwd(ctx.out, grad)

    _Sigmoid.cuda_forward = _sigmoid_cuda_fwd
    _Sigmoid.cuda_backward = _sigmoid_cuda_bwd


def sigmoid(x: Tensor) -> Tensor:
    """Apply sigmoid activation element-wise: 1 / (1 + exp(-x)), mapping inputs to (0, 1)."""
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

    """
    return _ELU.apply(x, alpha, oper=f"ELU(a={alpha})")


class _GELUTanh(Function):
    """GELU with tanh approximation: 0.5 * x * (1 + tanh(c * (x + 0.044715 * x^3)))."""

    oper = "GELUTanh"

    @staticmethod
    def forward(ctx: Context, x: Tensor) -> np.ndarray:
        xp = ctx.backend
        c = 0.7978845608028654  # sqrt(2/pi)
        ctx.x = x.values
        inner = c * (x.values + 0.044715 * x.values**3)
        ctx.t = xp.tanh(inner)
        return 0.5 * x.values * (1.0 + ctx.t)

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> np.ndarray:
        c = 0.7978845608028654
        sech2 = 1.0 - ctx.t**2
        dtanh_dx = c * (1.0 + 3.0 * 0.044715 * ctx.x**2)
        dy_dx = 0.5 * (1.0 + ctx.t) + 0.5 * ctx.x * sech2 * dtanh_dx
        return grad_output * dy_dx


if _CUPY:
    # fused forward: 0.5 * x * (1 + tanh(c * (x + 0.044715 * x^3)))
    _gelu_tanh_fwd = cp.ElementwiseKernel(
        "T x",
        "T y",
        """
        const T c = (T)0.7978845608028654;
        T inner = c * (x + (T)0.044715 * x * x * x);
        T t = tanh(inner);
        y = (T)0.5 * x * ((T)1.0 + t);
        """,
        "sg_gelu_tanh_fwd",
    )
    # fused backward
    _gelu_tanh_bwd = cp.ElementwiseKernel(
        "T x, T g",
        "T y",
        """
        const T c = (T)0.7978845608028654;
        T inner    = c * (x + (T)0.044715 * x * x * x);
        T t        = tanh(inner);
        T sech2    = (T)1.0 - t * t;
        T dtanh_dx = c * ((T)1.0 + (T)3.0 * (T)0.044715 * x * x);
        y = g * ((T)0.5 * ((T)1.0 + t) + (T)0.5 * x * sech2 * dtanh_dx);
        """,
        "sg_gelu_tanh_bwd",
    )

    def _gelu_tanh_cuda_fwd(ctx, x):
        ctx.x = x.values
        return _gelu_tanh_fwd(x.values)

    def _gelu_tanh_cuda_bwd(ctx, grad):
        return _gelu_tanh_bwd(ctx.x, grad)

    _GELUTanh.cuda_forward = _gelu_tanh_cuda_fwd
    _GELUTanh.cuda_backward = _gelu_tanh_cuda_bwd


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
    """
    if mode == "tanh":
        return _GELUTanh.apply(x)
    if mode == "erf":
        return _GELUErf.apply(x)
    raise ValueError(f"Unknown GELU mode '{mode}'. Expected 'erf' or 'tanh'.")


class _Softmax(Function):
    oper = "Softmax"

    @staticmethod
    def forward(ctx: Context, x: Tensor, dim: int | None) -> np.ndarray:
        xp = ctx.backend
        vals = x.values
        ctx.axis = dim % vals.ndim if dim is not None else None
        if ctx.axis is None:
            max_val = xp.max(vals)
            shifted = xp.exp(vals - max_val)
            sum_exp = xp.sum(shifted)
        else:
            max_val = xp.max(vals, axis=ctx.axis, keepdims=True)
            shifted = xp.exp(vals - max_val)
            sum_exp = xp.sum(shifted, axis=ctx.axis, keepdims=True)
        out = shifted / sum_exp
        ctx.out = out
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> np.ndarray:
        xp = ctx.backend
        if ctx.axis is None:
            dot = xp.sum(grad_output * ctx.out)
        else:
            dot = xp.sum(grad_output * ctx.out, axis=ctx.axis, keepdims=True)
        return ctx.out * (grad_output - dot)


if _CUPY:
    # fuses exp(x - max) + sum into one kernel pass
    _softmax_sumexp = cp.ReductionKernel(
        "T x, T m",
        "T y",
        "exp(x - m)",
        "a + b",
        "y = a",
        "0",
        "sg_softmax_sumexp",
    )
    # normalise: exp(x - max) / sum
    _softmax_norm = cp.ElementwiseKernel(
        "T x, T m, T s",
        "T y",
        "y = exp(x - m) / s",
        "sg_softmax_norm",
    )
    # fuses grad * softmax + sum into one kernel pass
    _softmax_bwd_dot = cp.ReductionKernel(
        "T g, T s",
        "T y",
        "g * s",
        "a + b",
        "y = a",
        "0",
        "sg_softmax_bwd_dot",
    )
    # softmax * (grad - dot)
    _softmax_bwd = cp.ElementwiseKernel(
        "T g, T s, T d",
        "T y",
        "y = s * (g - d)",
        "sg_softmax_bwd",
    )

    def _softmax_cuda_fwd(ctx, x, dim):
        vals = x.values
        if dim is None:
            max_val = cp.max(vals)
            shifted = cp.exp(vals - max_val)
            sum_exp = cp.sum(shifted)
            ctx.out = shifted / sum_exp
            ctx.axis = None
        else:
            ctx.axis = dim % vals.ndim
            max_val = cp.max(vals, axis=ctx.axis, keepdims=True)
            sum_exp = _softmax_sumexp(vals, max_val, axis=ctx.axis, keepdims=True)
            ctx.out = _softmax_norm(vals, max_val, sum_exp)
        return ctx.out

    def _softmax_cuda_bwd(ctx, grad):
        if ctx.axis is None:
            dot = cp.sum(grad * ctx.out)
            return ctx.out * (grad - dot)
        dot = _softmax_bwd_dot(grad, ctx.out, axis=ctx.axis, keepdims=True)
        return _softmax_bwd(grad, ctx.out, dot)

    _Softmax.cuda_forward = _softmax_cuda_fwd
    _Softmax.cuda_backward = _softmax_cuda_bwd


def softmax(x: Tensor, dim: int | None = None) -> Tensor:
    """Apply softmax along the given dimension.

    Softmax converts a vector of real values into a probability distribution:
    each output is in (0, 1) and the values along ``dim`` sum to 1. It is
    defined as exp(x_i) / sum_j(exp(x_j)).

    Numerically stable: subtracts max(x) along ``dim`` before exponentiation.

    Args:
        x: Input tensor of any shape.
        dim: Dimension to normalize over. If None, normalizes over all elements.

    """
    return _Softmax.apply(x, dim)
