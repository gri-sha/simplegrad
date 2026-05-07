"""Generalized normalization function."""

import numpy as np
from ..core import Tensor, Function, Context
from ..core.devices import _CUPY, cp


class _Norm(Function):
    oper = "Norm"

    @staticmethod
    def forward(ctx: Context, x: Tensor, dims, weight, bias, eps: float, correction: int):
        xp = ctx.backend
        vals = x.values
        ndim = vals.ndim
        axes = tuple(d % ndim for d in dims)
        ctx.axes = axes
        ctx.eps = eps
        ctx.has_weight = weight is not None
        ctx.has_bias = bias is not None

        mu = xp.mean(vals, axis=axes, keepdims=True)
        var = xp.mean((vals - mu) ** 2, axis=axes, keepdims=True)
        if correction:
            N = 1
            for a in axes:
                N *= vals.shape[a]
            var = var * N / (N - correction)

        sigma = xp.sqrt(var + eps)
        x_hat = (vals - mu) / sigma
        ctx.x_hat = x_hat
        ctx.sigma = sigma

        out = x_hat
        if weight is not None:
            ctx.weight_vals = weight.values
            out = out * weight.values
        else:
            ctx.weight_vals = None
        if bias is not None:
            out = out + bias.values
        return out

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray):
        xp = ctx.backend
        d_gamma_dy = grad * ctx.weight_vals if ctx.has_weight else grad

        mean_d = xp.mean(d_gamma_dy, axis=ctx.axes, keepdims=True)
        mean_d_xhat = xp.mean(d_gamma_dy * ctx.x_hat, axis=ctx.axes, keepdims=True)
        dx = (d_gamma_dy - mean_d - ctx.x_hat * mean_d_xhat) / ctx.sigma

        if ctx.has_weight and ctx.has_bias:
            return dx, grad * ctx.x_hat, grad
        if ctx.has_weight:
            return dx, grad * ctx.x_hat
        if ctx.has_bias:
            return dx, grad
        return dx


if _CUPY:
    # fused mean: sum(x) / _size
    _norm_mean = cp.ReductionKernel(
        "T x",
        "T y",
        "x",
        "a + b",
        "y = a / (T)_size",
        "0",
        "sg_norm_mean",
    )
    # fused variance: mean((x - mu)^2)
    _norm_var = cp.ReductionKernel(
        "T x, T mu",
        "T y",
        "(x - mu) * (x - mu)",
        "a + b",
        "y = a / (T)_size",
        "0",
        "sg_norm_var",
    )
    # fused normalize: (x - mu) / sqrt(var + eps)
    _norm_normalize = cp.ElementwiseKernel(
        "T x, T mu, T var, T eps",
        "T xhat",
        "xhat = (x - mu) / sqrt(var + eps)",
        "sg_norm_normalize",
    )
    # fused mean of element-wise product: mean(a * b)
    _norm_mean_dot = cp.ReductionKernel(
        "T x, T z",
        "T y",
        "x * z",
        "a + b",
        "y = a / (T)_size",
        "0",
        "sg_norm_mean_dot",
    )
    # fused dx: (dg - mean_dg - xhat * mean_dg_xhat) / sigma
    _norm_bwd_dx = cp.ElementwiseKernel(
        "T dg, T mean_dg, T xhat, T mean_dg_xhat, T sigma",
        "T dx",
        "dx = (dg - mean_dg - xhat * mean_dg_xhat) / sigma",
        "sg_norm_bwd_dx",
    )

    def _norm_cuda_fwd(ctx, x, dims, weight, bias, eps, correction):
        vals = x.values
        ndim = vals.ndim
        axes = tuple(d % ndim for d in dims)
        ctx.axes = axes
        ctx.eps = eps
        ctx.has_weight = weight is not None
        ctx.has_bias = bias is not None

        mu = _norm_mean(vals, axis=axes, keepdims=True)
        var = _norm_var(vals, mu, axis=axes, keepdims=True)
        if correction:
            N = 1
            for a in axes:
                N *= vals.shape[a]
            var = var * N / (N - correction)

        x_hat = _norm_normalize(vals, mu, var, eps)
        ctx.x_hat = x_hat
        ctx.sigma = cp.sqrt(var + eps)

        out = x_hat
        if weight is not None:
            ctx.weight_vals = weight.values
            out = out * weight.values
        else:
            ctx.weight_vals = None
        if bias is not None:
            out = out + bias.values
        return out

    def _norm_cuda_bwd(ctx, grad):
        d_gamma_dy = grad * ctx.weight_vals if ctx.has_weight else grad

        mean_d = _norm_mean(d_gamma_dy, axis=ctx.axes, keepdims=True)
        mean_d_xhat = _norm_mean_dot(d_gamma_dy, ctx.x_hat, axis=ctx.axes, keepdims=True)
        dx = _norm_bwd_dx(d_gamma_dy, mean_d, ctx.x_hat, mean_d_xhat, ctx.sigma)

        if ctx.has_weight and ctx.has_bias:
            return dx, grad * ctx.x_hat, grad
        if ctx.has_weight:
            return dx, grad * ctx.x_hat
        if ctx.has_bias:
            return dx, grad
        return dx

    _Norm.cuda_forward = _norm_cuda_fwd
    _Norm.cuda_backward = _norm_cuda_bwd


def norm(
    x: Tensor,
    dims: list[int],
    *,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    eps: float = 1e-5,
    correction: int = 0,
) -> Tensor:
    """Normalize a tensor over arbitrary dimensions with optional affine transform.

    This is a generalization of the common normalization layers: by choosing
    ``dims`` appropriately you get different norm variants:

    - **Layer norm**: ``dims = [-1]`` or the last N dims (normalizes each sample
      independently over its features).
    - **Batch norm** (approximate): ``dims = [0]`` (normalizes over the batch).
    - **Instance norm**: ``dims = [-2, -1]`` (normalizes each channel's spatial
      map independently).

    The normalization formula is::

        x_hat = (x - mean(x, dims)) / sqrt(variance(x, dims) + eps)

    followed by an optional affine rescaling ``x_hat * weight + bias``.

    Args:
        x: Input tensor of any shape.
        dims: List of axes to normalize over.
        weight: Optional learnable scale (γ). Shape must broadcast to ``x.shape``.
            Typically ``ones`` initialized. If ``None``, no scaling is applied.
        bias: Optional learnable shift (β). Shape must broadcast to ``x.shape``.
            Typically ``zeros`` initialized. If ``None``, no shift is applied.
        eps: Small constant added to the variance before taking the square root,
            preventing division by zero. Default ``1e-5``.
        correction: Degrees-of-freedom correction applied to the variance denominator.
            ``0`` (default) uses population variance, which is standard for norm layers.
            ``1`` uses Bessel's correction (unbiased sample variance).

    """
    return _Norm.apply(x, dims, weight, bias, eps, correction)
