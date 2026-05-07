"""Loss functions: cross-entropy and mean squared error."""

import numpy as np
from ..core import Tensor, Function, Context
from ..core.devices import _CUPY, cp
from .reduction import sum, mean


class _CELoss(Function):
    @staticmethod
    def output_shape(z: Tensor, y: Tensor, dim: int) -> tuple:
        return tuple(1 if i == dim % len(z.shape) else s for i, s in enumerate(z.shape))

    @staticmethod
    def forward(ctx: Context, z: Tensor, y: Tensor, dim: int) -> np.ndarray:
        xp = ctx.backend
        exps = xp.exp(z.values - xp.max(z.values, axis=dim, keepdims=True))
        ctx.s = exps / xp.sum(exps, axis=dim, keepdims=True)
        ctx.y_values = y.values
        return -xp.sum(y.values * xp.log(ctx.s + 1e-12), axis=dim, keepdims=True)

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> tuple:
        return (ctx.s - ctx.y_values) * grad_output, None


def ce_loss(z: Tensor, y: Tensor, dim: int = -1, reduction: str = "mean") -> Tensor:
    """Compute cross-entropy loss with built-in softmax.

    Numerically stable: uses the log-sum-exp trick internally.

    Args:
        z: Logits (raw unnormalized scores), shape ``(..., num_classes)``.
        y: Target probability distribution, same shape as ``z``.
        dim: Class dimension to apply softmax over. Defaults to -1 (last dim).
        reduction: How to reduce the per-sample losses. One of ``"mean"``,
            ``"sum"``, or ``None`` (return per-sample losses).

    Raises:
        ValueError: If ``reduction`` is not a valid option.
    """
    if dim > 0:
        dim = dim - len(z.shape)
    out = _CELoss.apply(z, y, dim, oper=f"CELoss(dim={dim})")
    if reduction == "mean":
        return mean(out)
    elif reduction == "sum":
        return sum(out)
    elif reduction is None:
        return out
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


class _MSELoss(Function):
    @staticmethod
    def output_shape(p: Tensor, y: Tensor, reduction) -> tuple:
        if reduction is None:
            return p.shape
        return (1,) * len(p.shape)

    @staticmethod
    def forward(ctx: Context, p: Tensor, y: Tensor, reduction) -> np.ndarray:
        xp = ctx.backend
        diff = p.values - y.values
        ctx.diff = diff
        ctx.reduction = reduction
        sq = diff * diff
        if reduction == "mean":
            ctx.N = diff.size
            return xp.sum(sq).reshape((1,) * diff.ndim) / diff.size
        if reduction == "sum":
            ctx.N = None
            return xp.sum(sq).reshape((1,) * diff.ndim)
        ctx.N = None
        return sq

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray) -> tuple:
        coeff = (2.0 / ctx.N) if ctx.reduction == "mean" else 2.0
        d = grad * coeff * ctx.diff
        return d, -d


if _CUPY:
    # fused (p - y)^2 in one pass
    _mse_sq = cp.ElementwiseKernel(
        "T p, T y",
        "T z",
        "z = (p - y) * (p - y)",
        "sg_mse_sq",
    )
    # reduction sum
    _mse_sum = cp.ReductionKernel(
        "T z",
        "T out",
        "z",
        "a + b",
        "out = a",
        "0",
        "sg_mse_sum",
    )
    # fused backward: 2 * diff * grad, returns both dp and -dp
    _mse_bwd_ew = cp.ElementwiseKernel(
        "T diff, T g",
        "T dp, T dy",
        "dp = (T)2.0 * diff * g; dy = -dp",
        "sg_mse_bwd",
    )

    def _mse_cuda_fwd(ctx, p, y, reduction):
        sq = _mse_sq(p.values, y.values)
        diff = p.values - y.values
        ctx.diff = diff
        ctx.reduction = reduction
        if reduction == "mean":
            ctx.N = diff.size
            return (_mse_sum(sq) / diff.size).reshape((1,) * diff.ndim)
        if reduction == "sum":
            ctx.N = None
            return _mse_sum(sq).reshape((1,) * diff.ndim)
        ctx.N = None
        return sq

    def _mse_cuda_bwd(ctx, grad):
        dp, dy = _mse_bwd_ew(ctx.diff, grad)
        if ctx.reduction == "mean":
            return dp / ctx.N, dy / ctx.N
        return dp, dy

    _MSELoss.cuda_forward = _mse_cuda_fwd
    _MSELoss.cuda_backward = _mse_cuda_bwd


def mse_loss(p: Tensor, y: Tensor, reduction: str = "mean") -> Tensor:
    """Compute mean squared error loss: mean((p - y)^2).

    Args:
        p: Predictions tensor.
        y: Targets tensor, same shape as ``p``.
        reduction: One of ``"mean"``, ``"sum"``, or ``None``.

    Raises:
        ValueError: If ``reduction`` is not a valid option.
    """
    if reduction not in ("mean", "sum", None):
        raise ValueError(f"Invalid reduction: {reduction}")
    return _MSELoss.apply(p, y, reduction, oper=f"MSELoss({reduction})")
