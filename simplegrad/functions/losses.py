"""Loss functions: cross-entropy and mean squared error."""

import numpy as np
from ..core import Tensor, Function, Context, compound_op
from .reduction import sum, mean


class _CELoss(Function):
    @staticmethod
    def output_shape(z: Tensor, y: Tensor, dim: int) -> tuple:
        return tuple(1 if i == dim % len(z.shape) else s for i, s in enumerate(z.shape))

    @staticmethod
    def forward(ctx: Context, z: Tensor, y: Tensor, dim: int) -> np.ndarray:
        exps = np.exp(z.values - np.max(z.values, axis=dim, keepdims=True))
        ctx.s = exps / np.sum(exps, axis=dim, keepdims=True)
        ctx.y_values = y.values
        return -np.sum(y.values * np.log(ctx.s + 1e-12), axis=dim, keepdims=True)

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

    Returns:
        Scalar loss tensor (or per-sample if ``reduction=None``).

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


@compound_op
def mse_loss(p: Tensor, y: Tensor, reduction: str = "mean") -> Tensor:
    """Compute mean squared error loss: mean((p - y)^2).

    Args:
        p: Predictions tensor.
        y: Targets tensor, same shape as ``p``.
        reduction: One of ``"mean"``, ``"sum"``, or ``None``.

    Returns:
        Scalar loss tensor (or element-wise if ``reduction=None``).

    Raises:
        ValueError: If ``reduction`` is not a valid option.
    """
    if reduction == "mean":
        return mean((p - y) ** 2)
    elif reduction == "sum":
        return sum((p - y) ** 2)
    elif reduction is None:
        return (p - y) ** 2
    else:
        raise ValueError(f"Invalid reduction: {reduction}")
