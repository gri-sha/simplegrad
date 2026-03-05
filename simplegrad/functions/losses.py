"""Loss functions: cross-entropy and mean squared error."""

import numpy as np
from simplegrad.core.tensor import Tensor, _should_compute_grad
import simplegrad as sg
from typing import Optional


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
    # z - layer output (logits, Tensor)
    # y - target probability distribution (Tensor)
    if dim > 0:
        dim = dim - len(z.values.shape)  # convert to negative dim

    # Softmax
    exps = np.exp(z.values - np.max(z.values, axis=dim, keepdims=True))  # for numerical stability
    s = exps / np.sum(exps, axis=dim, keepdims=True)

    # Cross-entropy loss per sample
    losses = -np.sum(y.values * np.log(s + 1e-12), axis=dim, keepdims=True)  # small epsilon for stability

    out = Tensor(losses)
    out.prev = {z, y}
    out.oper = f"CELoss(dim={dim})"
    out.comp_grad = _should_compute_grad(z)
    out.is_leaf = False

    if out.comp_grad:
        out.backward_step = lambda: _ce_loss_backward(z=z, s=s, y=y, out=out)

    if reduction == "mean":
        return sg.mean(out)
    elif reduction == "sum":
        return sg.sum(out)
    elif reduction is None:
        return out
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


def _ce_loss_backward(z: Tensor, s: np.ndarray, y: Tensor, out: Tensor) -> None:
    """Backward for cross-entropy: d/dz = (softmax(z) - y) * out.grad."""
    if z.comp_grad:
        z._init_grad_if_needed()
        z.grad += (s - y.values) * out.grad


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
        return sg.mean((p - y) ** 2)
    elif reduction == "sum":
        return sg.sum((p - y) ** 2)
    elif reduction is None:
        return (p - y) ** 2
    else:
        raise ValueError(f"Invalid reduction: {reduction}")
