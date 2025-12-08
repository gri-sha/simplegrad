import numpy as np
from simplegrad.core.tensor import Tensor, _should_compute_grad
import simplegrad as sg
from typing import Optional


def ce_loss(z: Tensor, y: Tensor, dim: int = -1, reduction: str = "mean") -> Tensor:
    # z - layer output (logits, Tensor)
    # y - target probability distribution (Tensor)
    if dim > 0:
        dim = dim - len(z.values.shape)  # convert to negative dim

    # Softmax
    exps = np.exp(z.values - np.max(z.values, axis=dim, keepdims=True))  # for numerical stability
    s = exps / np.sum(exps, axis=dim, keepdims=True)

    # print(s.shape, y.values.shape)

    # Cross-entropy loss per sample
    losses = -np.sum(y.values * np.log(s + 1e-12), axis=dim, keepdims=True)  # small epsilon for stability
    # print(dim)
    # print('input', z.shape)
    # print("losses shape:", losses.shape)

    out = Tensor(losses)
    out.prev = {z, y}
    out.oper = f"CELoss(dim={dim})"
    out.comp_grad = _should_compute_grad(z)
    out.is_leaf = False

    if out.comp_grad:

        def backward_step():
            if z.comp_grad:
                z._init_grad_if_needed()
                grad = s - y.values
                # это костыль, не могу понять почему в некоторых случаях неправильный upstream_grad.shape
                # upstream_grad = out.grad.T if out.grad.ndim <= 2 and out.grad.shape[0] == 1 else np.expand_dims(out.grad, axis=dim)
                # print('out.grad', out.grad.shape)
                # upstream_grad = np.expand_dims(out.grad, axis=dim)
                # print('upstream', upstream_grad.shape)
                # print('grad', grad.shape)
                z.grad += grad * out.grad

        out.backward_step = backward_step

    if reduction == "mean":
        return sg.mean(out)
    elif reduction == "sum":
        return sg.sum(out)
    elif reduction is None:
        return out
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


def mse_loss(p: Tensor, y: Tensor, reduction: str = "mean") -> Tensor:
    if reduction == "mean":
        return sg.mean((p - y) ** 2)
    elif reduction == "sum":
        return sg.sum((p - y) ** 2)
    elif reduction is None:
        return (p - y) ** 2
    else:
        raise ValueError(f"Invalid reduction: {reduction}")
