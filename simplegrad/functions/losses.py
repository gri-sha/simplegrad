import numpy as np
from simplegrad.core.tensor import Tensor
import simplegrad as sg


def ce_loss(z, y, dim=-1, reduction="mean"):
    # z - layer output (logits, Tensor)
    # y - target probability distribution (Tensor)

    # Softmax
    exps = np.exp(
        z.values - np.max(z.values, axis=dim, keepdims=True)
    )  # for numerical stability
    s = exps / np.sum(exps, axis=dim, keepdims=True)

    # print(s.shape, y.values.shape)

    # Cross-entropy loss per sample
    losses = -np.sum(
        y.values * np.log(s + 1e-12), axis=dim
    )  # small epsilon for stability

    out = Tensor(losses)
    out.prev = {z, y}
    out.oper = f"CELoss(dim={dim})"
    out.comp_grad = z.comp_grad
    out.is_leaf = False

    # Backward step
    def backward_step():
        if z.comp_grad:
            z._init_grad_if_needed()
            grad = s - y.values
            z.grad += grad * out.grad.T

    out.backward_step = backward_step

    if reduction == "mean":
        return sg.mean(out)
    elif reduction == "sum":
        return sg.sum(out)
    elif reduction is None:
        return out
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


def mse_loss(p, y, reduction="mean"):
    if reduction == "mean":
        return sg.mean((p - y) ** 2)
    elif reduction == "sum":
        return sg.sum((p - y) ** 2)
    elif reduction is None:
        return (p - y) ** 2
    else:
        raise ValueError(f"Invalid reduction: {reduction}")
