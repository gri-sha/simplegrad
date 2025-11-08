import numpy as np
from simplegrad.core.tensor import Tensor

def ce_loss(z, y, dim=-1, reduction='mean'):
    # z - layer output (logits, Tensor)
    # y - target probability distribution (Tensor)

    # Softmax
    exps = np.exp(z.values - np.max(z.values, axis=dim, keepdims=True))  # for numerical stability
    s = exps / np.sum(exps, axis=dim, keepdims=True)

    # print(s.shape, y.values.shape)

    # Cross-entropy loss per sample
    losses = -np.sum(y.values * np.log(s + 1e-12), axis=dim)  # small epsilon for stability

    # Reduction
    if reduction == 'mean':
        out_value = np.mean(losses)
    elif reduction == 'sum':
        out_value = np.sum(losses)
    elif reduction is None:
        out_value = losses
    else:
        raise ValueError(f"Invalid reduction: {reduction}")
    
    out = Tensor(out_value)
    out.prev = {z, y}
    out.oper = f"CELoss(reduction={reduction})"
    out.comp_grad = z.comp_grad
    out.is_leaf = False

    # Backward step
    def backward_step():
        if z.comp_grad:
            z._init_grad_if_needed()
            grad = s - y.values
            if reduction == 'mean':
                grad = grad / z.values.shape[0]  # normalize by batch size
            z.grad += grad

    out.backward_step = backward_step
    return out

def mse_loss(p, y, reduction='mean'):
    # p - predicted probabilities (Tensor)
    # y - true values (Tensor)

    # Mean Squared Error per sample
    losses = (p.values - y.values) ** 2

    # Reduction
    if reduction == 'mean':
        out_value = np.mean(losses)
    elif reduction == 'sum':
        out_value = np.sum(losses)
    elif reduction is None:
        out_value = losses
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

    out = Tensor(out_value)
    out.prev = {p, y}
    out.oper = f"MSELoss(reduction={reduction})"
    out.comp_grad = p.comp_grad
    out.is_leaf = False

    # Backward step
    def backward_step():
        if p.comp_grad:
            p._init_grad_if_needed()
            grad = 2 * (p.values - y.values)
            if reduction == 'mean':
                grad = grad / p.values.shape[0]  # normalize by batch size
            p.grad += grad

    out.backward_step = backward_step
    return out