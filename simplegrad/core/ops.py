import numpy as np
from .tensor import Tensor

def log(tensor):
    if np.any(tensor.values <= 0):
        raise ValueError("Log of negative value is undefined")

    out = Tensor(np.log(tensor.values))
    out.prev = {tensor}
    out.oper = "log"

    def backward_step():
        if tensor.comp_grad:
            tensor.grad += out.grad / tensor.values

    out.backward_step = backward_step
    return out

def exp(tensor):
    out = Tensor(np.exp(tensor.values))
    out.prev = {tensor}
    out.oper = "exp"

    def backward_step():
        if tensor.comp_grad:
            tensor.grad += out.grad * tensor.values

    out.backward_step = backward_step
    return out

def sum(tensor, dim=None):
    # dim 0: sum columns, resulting in a single row
    # dim 1: sum rows, resulting in a single column
    # etc.
    out = Tensor(np.sum(tensor.values, axis=dim, keepdims=True))
    out.prev = {tensor}
    out.oper = f"sum(d={dim})"

    def backward_step():
        if tensor.comp_grad:
            tensor.grad += np.ones_like(tensor.values) * out.grad

    out.backward_step = backward_step
    return out

def trace(tensor):
    if tensor.values.ndim != 2 or tensor.values.shape[0] != tensor.values.shape[1]:
        raise ValueError("Trace is only defined for square matrices")

    out = Tensor(np.array([[np.trace(tensor.values)]]))
    out.prev = {tensor}
    out.oper = "trace"

    def backward_step():
        if tensor.comp_grad:
            grad_matrix = np.zeros_like(tensor.values)
            np.fill_diagonal(grad_matrix, out.grad.flatten())
            tensor.grad += grad_matrix

    out.backward_step = backward_step
    return out