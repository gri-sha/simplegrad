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
    out = Tensor(np.sum(tensor.values, axis=dim, keepdims=True))
    out.prev = {tensor}
    out.oper = f"sum(d={dim})"

    def backward_step():
        if tensor.comp_grad:
            tensor.grad += np.ones_like(tensor.values) * out.grad

    out.backward_step = backward_step
    return out