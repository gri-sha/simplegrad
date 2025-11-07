import numpy as np
from simplegrad.core.tensor import Tensor
from simplegrad.core.ops import exp, log, sum


def relu(x):
    out = Tensor(np.maximum(0, x.values))
    out.prev = {x}
    out.oper = "ReLU"
    out.comp_grad = x.comp_grad
    out.is_leaf = False

    def backward_step():
        if x.comp_grad:
            x._init_grad_if_needed()
            x.grad = out.grad * np.where(x.values > 0, 1.0, 0.0)

    out.backward_step = backward_step
    return out


def softmax(x, dim=None):
    exps = exp(x)
    return exps / sum(exps, dim)

def tanh(x):
    out = Tensor(np.tanh(x.values))
    out.prev = {x}
    out.oper = "Tanh"
    out.comp_grad = x.comp_grad
    out.is_leaf = False

    def backward_step():
        if x.comp_grad:
            x._init_grad_if_needed()
            x.grad += out.grad * (1 - np.tanh(x.values) ** 2)

    out.backward_step = backward_step
    return out
