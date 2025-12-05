import numpy as np
from simplegrad.core.tensor import Tensor, _should_compute_grad
from .math import exp, log
from .reduction import sum
from typing import Optional


def relu(x: Tensor) -> Tensor:
    out = Tensor(np.maximum(0, x.values))
    out.prev = {x}
    out.oper = "ReLU"
    out.comp_grad = _should_compute_grad(x)
    out.is_leaf = False

    if out.comp_grad:
        def backward_step():
            if x.comp_grad:
                x._init_grad_if_needed()
                x.grad = out.grad * np.where(x.values > 0, 1.0, 0.0)

        out.backward_step = backward_step
    return out


def softmax(x: Tensor, dim: Optional[int] = None) -> Tensor:
    exps = exp(x)
    return exps / sum(exps, dim)


def tanh(x: Tensor) -> Tensor:
    out = Tensor(np.tanh(x.values))
    out.prev = {x}
    out.oper = "Tanh"
    out.comp_grad = _should_compute_grad(x)
    out.is_leaf = False

    if out.comp_grad:
        def backward_step():
            if x.comp_grad:
                x._init_grad_if_needed()
                x.grad += out.grad * (1 - np.tanh(x.values) ** 2)

        out.backward_step = backward_step
    return out


def sigmoid(x: Tensor) -> Tensor:
    out = Tensor(1 / (1 + np.exp(-x.values)))
    out.prev = {x}
    out.oper = "Sigmoid"
    out.comp_grad = _should_compute_grad(x)
    out.is_leaf = False

    if out.comp_grad:
        def backward_step():
            if x.comp_grad:
                x._init_grad_if_needed()
                sigmoid_x = 1 / (1 + np.exp(-x.values))
                x.grad += out.grad * sigmoid_x * (1 - sigmoid_x)

        out.backward_step = backward_step
    return out
