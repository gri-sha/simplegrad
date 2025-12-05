import numpy as np
from simplegrad.core.tensor import Tensor, _should_compute_grad
from typing import Optional
from simplegrad.dtypes import get_dtype_class


def log(x: Tensor) -> Tensor:
    if np.any(x.values <= 0):
        raise ValueError("Log of negative value is undefined")

    out = Tensor(np.log(x.values))
    out.prev = {x}
    out.oper = "log"
    out.comp_grad = _should_compute_grad(x)
    out.is_leaf = False

    if out.comp_grad:
        def backward_step():
            if x.comp_grad:
                x._init_grad_if_needed()
                x.grad += out.grad / x.values

        out.backward_step = backward_step
    return out


def exp(x: Tensor) -> Tensor:
    out = Tensor(np.exp(x.values))
    out.prev = {x}
    out.oper = "exp"
    out.comp_grad = _should_compute_grad(x)
    out.is_leaf = False

    if out.comp_grad:
        def backward_step():
            if x.comp_grad:
                x._init_grad_if_needed()
                x.grad += out.grad * np.exp(x.values)

        out.backward_step = backward_step
    return out


def sin(x: Tensor) -> Tensor:
    out = Tensor(np.sin(x.values))
    out.prev = {x}
    out.oper = "sin"
    out.comp_grad = _should_compute_grad(x)
    out.is_leaf = False

    if out.comp_grad:
        def backward_step():
            if x.comp_grad:
                x._init_grad_if_needed()
                x.grad += out.grad * np.cos(x.values)

        out.backward_step = backward_step
    return out


def cos(x: Tensor) -> Tensor:
    out = Tensor(np.cos(x.values))
    out.prev = {x}
    out.oper = "cos"
    out.comp_grad = _should_compute_grad(x)
    out.is_leaf = False

    if out.comp_grad:
        def backward_step():
            if x.comp_grad:
                x._init_grad_if_needed()
                x.grad += -out.grad * np.sin(x.values)

        out.backward_step = backward_step
    return out


def tan(x: Tensor) -> Tensor:
    out = Tensor(np.tan(x.values))
    out.prev = {x}
    out.oper = "tan"
    out.comp_grad = _should_compute_grad(x)
    out.is_leaf = False

    if out.comp_grad:
        def backward_step():
            if x.comp_grad:
                x._init_grad_if_needed()
                x.grad += out.grad / (np.cos(x.values) ** 2)

        out.backward_step = backward_step
    return out
