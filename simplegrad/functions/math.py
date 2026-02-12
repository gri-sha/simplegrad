import numpy as np
from simplegrad.core.tensor import Tensor, _should_compute_grad


def log(x: Tensor) -> Tensor:
    if np.any(x.values <= 0):
        raise ValueError("Log of negative value is undefined")

    out = Tensor(np.log(x.values))
    out.prev = {x}
    out.oper = "log"
    out.comp_grad = _should_compute_grad(x)
    out.is_leaf = False

    if out.comp_grad:
        out.backward_step = lambda: _log_backward(x, out)
    return out


def _log_backward(x: Tensor, out: Tensor):
    if x.comp_grad:
        x._init_grad_if_needed()
        x.grad += out.grad / x.values


def exp(x: Tensor) -> Tensor:
    out = Tensor(np.exp(x.values))
    out.prev = {x}
    out.oper = "exp"
    out.comp_grad = _should_compute_grad(x)
    out.is_leaf = False

    if out.comp_grad:
        out.backward_step = lambda: _exp_backward(x, out)

    return out


def _exp_backward(x: Tensor, out: Tensor):
    if x.comp_grad:
        x._init_grad_if_needed()
        x.grad += out.grad * np.exp(x.values)


def sin(x: Tensor) -> Tensor:
    out = Tensor(np.sin(x.values))
    out.prev = {x}
    out.oper = "sin"
    out.comp_grad = _should_compute_grad(x)
    out.is_leaf = False

    if out.comp_grad:
        out.backward_step = lambda: _sin_backward(x, out)
    return out


def _sin_backward(x: Tensor, out: Tensor):
    if x.comp_grad:
        x._init_grad_if_needed()
        x.grad += out.grad * np.cos(x.values)


def cos(x: Tensor) -> Tensor:
    out = Tensor(np.cos(x.values))
    out.prev = {x}
    out.oper = "cos"
    out.comp_grad = _should_compute_grad(x)
    out.is_leaf = False

    if out.comp_grad:
        out.backward_step = lambda: _cos_backward(x, out)
    return out


def _cos_backward(x: Tensor, out: Tensor):
    if x.comp_grad:
        x._init_grad_if_needed()
        x.grad += -out.grad * np.sin(x.values)


def tan(x: Tensor) -> Tensor:
    out = Tensor(np.tan(x.values))
    out.prev = {x}
    out.oper = "tan"
    out.comp_grad = _should_compute_grad(x)
    out.is_leaf = False

    if out.comp_grad:
        out.backward_step = lambda: _tan_backward(x, out)
    return out


def _tan_backward(x: Tensor, out: Tensor):
    if x.comp_grad:
        x._init_grad_if_needed()
        x.grad += out.grad / (np.cos(x.values) ** 2)
