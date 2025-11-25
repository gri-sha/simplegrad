import numpy as np
from simplegrad.core.tensor import Tensor


def log(x):
    if np.any(x.values <= 0):
        raise ValueError("Log of negative value is undefined")

    out = Tensor(np.log(x.values))
    out.prev = {x}
    out.oper = "log"
    out.comp_grad = x.comp_grad
    out.is_leaf = False

    def backward_step():
        if x.comp_grad:
            x._init_grad_if_needed()
            x.grad += out.grad / x.values

    out.backward_step = backward_step
    return out


def exp(x):
    out = Tensor(np.exp(x.values))
    out.prev = {x}
    out.oper = "exp"
    out.comp_grad = x.comp_grad
    out.is_leaf = False

    def backward_step():
        if x.comp_grad:
            x._init_grad_if_needed()
            x.grad += out.grad * np.exp(x.values)

    out.backward_step = backward_step
    return out


def sin(x):
    out = Tensor(np.sin(x.values))
    out.prev = {x}
    out.oper = "sin"
    out.comp_grad = x.comp_grad
    out.is_leaf = False

    def backward_step():
        if x.comp_grad:
            x._init_grad_if_needed()
            x.grad += out.grad * np.cos(x.values)

    out.backward_step = backward_step
    return out


def cos(x):
    out = Tensor(np.cos(x.values))
    out.prev = {x}
    out.oper = "cos"
    out.comp_grad = x.comp_grad
    out.is_leaf = False

    def backward_step():
        if x.comp_grad:
            x._init_grad_if_needed()
            x.grad += -out.grad * np.sin(x.values)

    out.backward_step = backward_step
    return out


def tan(x):
    out = Tensor(np.tan(x.values))
    out.prev = {x}
    out.oper = "tan"
    out.comp_grad = x.comp_grad
    out.is_leaf = False

    def backward_step():
        if x.comp_grad:
            x._init_grad_if_needed()
            x.grad += out.grad / (np.cos(x.values) ** 2)

    out.backward_step = backward_step
    return out


def sum(x, dim=None):
    # dim 0: sum columns, resulting in a single row
    # dim 1: sum rows, resulting in a single column
    # etc.
    out = Tensor(np.sum(x.values, axis=dim, keepdims=True))
    out.prev = {x}
    out.oper = f"sum(d={dim})"
    out.comp_grad = x.comp_grad
    out.is_leaf = False

    def backward_step():
        if x.comp_grad:
            x._init_grad_if_needed()
            x.grad += np.ones_like(x.values) * out.grad

    out.backward_step = backward_step
    return out


def trace(x):
    if x.values.ndim != 2 or x.values.shape[0] != x.values.shape[1]:
        raise ValueError("Trace is only defined for square matrices")

    out = Tensor(np.array([[np.trace(x.values)]]))
    out.prev = {x}
    out.oper = "trace"
    out.comp_grad = x.comp_grad
    out.is_leaf = False

    def backward_step():
        if x.comp_grad:
            x._init_grad_if_needed()
            grad_matrix = np.zeros_like(x.values)
            np.fill_diagonal(grad_matrix, out.grad.flatten())
            x.grad += grad_matrix

    out.backward_step = backward_step
    return out


def mean(x, dim=None):
    if dim is None:
        return sum(x) / x.values.size
    return sum(x, dim=dim) / x.values.shape[dim]


def argmax(x, dim=None):
    out = Tensor(np.argmax(x.values, axis=dim))
    out.prev = {x}
    out.oper = f"argmax(d={dim})"
    out.comp_grad = False
    out.is_leaf = False

    def backward_step():
        raise RuntimeError(
            "argmax is not differentiable and does not support backpropagation"
        )

    out.backward_step = backward_step
    return out
