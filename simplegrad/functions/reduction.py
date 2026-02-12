import numpy as np
from simplegrad.core.tensor import Tensor, _should_compute_grad
from typing import Optional
from simplegrad.dtypes import get_dtype_class


def sum(x: Tensor, dim: Optional[int] = None) -> Tensor:
    # dim 0: sum columns, resulting in a single row
    # dim 1: sum rows, resulting in a single column
    # etc.
    out = Tensor(np.sum(x.values, axis=dim, keepdims=True))
    out.prev = {x}
    out.oper = f"sum(d={dim})"
    out.comp_grad = _should_compute_grad(x)
    out.is_leaf = False

    if out.comp_grad:
        out.backward_step = lambda: _sum_backward(x=x, out=out)
    return out


def _sum_backward(x: Tensor, out: Tensor) -> None:
    if x.comp_grad:
        x._init_grad_if_needed()
        x.grad += np.ones_like(x.values) * out.grad


def trace(x: Tensor) -> Tensor:
    if x.values.ndim != 2 or x.values.shape[0] != x.values.shape[1]:
        raise ValueError("Trace is only defined for square matrices")

    out = Tensor(np.array([[np.trace(x.values)]]))
    out.prev = {x}
    out.oper = "trace"
    out.comp_grad = _should_compute_grad(x)
    out.is_leaf = False

    if out.comp_grad:
        out.backward_step = lambda: _trace_backward(x=x, out=out)
    return out


def _trace_backward(x: Tensor, out: Tensor) -> None:
    if x.comp_grad:
        x._init_grad_if_needed()
        grad_matrix = np.zeros_like(x.values)
        np.fill_diagonal(grad_matrix, out.grad.flatten())
        x.grad += grad_matrix


def mean(x: Tensor, dim: Optional[int] = None) -> Tensor:
    if dim is None:
        return sum(x) / x.values.size
    return sum(x, dim=dim) / x.values.shape[dim]


def argmax(x: Tensor, dim: Optional[int] = None, dtype: str = "int32") -> Tensor:
    out = Tensor(np.argmax(x.values, axis=dim), dtype=get_dtype_class(dtype))
    out.prev = {x}
    out.oper = f"argmax(d={dim})"
    out.comp_grad = False
    out.is_leaf = False

    if out.comp_grad:
        out.backward_step = lambda: _argmax_backward()
    return out


def _argmax_backward() -> None:
    raise RuntimeError("argmax is not differentiable and does not support backpropagation")


def argmin(x: Tensor, dim: Optional[int] = None, dtype: str = "int32") -> Tensor:
    out = Tensor(np.argmin(x.values, axis=dim), dtype=get_dtype_class(dtype))
    out.prev = {x}
    out.oper = f"argmin(d={dim})"
    out.comp_grad = False
    out.is_leaf = False

    if out.comp_grad:
        out.backward_step = lambda: _argmin_backward()
    return out


def _argmin_backward() -> None:
    raise RuntimeError("argmin is not differentiable and does not support backpropagation")
