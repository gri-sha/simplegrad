import numpy as np
from .tensor import Tensor
from typing import Optional, Union


def zeros(shape: tuple[int], dtype: "str" = "float32", comp_grad: Optional[bool] = None, label: Optional[bool] = None) -> Tensor:
    return Tensor(values=np.zeros(shape), dtype=dtype, comp_grad=comp_grad, label=label)


def ones(shape: tuple[int], dtype: "str" = "float32", comp_grad: Optional[bool] = None, label: Optional[bool] = None) -> Tensor:
    return Tensor(values=np.ones(shape), dtype=dtype, comp_grad=comp_grad, label=label)


def normal(
    shape: tuple[int],
    dtype: "str" = "float32",
    comp_grad: Optional[bool] = None,
    label: Optional[bool] = None,
    mu: Union[int, float] = 0,
    sigma: Union[int, float] = 1,
) -> Tensor:
    return Tensor(values=np.random.normal(size=shape, loc=mu, scale=sigma), dtype=dtype, comp_grad=comp_grad, label=label)


def uniform(
    shape: tuple[int],
    dtype: "str" = "float32",
    comp_grad: Optional[bool] = None,
    label: Optional[bool] = None,
    low: Union[int, float] = 0,
    high: Union[int, float] = 1,
) -> Tensor:
    return Tensor(values=np.random.uniform(size=shape, low=low, high=high), dtype=dtype, comp_grad=comp_grad, label=label)


def full(
    shape: tuple[int], fill_value: float, dtype: "str" = "float32", comp_grad: Optional[bool] = None, label: Optional[bool] = None
) -> Tensor:
    return Tensor(values=np.full(shape, fill_value), dtype=dtype, comp_grad=comp_grad, label=label)
