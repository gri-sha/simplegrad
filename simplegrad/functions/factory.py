import numpy as np
from simplegrad.core.tensor import Tensor
from typing import Optional


def zeros(shape: tuple[int], dtype: "str" = "float32", comp_grad: bool = True, label: Optional[bool] = None) -> Tensor:
    return Tensor(values=np.zeros(shape), dtype=dtype, comp_grad=comp_grad, label=label)


def ones(shape: tuple[int], dtype: "str" = "float32", comp_grad: bool = True, label: Optional[bool] = None) -> Tensor:
    return Tensor(values=np.ones(shape), dtype=dtype, comp_grad=comp_grad, label=label)


def random(shape: tuple[int], dtype: "str" = "float32", comp_grad: bool = True, label: Optional[bool] = None) -> Tensor:
    return Tensor(values=np.random.standard_normal(shape), dtype=dtype, comp_grad=comp_grad, label=label)
