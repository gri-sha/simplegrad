import numpy as np
from simplegrad.core.tensor import Tensor
from typing import Optional


def zeros(
    shape: tuple[int], req_grad: bool = True, label: Optional[bool] = None
) -> Tensor:
    return Tensor(np.zeros(shape), req_grad, label)


def ones(
    shape: tuple[int], req_grad: bool = True, label: Optional[bool] = None
) -> Tensor:
    return Tensor(np.ones(shape), req_grad, label)


def random(
    shape: tuple[int], req_grad: bool = True, label: Optional[bool] = None
) -> Tensor:
    return Tensor(np.random.standard_normal(shape), req_grad, label)
