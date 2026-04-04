"""Factory functions for creating common Tensor initializations."""

import numpy as np
from .autograd import Tensor


def zeros(
    shape: tuple[int, ...],
    dtype: str = "float32",
    comp_grad: bool | None = None,
    label: bool | None = None,
) -> Tensor:
    """Create a tensor filled with zeros.

    Args:
        shape: Output shape.
        dtype: Data type string. Defaults to ``"float32"``.
        comp_grad: Enable gradient tracking. Defaults to the global flag.
        label: Optional name for visualization.
    """
    return Tensor(values=np.zeros(shape), dtype=dtype, comp_grad=comp_grad, label=label)


def ones(
    shape: tuple[int, ...],
    dtype: str = "float32",
    comp_grad: bool | None = None,
    label: bool | None = None,
) -> Tensor:
    """Create a tensor filled with ones.

    Args:
        shape: Output shape.
        dtype: Data type string. Defaults to ``"float32"``.
        comp_grad: Enable gradient tracking. Defaults to the global flag.
        label: Optional name for visualization.
    """
    return Tensor(values=np.ones(shape), dtype=dtype, comp_grad=comp_grad, label=label)


def normal(
    shape: tuple[int, ...],
    dtype: "str" = "float32",
    comp_grad: bool | None = None,
    label: bool | None = None,
    mu: int | float = 0,
    sigma: int | float = 1,
) -> Tensor:
    """Create a tensor sampled from a normal distribution N(mu, sigma).

    Args:
        shape: Output shape.
        dtype: Data type string. Defaults to ``"float32"``.
        comp_grad: Enable gradient tracking. Defaults to the global flag.
        label: Optional name for visualization.
        mu: Distribution mean.
        sigma: Distribution standard deviation.
    """
    return Tensor(
        values=np.random.normal(size=shape, loc=mu, scale=sigma),
        dtype=dtype,
        comp_grad=comp_grad,
        label=label,
    )


def uniform(
    shape: tuple[int, ...],
    dtype: str = "float32",
    comp_grad: bool | None = None,
    label: bool | None = None,
    low: int | float = 0,
    high: int | float = 1,
) -> Tensor:
    """Create a tensor sampled from a uniform distribution U(low, high).

    Args:
        shape: Output shape.
        dtype: Data type string. Defaults to ``"float32"``.
        comp_grad: Enable gradient tracking. Defaults to the global flag.
        label: Optional name for visualization.
        low: Lower bound of the distribution.
        high: Upper bound of the distribution.
    """
    return Tensor(
        values=np.random.uniform(size=shape, low=low, high=high),
        dtype=dtype,
        comp_grad=comp_grad,
        label=label,
    )


def full(
    shape: tuple[int, ...],
    fill_value: float,
    dtype: str = "float32",
    comp_grad: bool | None = None,
    label: bool | None = None,
) -> Tensor:
    """Create a tensor filled with a constant value.

    Args:
        shape: Output shape.
        fill_value: Value to fill the tensor with.
        dtype: Data type string. Defaults to ``"float32"``.
        comp_grad: Enable gradient tracking. Defaults to the global flag.
        label: Optional name for visualization.
    """
    return Tensor(values=np.full(shape, fill_value), dtype=dtype, comp_grad=comp_grad, label=label)
