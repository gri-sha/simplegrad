"""Factory functions for creating common Tensor initializations."""

from .autograd import Tensor
from .devices import get_backend, get_default_device


def zeros(
    shape: tuple[int, ...],
    dtype: str = "float32",
    comp_grad: bool | None = None,
    label: bool | None = None,
    device: str | None = None,
) -> Tensor:
    """Create a tensor filled with zeros.

    Args:
        shape: Output shape.
        dtype: Data type string. Defaults to ``"float32"``.
        comp_grad: Enable gradient tracking. Defaults to the global flag.
        label: Optional name for visualization.
        device: Device string. Defaults to the current global default device.
    """
    dev = device if device is not None else get_default_device()
    xp = get_backend(dev)
    return Tensor(values=xp.zeros(shape), dtype=dtype, comp_grad=comp_grad, label=label, device=dev)


def ones(
    shape: tuple[int, ...],
    dtype: str = "float32",
    comp_grad: bool | None = None,
    label: bool | None = None,
    device: str | None = None,
) -> Tensor:
    """Create a tensor filled with ones.

    Args:
        shape: Output shape.
        dtype: Data type string. Defaults to ``"float32"``.
        comp_grad: Enable gradient tracking. Defaults to the global flag.
        label: Optional name for visualization.
        device: Device string. Defaults to the current global default device.
    """
    dev = device if device is not None else get_default_device()
    xp = get_backend(dev)
    return Tensor(values=xp.ones(shape), dtype=dtype, comp_grad=comp_grad, label=label, device=dev)


def normal(
    shape: tuple[int, ...],
    dtype: "str" = "float32",
    comp_grad: bool | None = None,
    label: bool | None = None,
    mu: int | float = 0,
    sigma: int | float = 1,
    device: str | None = None,
) -> Tensor:
    """Create a tensor sampled from a normal distribution N(mu, sigma).

    Args:
        shape: Output shape.
        dtype: Data type string. Defaults to ``"float32"``.
        comp_grad: Enable gradient tracking. Defaults to the global flag.
        label: Optional name for visualization.
        mu: Distribution mean.
        sigma: Distribution standard deviation.
        device: Device string. Defaults to the current global default device.
    """
    dev = device if device is not None else get_default_device()
    xp = get_backend(dev)
    return Tensor(
        values=xp.random.normal(size=shape, loc=mu, scale=sigma),
        dtype=dtype,
        comp_grad=comp_grad,
        label=label,
        device=dev,
    )


def uniform(
    shape: tuple[int, ...],
    dtype: str = "float32",
    comp_grad: bool | None = None,
    label: bool | None = None,
    low: int | float = 0,
    high: int | float = 1,
    device: str | None = None,
) -> Tensor:
    """Create a tensor sampled from a uniform distribution U(low, high).

    Args:
        shape: Output shape.
        dtype: Data type string. Defaults to ``"float32"``.
        comp_grad: Enable gradient tracking. Defaults to the global flag.
        label: Optional name for visualization.
        low: Lower bound of the distribution.
        high: Upper bound of the distribution.
        device: Device string. Defaults to the current global default device.
    """
    dev = device if device is not None else get_default_device()
    xp = get_backend(dev)
    return Tensor(
        values=xp.random.uniform(size=shape, low=low, high=high),
        dtype=dtype,
        comp_grad=comp_grad,
        label=label,
        device=dev,
    )


def full(
    shape: tuple[int, ...],
    fill_value: float,
    dtype: str = "float32",
    comp_grad: bool | None = None,
    label: bool | None = None,
    device: str | None = None,
) -> Tensor:
    """Create a tensor filled with a constant value.

    Args:
        shape: Output shape.
        fill_value: Value to fill the tensor with.
        dtype: Data type string. Defaults to ``"float32"``.
        comp_grad: Enable gradient tracking. Defaults to the global flag.
        label: Optional name for visualization.
        device: Device string. Defaults to the current global default device.
    """
    dev = device if device is not None else get_default_device()
    xp = get_backend(dev)
    return Tensor(values=xp.full(shape, fill_value), dtype=dtype, comp_grad=comp_grad, label=label, device=dev)
