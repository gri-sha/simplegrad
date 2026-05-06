"""Shared test utilities."""

from typing import Callable
import numpy as np


def _to_numpy(arr) -> np.ndarray:
    """Return a NumPy array, calling .get() for CuPy arrays."""
    if hasattr(arr, "get"):
        return arr.get()
    return np.asarray(arr)


def fwdcheck(out, expected, atol: float = 1e-6) -> None:
    """Assert that a tensor's forward values match an expected numpy array.

    Args:
        out: Simplegrad tensor whose .values to check.
        expected: Expected values as any array-like (converted via np.asarray).
        atol: Absolute tolerance passed to np.allclose.

    Raises:
        AssertionError: If values differ by more than atol.
    """
    values = _to_numpy(out.values)
    expected = np.asarray(expected)
    assert np.allclose(values, expected, atol=atol), (
        f"Forward mismatch: max diff = {np.max(np.abs(values - expected)):.2e}\n"
        f"  got      = {values}\n"
        f"  expected = {expected}"
    )


def gradcheck(
    fn: Callable,
    inputs: list,
    eps: float = 1e-5,
    atol: float = 1e-5,
    rtol: float = 1e-3,
) -> None:
    """Check analytical gradients against numerical central-difference estimates.

    Uses the central difference formula (f(x+eps) - f(x-eps)) / (2*eps) as the
    numerical reference. All inputs should use float64 for sufficient numerical
    precision; float32 requires larger eps and looser tolerances.

    Args:
        fn: Zero-argument callable that builds and returns a Tensor. Must
            rebuild the computation graph on every call, so all tensor
            operations must happen inside the function body (closure pattern).
        inputs: Tensors whose gradients to verify. All must have comp_grad=True.
        eps: Perturbation size for finite differences.
        atol: Absolute tolerance passed to np.allclose.
        rtol: Relative tolerance passed to np.allclose.

    Raises:
        AssertionError: If any analytical gradient disagrees with its
            numerical estimate within the given tolerances.

    Example:
        >>> a = sg.Tensor(np.array([1.0, 2.0, 3.0]), dtype="float64")
        >>> gradcheck(lambda: sg.sum(a ** 2), [a])
    """
    for inp in inputs:
        inp.grad = None

    out = fn()
    out.zero_grad()
    out.backward()

    analytical = {id(inp): _to_numpy(inp.grad) for inp in inputs}

    for inp in inputs:
        num_grad = np.zeros(inp.shape, dtype=np.float64)
        for idx in np.ndindex(inp.shape):
            orig = float(inp.values[idx])

            inp.values[idx] = orig + eps
            plus = float(fn().values.sum())

            inp.values[idx] = orig - eps
            minus = float(fn().values.sum())

            inp.values[idx] = orig
            num_grad[idx] = (plus - minus) / (2 * eps)

        ag = analytical[id(inp)]
        assert np.allclose(ag, num_grad, atol=atol, rtol=rtol), (
            f"Gradient mismatch for tensor shape {inp.shape}:\n"
            f"  max abs diff = {np.max(np.abs(ag - num_grad)):.2e}\n"
            f"  analytical   = {ag}\n"
            f"  numerical    = {num_grad}"
        )
