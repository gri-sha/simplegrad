"""Tensor shape transformation functions: flatten and reshape."""

from simplegrad.core.tensor import Tensor, _should_compute_grad


# both start and end are flattened
def flatten(x: Tensor, start_dim: int = 0, end_dim: int = -1) -> Tensor:
    """Flatten a range of dimensions into a single dimension.

    Args:
        x: Input tensor.
        start_dim: First dimension to flatten (inclusive). Supports negative indexing.
        end_dim: Last dimension to flatten (inclusive). Supports negative indexing.

    Returns:
        Tensor with dimensions ``[start_dim, end_dim]`` merged into one.
    """
    if end_dim < 0:
        end_dim = len(x.values.shape) + end_dim
    if start_dim < 0:
        start_dim = len(x.values.shape) + start_dim

    out = Tensor(x.values.reshape(*x.values.shape[:start_dim], -1, *x.values.shape[end_dim + 1 :]))
    out.prev = {x}
    out.oper = "Flatten"
    out.comp_grad = _should_compute_grad(x)
    out.is_leaf = False
    if out.comp_grad:
        out.backward_step = lambda: _flatten_backward(x, out)
    return out


def _flatten_backward(x: Tensor, out: Tensor) -> None:
    """Backward for flatten: reshape gradient back to the original shape."""
    if x.comp_grad:
        x.grad = out.grad.reshape(x.values.shape)


def reshape(x: Tensor, new_shape: tuple[int, ...]) -> Tensor:
    """Reshape a tensor to a new shape.

    Args:
        x: Input tensor.
        new_shape: Target shape. Total number of elements must match.

    Returns:
        Tensor with values laid out in ``new_shape``.
    """
    out = Tensor(x.values.reshape(new_shape))
    out.prev = {x}
    out.oper = f"reshape({new_shape})"
    out.comp_grad = _should_compute_grad(x)
    out.is_leaf = False

    if out.comp_grad:
        out.backward_step = lambda: _reshape_backward(x, out)
    return out


def _reshape_backward(x: Tensor, out: Tensor) -> None:
    """Backward for reshape: reshape gradient back to the original shape."""
    if x.comp_grad:
        x.grad = out.grad.reshape(x.values.shape)
