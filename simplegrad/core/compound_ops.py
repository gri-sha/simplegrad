"""Graph grouping utilities for compound operations."""

from contextlib import contextmanager
import functools

_CURRENT_GROUP: tuple[str, int] | None = None
_GROUP_COUNTER: int = 0


def get_current_group() -> tuple[str, int] | None:
    """Return the current graph group identifier, or None if no group is active."""
    return _CURRENT_GROUP


@contextmanager
def graph_group(name: str):
    """Context manager that tags all Tensors created inside with a named group.

    Used internally by the ``compound_op`` decorator.
    Every Tensor created while this context is active receives a ``group`` attribute of ``(name, unique_id)``,
    which the graph renderer uses to draw a labelled rectangle around all nodes belonging to the same compound operation call.

    Args:
        name: Display name for the group, shown as a label on the cluster
            rectangle in the computation graph.

    Example:
        >>> with graph_group("softmax"):
        ...     exps = exp(x)
        ...     result = exps / sum(exps, dim)
    """
    global _CURRENT_GROUP, _GROUP_COUNTER
    _GROUP_COUNTER += 1
    _CURRENT_GROUP = (name, _GROUP_COUNTER)
    try:
        yield
    finally:
        _CURRENT_GROUP = None


def compound_op(func):
    """Decorator for functions that compose multiple simplegrad operations.

    Apply to any public function that builds its output by calling other simplegrad ops rather than calling numpy directly.
    When the function is called, all Tensors created during its execution are tagged with a shared group identifier.
    The computation graph renderer draws a black-border rectangle around those nodes, labelled with the function name.

    Each call to the decorated function creates a new unique group, so two calls to ``softmax`` produce two separate rectangles in the graph.

    Args:
        func: The compound function to decorate. Its ``__name__`` is used as the cluster label.

    Returns:
        Wrapped function with identical signature and docstring.

    Example:
        >>> @compound_op
        ... def softmax(x: Tensor, dim: int | None = None) -> Tensor:
        ...     exps = exp(x)
        ...     return exps / sum(exps, dim)
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with graph_group(func.__name__):
            return func(*args, **kwargs)

    return wrapper
