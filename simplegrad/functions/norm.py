"""Generalized normalization function."""

from ..core import Tensor, compound_op
from .reduction import mean, variance


@compound_op
def norm(
    x: Tensor,
    dims: list[int],
    *,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    eps: float = 1e-5,
    correction: int = 0,
) -> Tensor:
    """Normalize a tensor over arbitrary dimensions with optional affine transform.

    This is a generalization of the common normalization layers: by choosing
    ``dims`` appropriately you get different norm variants:

    - **Layer norm**: ``dims = [-1]`` or the last N dims (normalizes each sample
      independently over its features).
    - **Batch norm** (approximate): ``dims = [0]`` (normalizes over the batch).
    - **Instance norm**: ``dims = [-2, -1]`` (normalizes each channel's spatial
      map independently).

    The normalization formula is::

        x_hat = (x - mean(x, dims)) / sqrt(variance(x, dims) + eps)

    followed by an optional affine rescaling ``x_hat * weight + bias``.

    Multi-dim normalization is applied sequentially: each dimension is reduced
    in turn. Because ``mean`` and ``variance`` preserve the number of dimensions
    (``keepdims=True``), the intermediate results broadcast back to ``x``'s shape
    without any explicit reshaping.

    Args:
        x: Input tensor of any shape.
        dims: List of axes to normalize over.
        weight: Optional learnable scale (γ). Shape must broadcast to ``x.shape``.
            Typically ``ones`` initialized. If ``None``, no scaling is applied.
        bias: Optional learnable shift (β). Shape must broadcast to ``x.shape``.
            Typically ``zeros`` initialized. If ``None``, no shift is applied.
        eps: Small constant added to the variance before taking the square root,
            preventing division by zero. Default ``1e-5``.
        correction: Degrees-of-freedom correction passed to ``variance``. ``0``
            (default) uses population variance, which is standard for norm layers.
            ``1`` uses Bessel's correction (unbiased sample variance).

    Returns:
        Normalized tensor of the same shape as ``x``.

    Example:
        >>> x = Tensor(np.random.randn(2, 4, 8).astype(np.float32))
        >>> y = norm(x, dims=[-1])               # layer norm over last dim
        >>> y = norm(x, dims=[-2, -1])           # instance norm over spatial dims
    """
    mu = x
    for d in dims:
        mu = mean(mu, dim=d)

    var = x
    for d in dims:
        var = variance(var, dim=d, correction=correction)

    x_hat = (x - mu) / (var + eps) ** 0.5
    if weight is not None:
        x_hat = x_hat * weight
    if bias is not None:
        x_hat = x_hat + bias
    return x_hat
