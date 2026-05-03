"""Generalized normalization layer."""

from ..core import Tensor, Module, ones, zeros
from ..functions.norm import norm


class Norm(Module):
    """Generalized normalization layer with optional learnable affine parameters.

    Unifies batch norm, layer norm, instance norm, and similar variants by
    accepting an explicit list of dimensions to normalize over. The normalization
    formula is::

        x_hat = (x - mean(x, dims)) / sqrt(var(x, dims) + eps)
        output = x_hat * weight + bias   # if elementwise_affine=True

    Choosing ``dims`` selects the normalization variant:

    - Layer norm: ``dims=[-1]`` or the last N feature dims
    - Batch norm (approximate): ``dims=[0]``
    - Instance norm: ``dims=[-2, -1]`` (spatial dims per channel)

    Args:
        normalized_shape: Size(s) of the normalized dimensions. Determines the
            shape of the learnable ``weight`` and ``bias`` tensors. Pass an int
            for a single dim or a list for multiple dims.
        dims: List of axes to normalize over (passed directly to :func:`norm`).
        eps: Small constant added to variance before sqrt. Default ``1e-5``.
        correction: Degrees-of-freedom correction for variance. ``0`` (default)
            uses population variance (standard for norm layers). ``1`` uses
            Bessel's correction.
        elementwise_affine: If ``True`` (default), adds learnable per-element
            scale (``weight``, initialized to ones) and shift (``bias``,
            initialized to zeros). If ``False``, the output is pure normalized
            values with no learnable parameters.
        dtype: Data type for ``weight`` and ``bias``. Default ``"float32"``.

    Example:
        >>> layer_norm = Norm(normalized_shape=64, dims=[-1])
        >>> instance_norm = Norm(normalized_shape=[8, 8], dims=[-2, -1])
        >>> x = Tensor(np.random.randn(4, 16, 64).astype(np.float32))
        >>> out = layer_norm(x)   # shape (4, 16, 64)
    """

    def __init__(
        self,
        normalized_shape: int | list[int],
        dims: list[int],
        eps: float = 1e-5,
        correction: int = 0,
        elementwise_affine: bool = True,
        dtype: str = "float32",
    ) -> None:
        super().__init__()
        self.dims = dims
        self.eps = eps
        self.correction = correction
        self.elementwise_affine = elementwise_affine
        self.dtype = dtype

        shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        if elementwise_affine:
            self.weight = ones(shape, dtype=dtype, label="gamma")
            self.bias = zeros(shape, dtype=dtype, label="beta")

    def forward(self, x: Tensor) -> Tensor:
        """Normalize ``x`` over ``self.dims`` and apply affine transform.

        Args:
            x: Input tensor of any shape.

        Returns:
            Normalized tensor of the same shape as ``x``.
        """
        return norm(
            x,
            dims=self.dims,
            weight=self.weight if self.elementwise_affine else None,
            bias=self.bias if self.elementwise_affine else None,
            eps=self.eps,
            correction=self.correction,
        )

    def __str__(self) -> str:
        return (
            f"{self.label}(dims={self.dims}, eps={self.eps}, "
            f"affine={self.elementwise_affine})"
        )
