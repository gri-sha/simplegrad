"""Token embedding layer."""

import numpy as np
from ..core import Tensor, Function, Context, Module, normal

class _EmbeddingLookup(Function):
    @staticmethod
    def output_shape(
        weight: Tensor, input_tensor: Tensor, output_shape: tuple, embedding_dim: int
    ) -> tuple:
        return output_shape

    @staticmethod
    def forward(
        ctx: Context, weight: Tensor, input_tensor: Tensor, output_shape: tuple, embedding_dim: int
    ) -> np.ndarray:
        ctx.flat_input = input_tensor.values.flatten()
        ctx.embedding_dim = embedding_dim
        ctx.weight_shape = weight.shape
        return weight.values[ctx.flat_input].reshape(output_shape)

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> tuple:
        weight_grad = np.zeros(ctx.weight_shape)
        grad_values = grad_output.reshape(-1, ctx.embedding_dim)
        np.add.at(weight_grad, ctx.flat_input, grad_values)
        return weight_grad, None


class Embedding(Module):
    """Lookup table that maps integer indices to dense vectors.

    Weights are initialized from N(0, 1) by default.

    Args:
        num_embeddings: Size of the vocabulary (number of rows in the embedding table).
        embedding_dim: Dimensionality of each embedding vector.
        weight: Optional pre-built embedding matrix of shape
            ``(num_embeddings, embedding_dim)``.
        dtype: Data type string. Defaults to ``"float32"``.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        weight: Tensor | None = None,
        dtype: str | None = None,
    ):
        super().__init__()
        self.dtype = dtype if dtype is not None else "float32"

        if weight is not None:
            assert (
                weight.shape[0] > 0 and len(weight.shape) == 2
            ), "Weight tensor must be 2-dimensional"
            assert isinstance(weight, Tensor), "Weight must be a sg.Tensor"
            self.weight = weight.convert_to(self.dtype, inplace=False)
            self.dtype = weight.dtype
            self.num_embeddings = weight.shape[0]
            self.embedding_dim = weight.shape[1]
        else:
            self.num_embeddings = num_embeddings
            self.embedding_dim = embedding_dim
            self.weight = normal(
                shape=(num_embeddings, embedding_dim),
                dtype=self.dtype,
                label="Embedding_weight",
                mu=0,
                sigma=1,
            )

    def forward(self, input: Tensor) -> Tensor:
        """Look up embeddings for the given indices.

        Args:
            input: Integer tensor of indices with any shape ``S``.

        Returns:
            Embedding tensor of shape ``(*S, embedding_dim)``.
        """
        output_shape = input.shape + (self.embedding_dim,)
        out = _EmbeddingLookup.apply(
            self.weight,
            input,
            output_shape,
            self.embedding_dim,
            oper=f"Embed(n_embd={self.num_embeddings}, embd_dim={self.embedding_dim})",
        )
        return out

    def __repr__(self):
        return f"Embedding(n_embd={self.num_embeddings}, embd_dim={self.embedding_dim}, dtype={self.dtype})"
