from simplegrad.core import Tensor, normal, _should_compute_grad
from typing import Optional
from .module import Module


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, weight: Optional[Tensor] = None, dtype: Optional[str] = None):
        super().__init__()
        self.dtype = dtype if dtype is not None else "float32"

        if weight is not None:
            assert weight.ndim == 2, "Weight tensor must be 2-dimensional"
            assert isinstance(weight, Tensor), "Weight must be a sg.Tensor"
            self.weight = weight.convert_to_dtype(self.dtype, inplace=False)
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
        # assert input.dtype.startswith("int"), "Input tensor must be of integer type"

        flat_input = input.values.flatten()
        embedded = self.weight.values[flat_input]
        output_shape = input.shape + (self.embedding_dim,)
        output_values = embedded.reshape(output_shape)

        out = Tensor(values=output_values, dtype=self.dtype)
        out.prev = {self.weight}
        out.oper = f"Embed(n_embd={self.num_embeddings}, embd_dim={self.embedding_dim})"
        out.comp_grad = _should_compute_grad(self.weight)

        if out.comp_grad:

            def backward_step():
                if self.weight.comp_grad:
                    self.weight._init_grad_if_needed()
                    grad_values = out.grad.reshape(-1, self.embedding_dim)
                    for idx, grad in zip(flat_input, grad_values):
                        self.weight.grad[idx] += grad

            out.backward_step = backward_step

        return out

    def __repr__(self):
        return f"Embedding(n_embd={self.num_embeddings}, embd_dim={self.embedding_dim}, dtype={self.dtype})"
