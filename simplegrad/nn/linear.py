"""Fully-connected (linear/dense) layer."""

import numpy as np
from simplegrad.core import Tensor, uniform
from simplegrad.dtypes import convert_to_dtype
from .module import Module
from typing import Optional


class Linear(Module):
    """Fully-connected linear layer: ``output = x @ W + b``.

    Weights are initialized with Kaiming uniform (range ``[-1/sqrt(in), 1/sqrt(in)]``).

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        weight: Optional pre-built weight tensor of shape ``(in_features, out_features)``.
        bias: Optional pre-built bias tensor of shape ``(out_features,)``.
        use_bias: Add a bias term. Defaults to True.
        dtype: Data type string. Defaults to ``"float32"``.
        weight_label: Label for the weight tensor (used in graph visualization).
        bias_label: Label for the bias tensor.
    """

    def __init__(
        self,
        in_features: Optional[int] = None,
        out_features: Optional[int] = None,
        weight: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        use_bias: bool = True,
        dtype: str = None,  # use global dtype if None
        weight_label: str = "W",
        bias_label: str = "b",
    ) -> None:
        super().__init__()
        self.dtype = dtype if dtype is not None else "float32"
        self.use_bias = use_bias

        if weight is not None:
            self.weight = weight.convert_to(dtype, inplace=False)
            self.weight.label = weight_label
            self.in_features = weight.shape[0]
            self.out_features = weight.shape[1]
        else:
            assert in_features > 0, "in_features must be a positive integer"
            self.in_features = in_features
            assert out_features > 0, "out_features must be a positive integer"
            self.out_features = out_features

            self.weight = uniform(
                shape=(self.in_features, self.out_features),
                dtype=self.dtype,
                label=weight_label,
                high=np.sqrt(1 / self.in_features),
                low=-np.sqrt(1 / self.in_features),
            )

        if self.use_bias:
            if bias is not None:
                assert bias.shape == (self.out_features,), "Bias shape must be (out_features,), " f"but got {bias.shape} instead."
                self.bias = bias
                self.bias.label = bias_label
            else:
                self.bias = uniform(
                    shape=(self.out_features,),
                    dtype=self.dtype,
                    label=bias_label,
                    high=np.sqrt(1 / self.in_features),
                    low=-np.sqrt(1 / self.in_features),
                )

    def forward(self, x: Tensor) -> Tensor:
        """Compute ``x @ W + b``.

        Args:
            x: Input tensor of shape ``(..., in_features)``.

        Returns:
            Output tensor of shape ``(..., out_features)``.
        """
        res = x @ self.weight
        if self.use_bias:
            res = res + self.bias
        return res

    def __str__(self) -> str:
        return f"{self.label}(in={self.in_features}, out={self.out_features}, bias={self.use_bias})"
