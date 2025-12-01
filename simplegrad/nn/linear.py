import numpy as np
from simplegrad.core import Tensor, Module
from simplegrad.dtypes import convert_to_dtype
from typing import Optional


class Linear(Module):
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
            if dtype is None:
                dtype = weight.dtype
                self.weight = weight
            else:
                self.weight = weight.convert_to_dtype(dtype, inplace=False)
            self.weight.label = weight_label 
            self.in_features = weight.shape[0]
            self.out_features = weight.shape[1]
        else:
            assert in_features > 0, "in_features must be a positive integer"
            self.in_features = in_features
            assert out_features > 0, "out_features must be a positive integer"
            self.out_features = out_features

            self.weight = self._init_param(
                shape=(self.in_features, self.out_features),
                dtype=self.dtype,
                label=bias_label,
                k=1 / self.in_features,
            )

        if self.use_bias:
            if bias is not None:
                assert bias.shape == (1, self.out_features), (
                    "Bias shape must be (1, out_features), "
                    f"but got {bias.shape} instead."
                )
                self.bias = bias
                self.bias.label = bias_label
            else:
                self.bias = self._init_param(
                    shape=(1, self.out_features),
                    dtype=self.dtype,
                    label=weight_label,
                    k=1 / self.in_features,
                )

    def forward(self, x: Tensor) -> Tensor:
        res = x @ self.weight
        if self.use_bias:
            res = res + self.bias
        return res

    def __str__(self) -> str:
        return f"{self.label}(in={self.in_features}, out={self.out_features}, bias={self.use_bias})"
