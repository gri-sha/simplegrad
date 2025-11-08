import numpy as np
from simplegrad.core.tensor import Tensor
from ..core.module import Module
from simplegrad.dtypes import convert_to_dtype 


class Linear(Module):
    """
    A fully connected linear layer that applies a linear transformation to the incoming data:
    y = (W @ x) + b, where x is a column vector of input features.
    For compatibility

    By default, weights are initialized with random values from standard distibution (numpy.random.standard_normal).
    You can vary them using init_multiplier parameter.
    """

    def __init__(
        self,
        in_features,
        out_features,
        weight=None,
        bias=None,
        use_bias=True,
        init_multiplier=0.01,  # random init by default with this multiplier
        init_dtype=None,  # use global dtype if None
        zeros_init=False,
        ones_init=False,
        weight_label="W",
        bias_label="b",
        label="Linear",
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.label = label

        assert (
            sum([zeros_init, ones_init]) <= 1
        ), "Choose only one special method: zeros_init or ones_init must be True."

        self.weight = (
            weight
            if weight is not None
            else self._init_weights(
                init_multiplier, init_dtype, zeros_init, ones_init, weight_label
            )
        )

        if self.use_bias:
            self.bias = (
                bias
                if bias is not None
                else self._init_bias(init_multiplier, init_dtype, zeros_init, ones_init, bias_label)
            )

    def _init_weights(self, multiplier, dtype, zeros_init, ones_init, weight_label):
        if zeros_init:
            data = np.zeros((self.in_features, self.out_features))
        elif ones_init:
            data = np.ones((self.in_features, self.out_features))
        else:
            data = np.random.randn(self.in_features, self.out_features) * multiplier

        return Tensor(convert_to_dtype(array=data, dtype=dtype), label=weight_label)

    def _init_bias(self, multiplier, dtype, zeros_init, ones_init, bias_label):
        if zeros_init:
            data = np.zeros((1, self.out_features))
        elif ones_init:
            data = np.ones((1, self.out_features))
        else:
            data = np.random.randn(1, self.out_features) * multiplier
        return Tensor(convert_to_dtype(array=data, dtype=dtype), label=bias_label)

    def forward(self, x):
        # print("Weights:", self.weight.shape, "Inputs:", input.shape)
        res = x @ self.weight
        # print("After matmul:", res.shape)
        if self.use_bias:
            # print("Bias:", self.bias.shape)
            res = res + self.bias
        return res
