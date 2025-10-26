import numpy as np
from simplegrad.core.tensor import Tensor
from .module import Module


class Linear(Module):
    def __init__(
        self,
        input_features,
        output_features,
        weights=None,
        biases=None,
        rand_init=True,
        init_multiplier=0.01,
        zeros_init=False,
        ones_init=False,
    ):
        self.input_features = input_features
        self.output_features = output_features
        self.init_multiplier = init_multiplier
        self.weights_shape = self.input_shape + self.output_shape
        self.biases_shape = self.output_shape
        self.weights = weights
        self.biases = biases
        self.activation = activation

        if self.weights is not None:
            assert (
                self.weights.shape == self.weights_shape
            ), f"Weights shape mismatch: expected {self.weights_shape}, got {self.weights.shape}"

        if self.biases is not None:
            assert (
                self.biases.shape == self.biases_shape
            ), f"Biases shape mismatch: expected {self.biases_shape}, got {self.biases.shape}"

    def _initialize(self):
        if self.weights is None:
            self.weights = Tensor(
                np.random.randn(*self.weights_shape) * self.init_multiplier,
                comp_grad=True,
                label="W",
            )
        if self.biases is None:
            self.biases = Tensor(
                np.random.randn(*self.biases_shape) * self.init_multiplier,
                comp_grad=True,
                label="b",
            )

    def _initialize_with_zeros(self):
        if self.weights is None:
            self.weights = Tensor(
                np.random.zeros(*self.weights_shape) * self.init_multiplier,
                comp_grad=True,
                label="W",
            )
        if self.biases is None:
            self.biases = Tensor(
                np.random.zeros(*self.biases_shape) * self.init_multiplier,
                comp_grad=True,
                label="b",
            )

    def _initialize_with_ones(self):
        if self.weights is None:
            self.weights = Tensor(
                np.random.ones(*self.weights_shape) * self.init_multiplier,
                comp_grad=True,
                label="W",
            )
        if self.biases is None:
            self.biases = Tensor(
                np.random.zeros(*self.biases_shape) * self.init_multiplier,
                comp_grad=True,
                label="b",
            )

    def forward(self, input_batch):
        pass
