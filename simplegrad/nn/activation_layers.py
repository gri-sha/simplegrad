from simplegrad.core import Tensor
from .module import Module
from simplegrad.functions.activations import *


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return relu(x)

    def __str__(self):
        return "ReLU"


class Softmax(Module):
    def __init__(self, dim: int | None =None):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor):
        return softmax(x, self.dim)

    def __str__(self):
        return f"Softmax(dim={self.dim})"


class Tanh(Module):
    def forward(self, x: Tensor):
        return tanh(x)

    def __str__(self):
        return "Tanh"


class Sigmoid(Module):
    def forward(self, x: Tensor):
        return sigmoid(x)

    def __str__(self):
        return "Sigmoid"
