from simplegrad.core import Tensor
from .module import Module
from simplegrad.core.activations import *


class ReLU(Module):
    def forward(self, x):
        return relu(x)

class Softmax(Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return softmax(x, self.dim)
    
class Tanh(Module):
    def forward(self, x):
        return tanh(x)