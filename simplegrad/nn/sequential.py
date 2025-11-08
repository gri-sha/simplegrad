from simplegrad.core.tensor import Tensor
from ..core.module import Module

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x):
        for module in self.modules:
            x = module(x)
        return x