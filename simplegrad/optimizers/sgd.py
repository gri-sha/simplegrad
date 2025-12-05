from .optimizer import Optimizer
from simplegrad.core.tensor import Tensor
from simplegrad.nn import Module
import numpy as np


class SGD(Optimizer):
    def __init__(self, model, lr=0.01, momentum=0, dampening=0):
        if not isinstance(model, Module):
            raise TypeError(f"model must be a Module")

        self.model = model
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.velocities = {name: np.zeros_like(param.values) for name, param in model.parameters().items()}

    def step(self):
        for name, param in self.model.parameters().items():
            if param.grad is None:
                raise ValueError(f"Gradient for {name} is None. Did you forget to call backward()?")

            self.velocities[name] = self.momentum * self.velocities[name] - self.lr * (1 - self.dampening) * param.grad
            param.values += self.velocities[name]
