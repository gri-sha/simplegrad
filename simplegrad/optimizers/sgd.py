from .optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, lr=0.01, momentum=0.0):
        self.lr = lr
        self.momentum = momentum
        self.velocities = {}

    def step(self, loss):
        for var in loss._prev:
            if var.requires_grad:
                var.data -= var.grad * self.lr