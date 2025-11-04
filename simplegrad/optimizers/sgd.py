from .optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, lr=0.01, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay= weight_decay
        self.nesterov = nesterov

        # dont forget about velocities
        # reminder: each trainable parameter of the model has its own velocity

    def step(self, loss):
        for var in loss._prev:
            if var.requires_grad:
                var.data -= var.grad * self.lr