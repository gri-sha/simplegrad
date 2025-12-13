import numpy as np

class Optimizer:
    def __init__(self):
        self.step_count = 0

    def zero_grad(self):
        for _, param in self.model.parameters().items():
            param.grad = np.zeros_like(param.values)

    def step(self):
        raise NotImplementedError("step() method is not implemented.")
    
    def reset_step_count(self):
        self.step_count = 0