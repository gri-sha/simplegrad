import numpy as np
from .optimizer import Optimizer

class Adam(Optimizer):
    def __init(self, beta_1, beta_2, eps=1e-8):
        super().__init__()
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.moments1 = {name: np.zeros_like(param.values) for name, param in self.model.parameters().items()}
        self.moments2 = {name: np.zeros_like(param.values) for name, param in self.model.parameters().items()}
        self.t = 0