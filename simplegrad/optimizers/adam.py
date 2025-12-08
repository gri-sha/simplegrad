import numpy as np
from .optimizer import Optimizer

class Adam(Optimizer):
    def __init__(self, model, lr, beta_1=0.9, beta_2=0.999, eps=1e-8):
        super().__init__()
        self.model = model
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.moments1 = {name: np.zeros_like(param.values) for name, param in self.model.parameters().items()}
        self.moments2 = {name: np.zeros_like(param.values) for name, param in self.model.parameters().items()}
        self.t = 0

    def step(self):
        self.t += 1
        for name, param in self.model.parameters().items():
            if param.grad is None:
                raise ValueError(f"Gradient for {name} is None. Did you forget to call backward()?")

            # Update biased first moment estimate
            self.moments1[name] = self.beta_1 * self.moments1[name] + (1 - self.beta_1) * param.grad

            # Update biased second raw moment estimate
            self.moments2[name] = self.beta_2 * self.moments2[name] + (1 - self.beta_2) * (param.grad ** 2)

            # Compute bias-corrected first moment estimate
            m_hat = self.moments1[name] / (1 - self.beta_1 ** self.t)

            # Compute bias-corrected second raw moment estimate
            v_hat = self.moments2[name] / (1 - self.beta_2 ** self.t)

            # Update parameters
            param.values -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)