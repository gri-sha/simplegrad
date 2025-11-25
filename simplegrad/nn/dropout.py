from simplegrad.core.tensor import Tensor
from ..core.module import Module
import numpy as np

class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        if not 0 <= p < 1:
            raise ValueError("Dropout probability must be in the range [0, 1).")
        self.p = p
        self.mask = None

    def forward(self, x):
        if not isinstance(x, Tensor):
            raise TypeError("Input must be a Tensor.")

        if self.p == 0:
            return x  # No dropout applied

        # Create dropout mask
        self.mask = Tensor(np.random.rand(*x.values.shape) >= self.p, comp_grad=False)
        out = Tensor(x.values * self.mask.values)
        out.prev = {x}
        out.comp_grad = x.comp_grad
        out.is_leaf = False
        out.oper = f"Dropout(p={self.p})"

        # Backward step
        def backward_step():
            if x.comp_grad and self.mask is not None:
                x._init_grad_if_needed()
                x.grad += out.grad * self.mask.values

        out.backward_step = backward_step
        return out
    
    def __str__(self):
        return f"Dropout(p={self.p})"