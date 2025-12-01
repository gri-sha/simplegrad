from ..core import Tensor, Module
from ..functions import flatten

class Flatten(Module):
    def __init__(self, start_dim=1, end_dim=-1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x):
        return flatten(x, self.start_dim, self.end_dim)

    def __str__(self):
        return f"Flatten(start_dim={self.start_dim}, end_dim={self.end_dim})"