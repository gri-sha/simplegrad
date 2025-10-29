import numpy as np
from .tensor import Tensor

def zeros(shape, comp_grad=True, label=None):
    return Tensor(np.zeros(shape), comp_grad, label)

def ones(shape, comp_grad=True, label=None):
    return Tensor(np.ones(shape), comp_grad, label)

def random(shape, comp_grad=True, label=None):
    return Tensor(np.random.standard_normal(shape), comp_grad, label)