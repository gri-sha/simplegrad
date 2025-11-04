import numpy as np
from .tensor import Tensor

def zeros(shape, req_grad=True, label=None):
    return Tensor(np.zeros(shape), req_grad, label)

def ones(shape, req_grad=True, label=None):
    return Tensor(np.ones(shape), req_grad, label)

def random(shape, req_grad=True, label=None):
    return Tensor(np.random.standard_normal(shape), req_grad, label)