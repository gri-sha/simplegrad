import numpy as np
from simplegrad.core.tensor import Tensor
from simplegrad.core.ops import exp, log, sum
from .function import Function


class ReLU(Function):
    def apply(self, tensor):
        out = Tensor(np.maximum(0, tensor.values))
        out.prev = {tensor}
        out.oper = "ReLU"

        def backward_step():
            if tensor.comp_grad:
                tensor.grad = out.grad * np.where(tensor.values > 0, 1.0, 0.0)

        out.backward_step = backward_step
        return out

    def __repr__(self):
        return "ReLU()"


class Softmax(Function):
    def apply(self, tensor, dim=None):
        exps = exp(tensor)
        return exps / sum(exps, dim)

    def __repr__(self):
        return "Softmax()"


class Tanh(Function):
    def apply(self, tensor):
        exps = exp(-2 * tensor)
        return (1 - exps) / (1 + exps)

    def __repr__(self):
        return "Tanh()"
