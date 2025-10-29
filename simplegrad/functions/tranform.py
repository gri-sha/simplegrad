import numpy as np
from simplegrad.core.tensor import Tensor
from simplegrad.core.ops import exp, log, sum
from .function import Function


class Flatten(Function):
    def apply(self, tensor):
        out = Tensor(tensor.values.flatten())
        out.prev = {tensor}
        out.oper = "Flatten"

        def backward_step():
            if tensor.comp_grad:
                tensor.grad = out.grad.reshape(tensor.values.shape)

        out.backward_step = backward_step
        return out

    def __repr__(self):
        return "Flatten()"
    
class Reshape(Function):
    def apply(self, tensor, shape):
        out = Tensor(tensor.values.reshape(shape))
        out.prev = {tensor}
        out.oper = "Reshape"

        def backward_step():
            if tensor.comp_grad:
                tensor.grad = out.grad.reshape(tensor.values.shape)

        out.backward_step = backward_step
        return out

    def __repr__(self):
        return "Reshape()"