import numpy as np
from simplegrad.core.ops import sum, log
from simplegrad.core.tensor import Tensor
from .function import Function


# Cross Entropy Loss
class CEL(Function):
    def apply(self, p, y):
        # p - obtained probabilities
        # y - target probabilities
        return -sum(y * log(p))

    def __repr__(self):
        return f"CEL()"


# Cross Entropy Loss after SoftMax
class CEL_after_Softmax(Function):
    def apply(self, z, y, dim=None):
        # z - layer output
        # s - softmax intermediate result
        # y - target probability distribution

        exps = np.exp(z.values)
        exps_sum = np.sum(exps, axis=dim, keepdims=True)
        s = exps / exps_sum  # softmax outputs
        -np.sum(y.values * np.log(s))
        out = Tensor(-np.sum(y.values * np.log(s)))
        out.prev = {z}
        out.oper = "CLE(softmax)"

        def backward_step():
            z.grad += s - y.values

        out.backward_step = backward_step
        return out

    def __repr__(self):
        return f"CEL(Softmax())"
