"""Tests for activation functions: relu, tanh, softmax."""

import numpy as np
import simplegrad as sg
import torch
from .utils import compare2tensors


def _activations_helper(arrays, dims):
    for array, dim in zip(arrays, dims):
        a = sg.Tensor(array, dtype="float64")
        softmax_out = sg.softmax(a, dim=dim)
        tanh_out = sg.tanh(softmax_out)
        c = sg.relu(tanh_out)
        c.zero_grad()
        c.backward()

        at = torch.from_numpy(array).to(torch.float64).requires_grad_(True)
        softmax_out_t = torch.softmax(at, dim=dim)
        tanh_out_t = torch.tanh(softmax_out_t)
        ct = torch.relu(tanh_out_t)
        loss = ct.sum()
        loss.backward()

        compare2tensors(sg=c, pt=ct)
        compare2tensors(sg=a.grad, pt=at.grad)


# softmax, relu, tanh across various shapes
def test_activations():
    array1 = np.array([[1.2, -0.5, 2.1], [-1.0, 0.3, 1.5], [0.8, -2.0, 0.5]])
    array2 = np.array([1.2, -0.5, 2.1, -1.0, 0.3, 1.5, 0.8, -2.0, 0.5])
    array3 = np.array([[-1.0, 2.0], [3.0, -0.5], [0.0, 1.0], [2.5, -2.5]])
    _activations_helper([array1, array2, array3], [1, -1, 0])
