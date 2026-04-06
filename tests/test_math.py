"""Tests for math functions: exp, log, sin, cos, tan, sum, mean, trace."""

import numpy as np
import simplegrad as sg
import torch
from .utils import compare2tensors


# exp, log, sin, cos, tan
def test_trig_and_exp():
    array1 = np.array([[0.5, 1.2, -0.3], [0.1, -0.5, 2.1]])
    array2 = np.array([[1.0, 0.5, 0.8], [0.3, 1.5, 0.2]])
    a, b = sg.Tensor(array1, dtype="float64"), sg.Tensor(array2, dtype="float64")

    c = sg.exp(a) * sg.log(b) + sg.sin(a * b) + sg.cos(a + 1) - sg.tan(a - b)
    c.zero_grad()
    c.backward()

    at = torch.from_numpy(array1).to(torch.float64).requires_grad_(True)
    bt = torch.from_numpy(array2).to(torch.float64).requires_grad_(True)
    ct = torch.exp(at) * torch.log(bt) + torch.sin(at * bt) + torch.cos(at + 1) - torch.tan(at - bt)
    loss = ct.sum()
    loss.backward()

    compare2tensors(sg=c, pt=ct)
    compare2tensors(sg=a.grad, pt=at.grad)
    compare2tensors(sg=b.grad, pt=bt.grad)


# sum, mean, trace
def test_reductions():
    array1 = np.array([[2.0, 3.0, 1.5], [0.5, 2.5, 1.0], [1.2, 0.8, 2.2]])
    a = sg.Tensor(array1, dtype="float64")
    c = sg.sum(a, dim=1) + sg.mean(a, dim=0) + sg.trace(a)
    c.zero_grad()
    c.backward()

    at = torch.from_numpy(array1).to(torch.float64).requires_grad_(True)
    ct = torch.sum(at, dim=1, keepdim=True) + torch.mean(at, dim=0, keepdim=True) + torch.trace(at)
    loss = ct.sum()
    loss.backward()

    compare2tensors(sg=c, pt=ct)
    compare2tensors(sg=a.grad, pt=at.grad)
