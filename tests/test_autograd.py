"""Tests for core tensor arithmetic operations and autograd."""

import numpy as np
import simplegrad as sg
import torch
from .utils import compare2tensors


# +, -, *, /, **
def test_tensor_methods():
    array1 = np.array([[32, 1, -23], [1, 0.1, -1.55], [-4.34, 1, 23]])
    array2 = np.array([[3, 1, 0], [1, -33.1, -1.5], [8.78, 2, 23]])
    a, b = sg.Tensor(array1, dtype="float64"), sg.Tensor(array2, dtype="float64")
    c = (1.55 * ((a + 2) ** 2)) / (30.3 / (7.7 - b)) * a
    c.zero_grad()
    c.backward()
    at = torch.from_numpy(array1).to(torch.float64).requires_grad_(True)
    bt = torch.from_numpy(array2).to(torch.float64).requires_grad_(True)
    ct = (1.55 * ((at + 2) ** 2)) / (30.3 / (7.7 - bt)) * at
    loss = ct.sum()
    loss.backward()
    compare2tensors(sg=c, pt=ct)
    compare2tensors(sg=a.grad, pt=at.grad)
    compare2tensors(sg=b.grad, pt=bt.grad)


# matmul
def test_matmul():
    array1 = np.array([[1.5, 2.0, -0.5], [0.5, 1.2, 0.3]])
    array2 = np.array([[2.0, -1.0], [0.5, 1.5], [1.2, -0.8]])
    a, b = sg.Tensor(array1, dtype="float64"), sg.Tensor(array2, dtype="float64")

    c = a @ b @ b.T
    c.zero_grad()
    c.backward()

    at = torch.from_numpy(array1).to(torch.float64).requires_grad_(True)
    bt = torch.from_numpy(array2).to(torch.float64).requires_grad_(True)
    ct = at @ bt @ bt.T
    loss = ct.sum()
    loss.backward()

    compare2tensors(sg=c, pt=ct)
    compare2tensors(sg=a.grad, pt=at.grad)
    compare2tensors(sg=b.grad, pt=bt.grad)


# broadcasting: +, *, @
def test_broadcasting_1():
    array1 = np.random.randn(3, 4, 5).astype(np.float64)
    array2 = np.random.randn(1, 4, 1).astype(np.float64)
    array3 = np.random.randn(3, 1, 5).astype(np.float64)

    a = sg.Tensor(array1, dtype="float64")
    b = sg.Tensor(array2, dtype="float64")
    c = sg.Tensor(array3, dtype="float64")

    result = (a + b) * c + a * (b + c)
    result.zero_grad()
    result.backward()

    at = torch.from_numpy(array1).to(torch.float64).requires_grad_(True)
    bt = torch.from_numpy(array2).to(torch.float64).requires_grad_(True)
    ct = torch.from_numpy(array3).to(torch.float64).requires_grad_(True)

    result_t = (at + bt) * ct + at * (bt + ct)
    loss = result_t.sum()
    loss.backward()

    compare2tensors(sg=result, pt=result_t)
    compare2tensors(sg=a.grad, pt=at.grad)
    compare2tensors(sg=b.grad, pt=bt.grad)
    compare2tensors(sg=c.grad, pt=ct.grad)


# broadcasting with matmul
def test_broadcasting_2():
    array1 = np.random.randn(2, 3, 4).astype(np.float64)
    array2 = np.random.randn(4, 5).astype(np.float64)
    array3 = np.random.randn(1, 1, 5).astype(np.float64)
    array4 = np.random.randn(2, 3, 1).astype(np.float64)

    a = sg.Tensor(array1, dtype="float64")
    b = sg.Tensor(array2, dtype="float64")
    c = sg.Tensor(array3, dtype="float64")
    d = sg.Tensor(array4, dtype="float64")

    matmul_result = a @ b
    result = (matmul_result + c) * d + matmul_result
    result.zero_grad()
    result.backward()

    at = torch.from_numpy(array1).to(torch.float64).requires_grad_(True)
    bt = torch.from_numpy(array2).to(torch.float64).requires_grad_(True)
    ct = torch.from_numpy(array3).to(torch.float64).requires_grad_(True)
    dt = torch.from_numpy(array4).to(torch.float64).requires_grad_(True)

    matmul_result_t = at @ bt
    result_t = (matmul_result_t + ct) * dt + matmul_result_t
    loss = result_t.sum()
    loss.backward()

    compare2tensors(sg=result, pt=result_t)
    compare2tensors(sg=a.grad, pt=at.grad)
    compare2tensors(sg=b.grad, pt=bt.grad)
    compare2tensors(sg=c.grad, pt=ct.grad)
    compare2tensors(sg=d.grad, pt=dt.grad)
