"""Tests for tensor transform operations: flatten, reshape."""

import numpy as np
import simplegrad as sg
import torch
from .utils import compare2tensors


def test_flatten():
    array1 = np.random.randn(2, 3, 4, 5).astype(np.float64)

    a = sg.Tensor(array1, dtype="float64")
    flattened = sg.flatten(a, start_dim=1, end_dim=-1)
    result = flattened * 2 + 1
    result.zero_grad()
    result.backward()

    at = torch.from_numpy(array1).to(torch.float64).requires_grad_(True)
    flattened_t = torch.flatten(at, start_dim=1, end_dim=-1)
    result_t = flattened_t * 2 + 1
    loss_t = result_t.sum()
    loss_t.backward()

    compare2tensors(sg=flattened, pt=flattened_t)
    compare2tensors(sg=result, pt=result_t)
    compare2tensors(sg=a.grad, pt=at.grad)
