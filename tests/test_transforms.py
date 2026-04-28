"""Tests for tensor transform operations: flatten, reshape."""

import numpy as np
import simplegrad as sg
from .utils import gradcheck


def test_flatten():
    array1 = np.random.randn(2, 3, 4, 5).astype(np.float64)
    a = sg.Tensor(array1, dtype="float64")

    def fn():
        return sg.flatten(a, start_dim=1, end_dim=-1) * 2 + 1

    gradcheck(fn, [a])
