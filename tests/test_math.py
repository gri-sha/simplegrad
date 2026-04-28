"""Tests for math functions: exp, log, sin, cos, tan, sum, mean, trace."""

import numpy as np
import simplegrad as sg
from .utils import gradcheck


def test_trig_and_exp():
    array1 = np.array([[0.5, 1.2, -0.3], [0.1, -0.5, 2.1]])
    array2 = np.array([[1.0, 0.5, 0.8], [0.3, 1.5, 0.2]])
    a = sg.Tensor(array1, dtype="float64")
    b = sg.Tensor(array2, dtype="float64")

    gradcheck(
        lambda: sg.exp(a) * sg.log(b) + sg.sin(a * b) + sg.cos(a + 1) - sg.tan(a - b),
        [a, b],
    )


def test_reductions():
    array1 = np.array([[2.0, 3.0, 1.5], [0.5, 2.5, 1.0], [1.2, 0.8, 2.2]])
    a = sg.Tensor(array1, dtype="float64")

    gradcheck(lambda: sg.sum(a, dim=1) + sg.mean(a, dim=0) + sg.trace(a), [a])
