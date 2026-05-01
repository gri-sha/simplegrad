"""Tests for core tensor arithmetic operations and autograd."""

import numpy as np
import pytest

pytestmark = pytest.mark.usefixtures("device")
import simplegrad as sg
from .utils import gradcheck


def test_tensor_methods():
    array1 = np.array([[32, 1, -23], [1, 0.1, -1.55], [-4.34, 1, 23]])
    array2 = np.array([[3, 1, 0.1], [1, -33.1, -1.5], [8.78, 2, 23]])
    a = sg.Tensor(array1, dtype="float64")
    b = sg.Tensor(array2, dtype="float64")

    gradcheck(lambda: (1.55 * ((a + 2) ** 2)) / (30.3 / (7.7 - b)) * a, [a, b])


def test_matmul():
    array1 = np.array([[1.5, 2.0, -0.5], [0.5, 1.2, 0.3]])
    array2 = np.array([[2.0, -1.0], [0.5, 1.5], [1.2, -0.8]])
    a = sg.Tensor(array1, dtype="float64")
    b = sg.Tensor(array2, dtype="float64")

    gradcheck(lambda: a @ b @ b.T, [a, b])


def test_broadcasting_1():
    array1 = np.random.randn(3, 4, 5).astype(np.float64)
    array2 = np.random.randn(1, 4, 1).astype(np.float64)
    array3 = np.random.randn(3, 1, 5).astype(np.float64)
    a = sg.Tensor(array1, dtype="float64")
    b = sg.Tensor(array2, dtype="float64")
    c = sg.Tensor(array3, dtype="float64")

    gradcheck(lambda: (a + b) * c + a * (b + c), [a, b, c])


def test_broadcasting_2():
    array1 = np.random.randn(2, 3, 4).astype(np.float64)
    array2 = np.random.randn(4, 5).astype(np.float64)
    array3 = np.random.randn(1, 1, 5).astype(np.float64)
    array4 = np.random.randn(2, 3, 1).astype(np.float64)
    a = sg.Tensor(array1, dtype="float64")
    b = sg.Tensor(array2, dtype="float64")
    c = sg.Tensor(array3, dtype="float64")
    d = sg.Tensor(array4, dtype="float64")

    def fn():
        mm = a @ b
        return (mm + c) * d + mm

    gradcheck(fn, [a, b, c, d])
