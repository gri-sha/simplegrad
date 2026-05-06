"""Tests for math functions: exp, log, sin, cos, tan, sum, mean, trace."""

import numpy as np
import pytest

pytestmark = pytest.mark.usefixtures("device")
import simplegrad as sg
from .utils import gradcheck, fwdcheck, _to_numpy


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


def test_sum_keepdims_shape():
    x = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    a = sg.Tensor(x.copy(), dtype="float64")

    out = sg.sum(a, dim=1)
    assert out.shape == (2, 1, 4), f"expected (2,1,4) got {out.shape}"
    fwdcheck(out, np.sum(x, axis=1, keepdims=True))


def test_mean_negative_dim():
    x = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    a = sg.Tensor(x.copy(), dtype="float64")

    out = sg.mean(a, dim=-1)
    assert out.shape == (2, 3, 1), f"expected (2,3,1) got {out.shape}"
    fwdcheck(out, np.mean(x, axis=-1, keepdims=True))


def test_variance_bessel_correction():
    x = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    a = sg.Tensor(x.copy(), dtype="float64")

    fwdcheck(sg.variance(a, dim=1, correction=0), [[2.0 / 3.0]])
    fwdcheck(sg.variance(a, dim=1, correction=1), [[1.0]])


def test_variance_computational_identity():
    # var(x, correction=0) == mean(x^2) - mean(x)^2 for any input
    x = np.array([[1.0, 3.0, 5.0, 7.0], [2.0, 4.0, 6.0, 8.0]], dtype=np.float64)
    a = sg.Tensor(x.copy(), dtype="float64")

    var_out = sg.variance(a, dim=1, correction=0).values
    identity = np.mean(x**2, axis=1, keepdims=True) - np.mean(x, axis=1, keepdims=True) ** 2
    assert np.allclose(var_out, identity, atol=1e-10)


def test_trace_output_shape_and_value():
    x = np.arange(1, 10, dtype=np.float64).reshape(3, 3)
    a = sg.Tensor(x.copy(), dtype="float64")

    out = sg.trace(a)
    assert out.shape == (1, 1), f"expected (1,1) got {out.shape}"
    fwdcheck(out, [[15.0]])


def test_argmax_along_dim():
    x = np.array([[1.0, 3.0, 2.0], [5.0, 1.0, 4.0]], dtype=np.float64)
    a = sg.Tensor(x.copy(), dtype="float64")

    # row-wise: max at col 1 for row 0, col 0 for row 1
    out_row = sg.argmax(a, dim=1)
    assert np.array_equal(_to_numpy(out_row.values).flatten(), [1, 0])

    # col-wise: max at row 1 for all cols (5>1, 3>1, 4>2)
    out_col = sg.argmax(a, dim=0)
    assert np.array_equal(_to_numpy(out_col.values).flatten(), [1, 0, 1])


def test_argmin_along_dim():
    x = np.array([[1.0, 3.0, 2.0], [5.0, 1.0, 4.0]], dtype=np.float64)
    a = sg.Tensor(x.copy(), dtype="float64")

    # row-wise: min at col 0 for row 0, col 1 for row 1
    out_row = sg.argmin(a, dim=1)
    assert np.array_equal(_to_numpy(out_row.values).flatten(), [0, 1])

    # col-wise: min at row 0 for col 0 and col 2 (1<5, 2<4), row 1 for col 1 (1<3)
    out_col = sg.argmin(a, dim=0)
    assert np.array_equal(_to_numpy(out_col.values).flatten(), [0, 1, 0])
