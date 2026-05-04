"""Tests for tensor transform operations: flatten, reshape."""

import numpy as np
import pytest

pytestmark = pytest.mark.usefixtures("device")
import simplegrad as sg
from .utils import gradcheck, fwdcheck


def test_flatten():
    array1 = np.random.randn(2, 3, 4, 5).astype(np.float64)
    a = sg.Tensor(array1, dtype="float64")

    def fn():
        return sg.flatten(a, start_dim=1, end_dim=-1) * 2 + 1

    gradcheck(fn, [a])


def test_flatten_shapes_and_values():
    x = np.arange(24, dtype=np.float64).reshape(2, 3, 4)

    # flatten middle two dims
    a = sg.Tensor(x.copy(), dtype="float64")
    out = sg.flatten(a, start_dim=1, end_dim=2)
    assert out.shape == (2, 12), f"expected (2,12) got {out.shape}"
    fwdcheck(out, x.reshape(2, 12))

    # flatten first two dims
    b = sg.Tensor(x.copy(), dtype="float64")
    out2 = sg.flatten(b, start_dim=0, end_dim=1)
    assert out2.shape == (6, 4), f"expected (6,4) got {out2.shape}"
    fwdcheck(out2, x.reshape(6, 4))

    # flatten all dims
    c = sg.Tensor(x.copy(), dtype="float64")
    out3 = sg.flatten(c, start_dim=0, end_dim=2)
    assert out3.shape == (24,), f"expected (24,) got {out3.shape}"
    fwdcheck(out3, x.reshape(24))
