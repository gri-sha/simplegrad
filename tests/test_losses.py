"""Tests for loss functions: mse_loss, ce_loss."""

import numpy as np
import pytest

pytestmark = pytest.mark.usefixtures("device")
import simplegrad as sg
from .utils import gradcheck, fwdcheck


def test_mse_forward_all_reductions():
    p = sg.Tensor(np.array([[1.0, 2.0]], dtype=np.float64), dtype="float64")
    y = sg.Tensor(np.zeros((1, 2), dtype=np.float64), dtype="float64")

    fwdcheck(sg.mse_loss(p, y, reduction="mean"), [[2.5]])
    fwdcheck(sg.mse_loss(p, y, reduction="sum"), [[5.0]])
    fwdcheck(sg.mse_loss(p, y, reduction=None), [[1.0, 4.0]])


@pytest.mark.parametrize("reduction", ["mean", "sum", None])
def test_mse_gradcheck(reduction):
    predictions = sg.Tensor(np.array([[0.2, 0.8, 0.5], [1.5, 0.3, 0.7]]), dtype="float64")
    targets = sg.Tensor(np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]), dtype="float64")
    gradcheck(lambda: sg.mse_loss(predictions, targets, reduction=reduction), [predictions])


def _ce_helper(array1, array2, dim=-1):
    for reduction in ["mean", "sum", None]:
        predictions = sg.Tensor(array1.copy(), dtype="float64")
        targets = sg.Tensor(array2.copy(), dtype="float64")

        def fn(pred=predictions, targ=targets, red=reduction):
            return sg.ce_loss(pred, targ, dim=dim, reduction=red)

        gradcheck(fn, [predictions])


def test_ce_uniform_logits_loss_equals_log_classes():
    # with equal logits and uniform target, ce = log(num_classes) — exact mathematical identity
    z = sg.Tensor(np.zeros((1, 3), dtype=np.float64), dtype="float64")
    y = sg.Tensor(np.full((1, 3), 1.0 / 3, dtype=np.float64), dtype="float64")
    fwdcheck(sg.ce_loss(z, y, dim=-1, reduction=None), [[np.log(3)]], atol=1e-10)


def test_ce_translation_invariance():
    # softmax is translation-invariant, so ce_loss(z, y) == ce_loss(z+c, y) for any c
    z_data = np.array([[2.0, 1.0, 0.5]], dtype=np.float64)
    y_data = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)

    loss = sg.ce_loss(
        sg.Tensor(z_data.copy(), dtype="float64"),
        sg.Tensor(y_data.copy(), dtype="float64"),
        dim=-1,
        reduction=None,
    ).values
    loss_shifted = sg.ce_loss(
        sg.Tensor(z_data + 100.0, dtype="float64"),
        sg.Tensor(y_data.copy(), dtype="float64"),
        dim=-1,
        reduction=None,
    ).values
    assert np.allclose(loss, loss_shifted, atol=1e-8)


def test_ce_positive_dim_matches_negative_dim():
    # dim=1 and dim=-2 must give identical outputs on shape (2, 4, 3)
    rng = np.random.default_rng(0)
    z_data = rng.standard_normal((2, 4, 3))
    y_data = np.zeros_like(z_data)
    for i in range(2):
        for k in range(3):
            y_data[i, rng.integers(0, 4), k] = 1.0

    out_pos = sg.ce_loss(
        sg.Tensor(z_data.copy(), dtype="float64"),
        sg.Tensor(y_data.copy(), dtype="float64"),
        dim=1,
        reduction=None,
    ).values
    out_neg = sg.ce_loss(
        sg.Tensor(z_data.copy(), dtype="float64"),
        sg.Tensor(y_data.copy(), dtype="float64"),
        dim=-2,
        reduction=None,
    ).values
    assert np.allclose(out_pos, out_neg, atol=1e-10)


def test_ce_2d():
    array1 = np.array([[0.2, 0.8, 0.5, 0.1], [1.5, 0.3, 0.7, 3.76], [0.22, 0.28, 0.25, 9.1]])
    array2 = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float64)
    _ce_helper(array1=array1, array2=array2)


def test_ce_3d():
    rng = np.random.default_rng(10)
    shape = (3, 4, 5)
    array1 = rng.standard_normal(shape)
    array2 = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            array2[i, j, rng.integers(0, shape[2])] = 1
    _ce_helper(array1=array1, array2=array2)


def test_ce_4d():
    rng = np.random.default_rng(11)
    shape = (3, 4, 5, 6)
    array1 = rng.standard_normal(shape)
    array2 = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                array2[i, j, k, rng.integers(0, shape[3])] = 1
    _ce_helper(array1=array1, array2=array2)


def test_ce_dim_0():
    rng = np.random.default_rng(12)
    shape = (6, 3, 4)
    array1 = rng.standard_normal(shape)
    array2 = np.zeros(shape)
    for j in range(shape[1]):
        for k in range(shape[2]):
            array2[rng.integers(0, shape[0]), j, k] = 1
    _ce_helper(array1=array1, array2=array2, dim=0)


def test_ce_dim_1():
    rng = np.random.default_rng(13)
    shape = (3, 6, 4)
    array1 = rng.standard_normal(shape)
    array2 = np.zeros(shape)
    for i in range(shape[0]):
        for k in range(shape[2]):
            array2[i, rng.integers(0, shape[1]), k] = 1
    _ce_helper(array1=array1, array2=array2, dim=1)


def test_ce_dim_negative_1():
    rng = np.random.default_rng(14)
    shape = (3, 4, 6)
    array1 = rng.standard_normal(shape)
    array2 = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            array2[i, j, rng.integers(0, shape[2])] = 1
    _ce_helper(array1=array1, array2=array2, dim=-1)


def test_ce_dim_negative_2():
    rng = np.random.default_rng(15)
    shape = (3, 6, 4)
    array1 = rng.standard_normal(shape)
    array2 = np.zeros(shape)
    for i in range(shape[0]):
        for k in range(shape[2]):
            array2[i, rng.integers(0, shape[1]), k] = 1
    _ce_helper(array1=array1, array2=array2, dim=-2)


def test_ce_dim_negative_3():
    rng = np.random.default_rng(16)
    shape = (6, 3, 4)
    array1 = rng.standard_normal(shape)
    array2 = np.zeros(shape)
    for j in range(shape[1]):
        for k in range(shape[2]):
            array2[rng.integers(0, shape[0]), j, k] = 1
    _ce_helper(array1=array1, array2=array2, dim=-3)


def test_ce_loss_numerical_stability():
    # ce_loss uses log-sum-exp internally, so large logits must not produce NaN/Inf
    z = sg.Tensor(np.array([[1000.0, 1001.0, 1002.0]], dtype=np.float64), dtype="float64")
    y = sg.Tensor(np.array([[0.0, 0.0, 1.0]], dtype=np.float64), dtype="float64")
    loss = sg.ce_loss(z, y, dim=-1)
    assert not np.any(np.isnan(loss.values)), "ce_loss produced NaN for large logits"
    assert not np.any(np.isinf(loss.values)), "ce_loss produced Inf for large logits"
