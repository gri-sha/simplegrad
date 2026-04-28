"""Tests for loss functions: mse_loss, ce_loss."""

import numpy as np
import simplegrad as sg
from .utils import gradcheck


def test_mse_mean():
    predictions = sg.Tensor(np.array([[0.2, 0.8, 0.5], [1.5, 0.3, 0.7]]), dtype="float64")
    targets = sg.Tensor(np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]), dtype="float64")

    gradcheck(lambda: sg.mse_loss(predictions, targets, reduction="mean"), [predictions])


def test_mse_sum():
    predictions = sg.Tensor(np.array([[0.2, 0.8, 0.5], [1.5, 0.3, 0.7]]), dtype="float64")
    targets = sg.Tensor(np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]), dtype="float64")

    gradcheck(lambda: sg.mse_loss(predictions, targets, reduction="sum"), [predictions])


def test_mse_no_reduction():
    predictions = sg.Tensor(np.array([[0.2, 0.8, 0.5], [1.5, 0.3, 0.7]]), dtype="float64")
    targets = sg.Tensor(np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]), dtype="float64")

    gradcheck(lambda: sg.mse_loss(predictions, targets, reduction=None), [predictions])


def _ce_helper(array1, array2, dim=-1):
    for reduction in ["mean", "sum", None]:
        predictions = sg.Tensor(array1.copy(), dtype="float64")
        targets = sg.Tensor(array2.copy(), dtype="float64")

        def fn(pred=predictions, targ=targets, red=reduction):
            return sg.ce_loss(pred, targ, dim=dim, reduction=red)

        gradcheck(fn, [predictions])


def test_ce_2d():
    array1 = np.array([[0.2, 0.8, 0.5, 0.1], [1.5, 0.3, 0.7, 3.76], [0.22, 0.28, 0.25, 9.1]])
    array2 = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float64)
    _ce_helper(array1=array1, array2=array2)


def test_ce_3d():
    shape = (3, 4, 5)
    array1 = np.random.randn(*shape)
    array2 = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            array2[i, j, np.random.randint(0, shape[2])] = 1
    _ce_helper(array1=array1, array2=array2)


def test_ce_4d():
    shape = (3, 4, 5, 6)
    array1 = np.random.randn(*shape)
    array2 = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                array2[i, j, k, np.random.randint(0, shape[3])] = 1
    _ce_helper(array1=array1, array2=array2)


def test_ce_dim_0():
    shape = (6, 3, 4)
    array1 = np.random.randn(*shape)
    array2 = np.zeros(shape)
    for j in range(shape[1]):
        for k in range(shape[2]):
            array2[np.random.randint(0, shape[0]), j, k] = 1
    _ce_helper(array1=array1, array2=array2, dim=0)


def test_ce_dim_1():
    shape = (3, 6, 4)
    array1 = np.random.randn(*shape)
    array2 = np.zeros(shape)
    for i in range(shape[0]):
        for k in range(shape[2]):
            array2[i, np.random.randint(0, shape[1]), k] = 1
    _ce_helper(array1=array1, array2=array2, dim=1)


def test_ce_dim_2():
    shape = (3, 4, 6)
    array1 = np.random.randn(*shape)
    array2 = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            array2[i, j, np.random.randint(0, shape[2])] = 1
    _ce_helper(array1=array1, array2=array2, dim=2)


def test_ce_dim_negative_1():
    shape = (3, 4, 6)
    array1 = np.random.randn(*shape)
    array2 = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            array2[i, j, np.random.randint(0, shape[2])] = 1
    _ce_helper(array1=array1, array2=array2, dim=-1)


def test_ce_dim_negative_2():
    shape = (3, 6, 4)
    array1 = np.random.randn(*shape)
    array2 = np.zeros(shape)
    for i in range(shape[0]):
        for k in range(shape[2]):
            array2[i, np.random.randint(0, shape[1]), k] = 1
    _ce_helper(array1=array1, array2=array2, dim=-2)


def test_ce_dim_negative_3():
    shape = (6, 3, 4)
    array1 = np.random.randn(*shape)
    array2 = np.zeros(shape)
    for j in range(shape[1]):
        for k in range(shape[2]):
            array2[np.random.randint(0, shape[0]), j, k] = 1
    _ce_helper(array1=array1, array2=array2, dim=-3)
