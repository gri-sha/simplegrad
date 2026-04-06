"""Tests for loss functions: mse_loss, ce_loss."""

import numpy as np
import simplegrad as sg
import torch
from .utils import compare2tensors


def test_mse_mean():
    array1 = np.array([[0.2, 0.8, 0.5], [1.5, 0.3, 0.7]])
    array2 = np.array([[0, 1, 0], [1, 0, 0]])

    predictions = sg.Tensor(array1, dtype="float64")
    targets = sg.Tensor(array2, dtype="float64")
    predictions_t = torch.from_numpy(array1).to(torch.float64).requires_grad_(True)
    targets_t = torch.from_numpy(array2).to(torch.float64)

    mse_loss = sg.mse_loss(predictions, targets, reduction="mean")
    mse_loss.zero_grad()
    mse_loss.backward()

    mse_loss_t = torch.nn.functional.mse_loss(predictions_t, targets_t, reduction="mean")
    mse_loss_t.backward()

    compare2tensors(sg=mse_loss, pt=mse_loss_t)
    compare2tensors(sg=predictions.grad, pt=predictions_t.grad)


def test_mse_sum():
    array1 = np.array([[0.2, 0.8, 0.5], [1.5, 0.3, 0.7]])
    array2 = np.array([[0, 1, 0], [1, 0, 0]])

    predictions = sg.Tensor(array1, dtype="float64")
    targets = sg.Tensor(array2, dtype="float64")
    predictions_t = torch.from_numpy(array1).to(torch.float64).requires_grad_(True)
    targets_t = torch.from_numpy(array2).to(torch.float64)

    mse_loss = sg.mse_loss(predictions, targets, reduction="sum")
    mse_loss.zero_grad()
    mse_loss.backward()

    mse_loss_t = torch.nn.functional.mse_loss(predictions_t, targets_t, reduction="sum")
    mse_loss_t.backward()

    compare2tensors(sg=mse_loss, pt=mse_loss_t)
    compare2tensors(sg=predictions.grad, pt=predictions_t.grad)


def test_mse_no_reduction():
    array1 = np.array([[0.2, 0.8, 0.5], [1.5, 0.3, 0.7]])
    array2 = np.array([[0, 1, 0], [1, 0, 0]])

    predictions = sg.Tensor(array1, dtype="float64")
    targets = sg.Tensor(array2, dtype="float64")
    predictions_t = torch.from_numpy(array1).to(torch.float64).requires_grad_(True)
    targets_t = torch.from_numpy(array2).to(torch.float64)

    mse_loss = sg.mse_loss(predictions, targets, reduction=None)
    mse_loss_sum = sg.sum(mse_loss)
    mse_loss_sum.zero_grad()
    mse_loss_sum.backward()

    mse_loss_t = torch.nn.functional.mse_loss(predictions_t, targets_t, reduction="none")
    mse_loss_sum_t = torch.sum(mse_loss_t)
    mse_loss_sum_t.backward()

    compare2tensors(sg=mse_loss, pt=mse_loss_t)
    compare2tensors(sg=predictions.grad, pt=predictions_t.grad)


def _ce_helper(array1, array2, dim=-1):
    # Test reduction='mean'
    predictions = sg.Tensor(array1, dtype="float64")
    targets = sg.Tensor(array2, dtype="float64")
    predictions_t = torch.from_numpy(array1).to(torch.float64).requires_grad_(True)
    targets_t = torch.from_numpy(array2).to(torch.float64)

    ce_loss = sg.ce_loss(predictions, targets, dim=dim, reduction="mean")
    ce_loss.zero_grad()
    ce_loss.backward()

    log_softmax_t = torch.log_softmax(predictions_t, dim=dim)
    ce_loss_t = -torch.sum(targets_t * log_softmax_t, dim=dim, keepdim=True).mean()
    ce_loss_t.backward()

    compare2tensors(sg=ce_loss, pt=ce_loss_t)
    compare2tensors(sg=predictions.grad, pt=predictions_t.grad)

    # Test reduction='sum'
    predictions = sg.Tensor(array1, dtype="float64")
    targets = sg.Tensor(array2, dtype="float64")
    predictions_t = torch.from_numpy(array1).to(torch.float64).requires_grad_(True)
    targets_t = torch.from_numpy(array2).to(torch.float64)

    ce_loss = sg.ce_loss(predictions, targets, dim=dim, reduction="sum")
    ce_loss.zero_grad()
    ce_loss.backward()

    log_softmax_t = torch.log_softmax(predictions_t, dim=dim)
    ce_loss_t = -torch.sum(targets_t * log_softmax_t, dim=dim, keepdim=True).sum()
    ce_loss_t.backward()

    compare2tensors(sg=ce_loss, pt=ce_loss_t)
    compare2tensors(sg=predictions.grad, pt=predictions_t.grad)

    # Test reduction=None
    predictions = sg.Tensor(array1, dtype="float64")
    targets = sg.Tensor(array2, dtype="float64")
    predictions_t = torch.from_numpy(array1).to(torch.float64).requires_grad_(True)
    targets_t = torch.from_numpy(array2).to(torch.float64)

    ce_loss = sg.ce_loss(predictions, targets, dim=dim, reduction=None)
    ce_loss_sum = sg.sum(ce_loss)
    ce_loss_sum.zero_grad()
    ce_loss_sum.backward()

    log_softmax_t = torch.log_softmax(predictions_t, dim=dim)
    ce_loss_t = -torch.sum(targets_t * log_softmax_t, dim=dim, keepdim=True)
    ce_loss_sum_t = torch.sum(ce_loss_t)
    ce_loss_sum_t.backward()

    compare2tensors(sg=ce_loss, pt=ce_loss_t)
    compare2tensors(sg=predictions.grad, pt=predictions_t.grad)


def test_ce_2d():
    array1 = np.array([[0.2, 0.8, 0.5, 0.1], [1.5, 0.3, 0.7, 3.76], [0.22, 0.28, 0.25, 9.1]])
    array2 = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
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
