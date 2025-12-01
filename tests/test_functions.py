import simplegrad as sg
import torch
import numpy as np
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


# exp, log, sin, cos, tan
def test_operations_1():
    array1 = np.array([[0.5, 1.2, -0.3], [0.1, -0.5, 2.1]])
    array2 = np.array([[1.0, 0.5, 0.8], [0.3, 1.5, 0.2]])
    a, b = sg.Tensor(array1, dtype="float64"), sg.Tensor(array2, dtype="float64")

    c = sg.exp(a) * sg.log(b) + sg.sin(a * b) + sg.cos(a + 1) - sg.tan(a - b)
    c.zero_grad()
    c.backward()

    at = torch.from_numpy(array1).to(torch.float64).requires_grad_(True)
    bt = torch.from_numpy(array2).to(torch.float64).requires_grad_(True)
    ct = (
        torch.exp(at) * torch.log(bt)
        + torch.sin(at * bt)
        + torch.cos(at + 1)
        - torch.tan(at - bt)
    )
    loss = ct.sum()
    loss.backward()

    compare2tensors(sg=c, pt=ct)
    compare2tensors(sg=a.grad, pt=at.grad)
    compare2tensors(sg=b.grad, pt=bt.grad)


# sum, mean, trace
def test_operations_2():
    array1 = np.array([[2.0, 3.0, 1.5], [0.5, 2.5, 1.0], [1.2, 0.8, 2.2]])
    a = sg.Tensor(array1, dtype="float64")
    c = sg.sum(a, dim=1) + sg.mean(a, dim=0) + sg.trace(a)
    c.zero_grad()
    c.backward()

    at = torch.from_numpy(array1).to(torch.float64).requires_grad_(True)
    ct = (
        torch.sum(at, dim=1, keepdim=True)
        + torch.mean(at, dim=0, keepdim=True)
        + torch.trace(at)
    )
    loss = ct.sum()
    loss.backward()

    compare2tensors(sg=c, pt=ct)
    compare2tensors(sg=a.grad, pt=at.grad)


# softmax, relu, tanh
def test_activations():
    array1 = np.array([[1.2, -0.5, 2.1], [-1.0, 0.3, 1.5], [0.8, -2.0, 0.5]])
    a = sg.Tensor(array1, dtype="float64")
    softmax_out = sg.softmax(a, dim=1)
    tanh_out = sg.tanh(softmax_out)
    c = sg.relu(tanh_out)
    c.zero_grad()
    c.backward()

    at = torch.from_numpy(array1).to(torch.float64).requires_grad_(True)
    softmax_out_t = torch.softmax(at, dim=1)
    tanh_out_t = torch.tanh(softmax_out_t)
    ct = torch.relu(tanh_out_t)
    loss = ct.sum()
    loss.backward()

    compare2tensors(sg=c, pt=ct)
    compare2tensors(sg=a.grad, pt=at.grad)


# Complex shapes with broadcasting: +, *, @
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


# Complex shapes with broadcasting: +, *, @
def test_broadcastiong_2():
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


def test_mse():
    array1 = np.array([[0.2, 0.8, 0.5], [1.5, 0.3, 0.7]])
    array2 = np.array([[0, 1, 0], [1, 0, 0]])

    # Test reduction='mean' (default)
    predictions = sg.Tensor(array1, dtype="float64")
    targets = sg.Tensor(array2, dtype="float64")

    predictions_t = torch.from_numpy(array1).to(torch.float64).requires_grad_(True)
    targets_t = torch.from_numpy(array2).to(torch.float64)

    mse_loss = sg.mse_loss(predictions, targets, reduction="mean")
    mse_loss.zero_grad()
    mse_loss.backward()

    print("Sg:", mse_loss.values)
    print("Sg Grad:\n", predictions.grad)

    mse_loss_t = torch.nn.functional.mse_loss(
        predictions_t, targets_t, reduction="mean"
    )
    mse_loss_t.backward()

    print("Pt:", mse_loss_t.item())
    print("Pt Grad:\n", predictions_t.grad)

    compare2tensors(sg=mse_loss, pt=mse_loss_t)
    compare2tensors(sg=predictions.grad, pt=predictions_t.grad)

    # Test reduction='sum'
    predictions = sg.Tensor(array1, dtype="float64")
    targets = sg.Tensor(array2, dtype="float64")

    predictions_t = torch.from_numpy(array1).to(torch.float64).requires_grad_(True)
    targets_t = torch.from_numpy(array2).to(torch.float64)

    mse_loss = sg.mse_loss(predictions, targets, reduction="sum")
    mse_loss.zero_grad()
    mse_loss.backward()

    print("Sg:", mse_loss.values)
    print("Sg Grad:\n", predictions.grad)

    mse_loss_t = torch.nn.functional.mse_loss(predictions_t, targets_t, reduction="sum")
    mse_loss_t.backward()

    print("Pt:", mse_loss_t.item())
    print("Pt Grad:\n", predictions_t.grad)

    compare2tensors(sg=mse_loss, pt=mse_loss_t)
    compare2tensors(sg=predictions.grad, pt=predictions_t.grad)

    # Test reduction=None
    predictions = sg.Tensor(array1, dtype="float64")
    targets = sg.Tensor(array2, dtype="float64")

    predictions_t = torch.from_numpy(array1).to(torch.float64).requires_grad_(True)
    targets_t = torch.from_numpy(array2).to(torch.float64)

    mse_loss = sg.mse_loss(predictions, targets, reduction=None)
    mse_loss_sum = sg.sum(mse_loss)  # Sum to get scalar for backward
    mse_loss_sum.zero_grad()
    mse_loss_sum.backward()

    print("Sg shape:", mse_loss.values.shape)
    print("Sg values:\n", mse_loss.values)
    print("Sg Grad:\n", predictions.grad)

    mse_loss_t = torch.nn.functional.mse_loss(
        predictions_t, targets_t, reduction="none"
    )
    mse_loss_sum_t = torch.sum(mse_loss_t)
    mse_loss_sum_t.backward()

    print("Pt shape:", mse_loss_t.shape)
    print("Pt values:\n", mse_loss_t.detach())
    print("Pt Grad:\n", predictions_t.grad)

    compare2tensors(sg=mse_loss, pt=mse_loss_t)
    compare2tensors(sg=predictions.grad, pt=predictions_t.grad)


def test_ce():
    array1 = np.array(
        [[0.2, 0.8, 0.5, 0.1], [1.5, 0.3, 0.7, 3.76], [0.22, 0.28, 0.25, 9.1]]
    )
    array2 = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]])

    # Test reduction='mean' (default)
    predictions = sg.Tensor(array1, dtype="float64")
    targets = sg.Tensor(array2, dtype="float64")

    predictions_t = torch.from_numpy(array1).to(torch.float64).requires_grad_(True)
    targets_t = torch.from_numpy(array2).to(torch.float64)

    ce_loss = sg.ce_loss(predictions, targets, reduction="mean")
    ce_loss.zero_grad()
    ce_loss.backward()

    print("Sg:", ce_loss.values)
    print("Sg Grad:\n", predictions.grad)

    ce_loss_t = torch.nn.functional.cross_entropy(
        predictions_t, targets_t, reduction="mean"
    )
    ce_loss_t.backward()

    print("Pt:", ce_loss_t.item())
    print("Pt Grad:\n", predictions_t.grad)

    compare2tensors(sg=ce_loss, pt=ce_loss_t)
    compare2tensors(sg=predictions.grad, pt=predictions_t.grad)

    # Test reduction='sum'
    predictions = sg.Tensor(array1, dtype="float64")
    targets = sg.Tensor(array2, dtype="float64")

    predictions_t = torch.from_numpy(array1).to(torch.float64).requires_grad_(True)
    targets_t = torch.from_numpy(array2).to(torch.float64)

    ce_loss = sg.ce_loss(predictions, targets, reduction="sum")
    ce_loss.zero_grad()
    ce_loss.backward()

    print("Sg:", ce_loss.values)
    print("Sg Grad:\n", predictions.grad)

    ce_loss_t = torch.nn.functional.cross_entropy(
        predictions_t, targets_t, reduction="sum"
    )
    ce_loss_t.backward()

    print("Pt:", ce_loss_t.item())
    print("Pt Grad:\n", predictions_t.grad)

    compare2tensors(sg=ce_loss, pt=ce_loss_t)
    compare2tensors(sg=predictions.grad, pt=predictions_t.grad)

    # Test reduction=None
    predictions = sg.Tensor(array1, dtype="float64")
    targets = sg.Tensor(array2, dtype="float64")

    predictions_t = torch.from_numpy(array1).to(torch.float64).requires_grad_(True)
    targets_t = torch.from_numpy(array2).to(torch.float64)

    ce_loss = sg.ce_loss(predictions, targets, reduction=None)
    ce_loss_sum = sg.sum(ce_loss)  # Sum to get scalar for backward
    ce_loss_sum.zero_grad()
    ce_loss_sum.backward()

    print("Sg shape:", ce_loss.values.shape)
    print("Sg values:\n", ce_loss.values)
    print("Sg Grad:\n", predictions.grad)

    ce_loss_t = torch.nn.functional.cross_entropy(
        predictions_t, targets_t, reduction="none"
    )
    ce_loss_sum_t = torch.sum(ce_loss_t)
    ce_loss_sum_t.backward()

    print("Pt shape:", ce_loss_t.shape)
    print("Pt values:\n", ce_loss_t.detach())
    print("Pt Grad:\n", predictions_t.grad)

    compare2tensors(sg=ce_loss, pt=ce_loss_t)
    compare2tensors(sg=predictions.grad, pt=predictions_t.grad)


def test_flatten():
    """Test flatten operation forward and backward pass"""
    # Test flattening a 4D tensor (typical CNN output) from dim 1 to -1
    array1 = np.random.randn(2, 3, 4, 5).astype(np.float64)

    a = sg.Tensor(array1, dtype="float64")
    flattened = sg.flatten(a, start_dim=1, end_dim=-1)
    result = flattened * 2 + 1  # Add some operations after flatten
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
