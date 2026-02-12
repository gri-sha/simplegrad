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
def test_operations_part1():
    array1 = np.array([[0.5, 1.2, -0.3], [0.1, -0.5, 2.1]])
    array2 = np.array([[1.0, 0.5, 0.8], [0.3, 1.5, 0.2]])
    a, b = sg.Tensor(array1, dtype="float64"), sg.Tensor(array2, dtype="float64")

    c = sg.exp(a) * sg.log(b) + sg.sin(a * b) + sg.cos(a + 1) - sg.tan(a - b)
    c.zero_grad()
    c.backward()

    at = torch.from_numpy(array1).to(torch.float64).requires_grad_(True)
    bt = torch.from_numpy(array2).to(torch.float64).requires_grad_(True)
    ct = torch.exp(at) * torch.log(bt) + torch.sin(at * bt) + torch.cos(at + 1) - torch.tan(at - bt)
    loss = ct.sum()
    loss.backward()

    compare2tensors(sg=c, pt=ct)
    compare2tensors(sg=a.grad, pt=at.grad)
    compare2tensors(sg=b.grad, pt=bt.grad)


# sum, mean, trace
def test_operations_part2():
    array1 = np.array([[2.0, 3.0, 1.5], [0.5, 2.5, 1.0], [1.2, 0.8, 2.2]])
    a = sg.Tensor(array1, dtype="float64")
    c = sg.sum(a, dim=1) + sg.mean(a, dim=0) + sg.trace(a)
    c.zero_grad()
    c.backward()

    at = torch.from_numpy(array1).to(torch.float64).requires_grad_(True)
    ct = torch.sum(at, dim=1, keepdim=True) + torch.mean(at, dim=0, keepdim=True) + torch.trace(at)
    loss = ct.sum()
    loss.backward()

    compare2tensors(sg=c, pt=ct)
    compare2tensors(sg=a.grad, pt=at.grad)


# softmax, relu, tanh
def activations_helper(arrays, dims):
    for array, dim in zip(arrays, dims):
        a = sg.Tensor(array, dtype="float64")
        softmax_out = sg.softmax(a, dim=dim)
        tanh_out = sg.tanh(softmax_out)
        c = sg.relu(tanh_out)
        c.zero_grad()
        c.backward()

        at = torch.from_numpy(array).to(torch.float64).requires_grad_(True)
        softmax_out_t = torch.softmax(at, dim=dim)
        tanh_out_t = torch.tanh(softmax_out_t)
        ct = torch.relu(tanh_out_t)
        loss = ct.sum()
        loss.backward()

        compare2tensors(sg=c, pt=ct)
        compare2tensors(sg=a.grad, pt=at.grad)


def test_activations():
    array1 = np.array([[1.2, -0.5, 2.1], [-1.0, 0.3, 1.5], [0.8, -2.0, 0.5]])
    array2 = np.array([1.2, -0.5, 2.1, -1.0, 0.3, 1.5, 0.8, -2.0, 0.5])
    array3 = np.array([[-1.0, 2.0], [3.0, -0.5], [0.0, 1.0], [2.5, -2.5]])
    activations_helper([array1, array2, array3], [1, -1, 0])


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

    mse_loss_t = torch.nn.functional.mse_loss(predictions_t, targets_t, reduction="mean")
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

    mse_loss_t = torch.nn.functional.mse_loss(predictions_t, targets_t, reduction="none")
    mse_loss_sum_t = torch.sum(mse_loss_t)
    mse_loss_sum_t.backward()

    print("Pt shape:", mse_loss_t.shape)
    print("Pt values:\n", mse_loss_t.detach())
    print("Pt Grad:\n", predictions_t.grad)

    compare2tensors(sg=mse_loss, pt=mse_loss_t)
    compare2tensors(sg=predictions.grad, pt=predictions_t.grad)


def _test_ce_helper(array1=None, array2=None, dim=-1):
    # Use manual cross-entropy calculation to match simplegrad's implementation
    # simplegrad uses one-hot targets and computes CE along specified dim
    # PyTorch's cross_entropy expects class indices or handles soft labels differently

    # Test reduction='mean' (default)
    predictions = sg.Tensor(array1, dtype="float64")
    targets = sg.Tensor(array2, dtype="float64")

    predictions_t = torch.from_numpy(array1).to(torch.float64).requires_grad_(True)
    targets_t = torch.from_numpy(array2).to(torch.float64)

    ce_loss = sg.ce_loss(predictions, targets, dim=dim, reduction="mean")
    ce_loss.zero_grad()
    ce_loss.backward()

    # Manual cross-entropy matching simplegrad's implementation
    log_softmax_t = torch.log_softmax(predictions_t, dim=dim)
    ce_loss_t = -torch.sum(targets_t * log_softmax_t, dim=dim, keepdim=True).mean()
    ce_loss_t.backward()

    compare2tensors(sg=ce_loss, pt=ce_loss_t)
    print("mean forward: ok")
    compare2tensors(sg=predictions.grad, pt=predictions_t.grad)
    print("mean backward: ok")

    # Test reduction='sum'
    predictions = sg.Tensor(array1, dtype="float64")
    targets = sg.Tensor(array2, dtype="float64")

    predictions_t = torch.from_numpy(array1).to(torch.float64).requires_grad_(True)
    targets_t = torch.from_numpy(array2).to(torch.float64)

    ce_loss = sg.ce_loss(predictions, targets, dim=dim, reduction="sum")
    ce_loss.zero_grad()
    ce_loss.backward()

    # Manual cross-entropy matching simplegrad's implementation
    log_softmax_t = torch.log_softmax(predictions_t, dim=dim)
    ce_loss_t = -torch.sum(targets_t * log_softmax_t, dim=dim, keepdim=True).sum()
    ce_loss_t.backward()

    compare2tensors(sg=ce_loss, pt=ce_loss_t)
    print("sum forward: ok")
    compare2tensors(sg=predictions.grad, pt=predictions_t.grad)
    print("sum backward: ok")

    # Test reduction=None
    predictions = sg.Tensor(array1, dtype="float64")
    targets = sg.Tensor(array2, dtype="float64")

    predictions_t = torch.from_numpy(array1).to(torch.float64).requires_grad_(True)
    targets_t = torch.from_numpy(array2).to(torch.float64)

    ce_loss = sg.ce_loss(predictions, targets, dim=dim, reduction=None)
    ce_loss_sum = sg.sum(ce_loss)  # Sum to get scalar for backward
    ce_loss_sum.zero_grad()
    ce_loss_sum.backward()

    # Manual cross-entropy matching simplegrad's implementation
    log_softmax_t = torch.log_softmax(predictions_t, dim=dim)
    ce_loss_t = -torch.sum(targets_t * log_softmax_t, dim=dim, keepdim=True)
    ce_loss_sum_t = torch.sum(ce_loss_t)
    ce_loss_sum_t.backward()
    print("ce_loss shape sg:", ce_loss.values.shape)
    print("ce_loss shape pt:", ce_loss_t.shape)
    compare2tensors(sg=ce_loss, pt=ce_loss_t)
    print("none forward: ok")
    compare2tensors(sg=predictions.grad, pt=predictions_t.grad)
    print("none backward: ok")


def test_ce_1():
    array1 = np.array([[0.2, 0.8, 0.5, 0.1], [1.5, 0.3, 0.7, 3.76], [0.22, 0.28, 0.25, 9.1]])
    array2 = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
    _test_ce_helper(array1=array1, array2=array2)


def test_ce_2():
    shape = (3, 4, 5)
    array1 = np.random.randn(*shape)
    array2 = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            target_index = np.random.randint(0, shape[2])
            array2[i, j, target_index] = 1
    _test_ce_helper(array1=array1, array2=array2)


def test_ce_3():
    shape = (3, 4, 5, 6)
    array1 = np.random.randn(*shape)
    array2 = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                target_index = np.random.randint(0, shape[3])
                array2[i, j, k, target_index] = 1
    _test_ce_helper(array1=array1, array2=array2)


def test_ce_dim_0():
    """Test ce_loss with dim=0 (softmax over first dimension)"""
    shape = (6, 3, 4)  # 6 classes along dim 0
    array1 = np.random.randn(*shape)
    array2 = np.zeros(shape)
    # One-hot along dim 0
    for j in range(shape[1]):
        for k in range(shape[2]):
            target_index = np.random.randint(0, shape[0])
            array2[target_index, j, k] = 1
    _test_ce_helper(array1=array1, array2=array2, dim=0)


def test_ce_dim_1():
    """Test ce_loss with dim=1 (softmax over second dimension)"""
    shape = (3, 6, 4)  # 6 classes along dim 1
    array1 = np.random.randn(*shape)
    array2 = np.zeros(shape)
    # One-hot along dim 1
    for i in range(shape[0]):
        for k in range(shape[2]):
            target_index = np.random.randint(0, shape[1])
            array2[i, target_index, k] = 1
    _test_ce_helper(array1=array1, array2=array2, dim=1)


def test_ce_dim_2():
    """Test ce_loss with dim=2 (softmax over third dimension)"""
    shape = (3, 4, 6)  # 6 classes along dim 2
    array1 = np.random.randn(*shape)
    array2 = np.zeros(shape)
    # One-hot along dim 2
    for i in range(shape[0]):
        for j in range(shape[1]):
            target_index = np.random.randint(0, shape[2])
            array2[i, j, target_index] = 1
    _test_ce_helper(array1=array1, array2=array2, dim=2)


def test_ce_dim_negative_1():
    """Test ce_loss with dim=-1 (last dimension, same as default)"""
    shape = (3, 4, 6)  # 6 classes along dim -1
    array1 = np.random.randn(*shape)
    array2 = np.zeros(shape)
    # One-hot along dim -1
    for i in range(shape[0]):
        for j in range(shape[1]):
            target_index = np.random.randint(0, shape[2])
            array2[i, j, target_index] = 1
    _test_ce_helper(array1=array1, array2=array2, dim=-1)


def test_ce_dim_negative_2():
    """Test ce_loss with dim=-2 (second to last dimension)"""
    shape = (3, 6, 4)  # 6 classes along dim -2 (which is dim 1)
    array1 = np.random.randn(*shape)
    array2 = np.zeros(shape)
    # One-hot along dim -2 (same as dim 1)
    for i in range(shape[0]):
        for k in range(shape[2]):
            target_index = np.random.randint(0, shape[1])
            array2[i, target_index, k] = 1
    _test_ce_helper(array1=array1, array2=array2, dim=-2)


def test_ce_dim_negative_3():
    """Test ce_loss with dim=-3 (third to last dimension)"""
    shape = (6, 3, 4)  # 6 classes along dim -3 (which is dim 0)
    array1 = np.random.randn(*shape)
    array2 = np.zeros(shape)
    # One-hot along dim -3 (same as dim 0)
    for j in range(shape[1]):
        for k in range(shape[2]):
            target_index = np.random.randint(0, shape[0])
            array2[target_index, j, k] = 1
    _test_ce_helper(array1=array1, array2=array2, dim=-3)


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
