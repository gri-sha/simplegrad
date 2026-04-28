"""Tests for device management and multi-backend dispatch."""

import numpy as np
import pytest
import simplegrad as sg
from simplegrad.core.devices import (
    available_devices,
    cuda_is_available,
    get_default_device,
    set_default_device,
    validate_device,
    validate_same_device,
    get_backend,
)
import simplegrad.nn as nn

CUDA_AVAILABLE = cuda_is_available()
needs_cuda = pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")


@pytest.fixture(autouse=True)
def restore_default_device():
    """Reset the global default device to 'cpu' after every test."""
    yield
    set_default_device("cpu")


# available_devices

def test_available_devices_returns_dict():
    devices = available_devices()
    assert isinstance(devices, dict)


def test_available_devices_always_has_cpu():
    devices = available_devices()
    assert "cpu" in devices


def test_available_devices_cpu_description():
    devices = available_devices()
    assert isinstance(devices["cpu"], str)
    assert len(devices["cpu"]) > 0


@needs_cuda
def test_available_devices_has_cuda():
    devices = available_devices()
    assert any(k.startswith("cuda:") for k in devices)


@needs_cuda
def test_available_devices_cuda_description():
    devices = available_devices()
    cuda_keys = [k for k in devices if k.startswith("cuda:")]
    for key in cuda_keys:
        assert isinstance(devices[key], str)
        assert len(devices[key]) > 0


# cuda_is_available

def test_cuda_is_available_returns_bool():
    assert isinstance(cuda_is_available(), bool)


# validate_device

def test_validate_device_cpu():
    assert validate_device("cpu") == "cpu"


def test_validate_device_cuda():
    assert validate_device("cuda:0") == "cuda:0"
    assert validate_device("cuda:3") == "cuda:3"


def test_validate_device_invalid():
    with pytest.raises(ValueError):
        validate_device("gpu")
    with pytest.raises(ValueError):
        validate_device("cuda")
    with pytest.raises(ValueError):
        validate_device("cuda:")
    with pytest.raises(ValueError):
        validate_device("CPU")


# get_backend

def test_get_backend_cpu_is_numpy():
    import numpy as np_mod
    assert get_backend("cpu") is np_mod


@needs_cuda
def test_get_backend_cuda_is_cupy():
    import cupy as cp
    assert get_backend("cuda:0") is cp


def test_get_backend_invalid():
    with pytest.raises(ValueError):
        get_backend("tpu:0")


# default device

def test_default_device_is_cpu():
    assert get_default_device() == "cpu"


def test_set_default_device():
    set_default_device("cpu")
    assert get_default_device() == "cpu"


def test_set_default_device_invalid():
    with pytest.raises(ValueError):
        set_default_device("gpu")


# validate_same_device

def test_validate_same_device_single():
    a = sg.Tensor([1.0], device="cpu")
    assert validate_same_device(a) == "cpu"


def test_validate_same_device_matching():
    a = sg.Tensor([1.0], device="cpu")
    b = sg.Tensor([2.0], device="cpu")
    assert validate_same_device(a, b) == "cpu"


def test_validate_same_device_mismatch():
    a = sg.Tensor([1.0], device="cpu")
    b = sg.Tensor.__new__(sg.Tensor)
    b.device = "cuda:0"
    with pytest.raises(RuntimeError, match="same device"):
        validate_same_device(a, b)


def test_validate_same_device_empty():
    result = validate_same_device()
    assert result == get_default_device()


# Tensor device attribute

def test_tensor_default_device():
    x = sg.Tensor([1.0, 2.0])
    assert x.device == "cpu"


def test_tensor_explicit_device():
    x = sg.Tensor([1.0, 2.0], device="cpu")
    assert x.device == "cpu"
    assert isinstance(x.values, np.ndarray)


def test_tensor_values_are_numpy_on_cpu():
    x = sg.Tensor([1.0, 2.0, 3.0])
    assert isinstance(x.values, np.ndarray)


# factory functions — CPU

def test_zeros_device():
    t = sg.zeros((3, 4), device="cpu")
    assert t.device == "cpu"
    assert isinstance(t.values, np.ndarray)
    assert t.values.shape == (3, 4)
    assert np.all(t.values == 0)


def test_ones_device():
    t = sg.ones((2, 3), device="cpu")
    assert t.device == "cpu"
    assert np.all(t.values == 1)


def test_normal_device():
    t = sg.normal((10,), device="cpu")
    assert t.device == "cpu"
    assert t.values.shape == (10,)


def test_uniform_device():
    t = sg.uniform((10,), low=0, high=1, device="cpu")
    assert t.device == "cpu"
    assert np.all(t.values >= 0) and np.all(t.values <= 1)


def test_full_device():
    t = sg.full((3,), fill_value=7.0, device="cpu")
    assert t.device == "cpu"
    assert np.all(t.values == 7.0)


# to_device — cpu → cpu

def test_to_device_cpu_to_cpu():
    x = sg.Tensor([1.0, 2.0, 3.0])
    y = x.to_device("cpu")
    assert y.device == "cpu"
    assert isinstance(y.values, np.ndarray)
    assert np.allclose(y.values, x.values)


def test_to_device_returns_new_tensor():
    x = sg.Tensor([1.0, 2.0])
    y = x.to_device("cpu")
    assert x is not y


def test_to_device_unrealized_raises():
    t = sg.Tensor.deferred(lambda: np.array([1.0]), shape=(1,))
    with pytest.raises(RuntimeError, match="unrealized"):
        t.to_device("cpu")


def test_to_device_invalid_device():
    x = sg.Tensor([1.0])
    with pytest.raises(ValueError):
        x.to_device("tpu:0")


# ops raise on mixed-device inputs

def test_mixed_device_op_raises():
    a = sg.Tensor([1.0], device="cpu")
    b = sg.Tensor.__new__(sg.Tensor)
    b.device = "cuda:0"
    b.values = np.array([2.0])
    b.shape = (1,)
    b.comp_grad = False
    with pytest.raises(RuntimeError, match="same device"):
        _ = a + b


# CUDA-specific tests

@needs_cuda
def test_tensor_on_cuda():
    x = sg.Tensor([1.0, 2.0, 3.0], device="cuda:0")
    assert x.device == "cuda:0"
    import cupy as cp
    assert isinstance(x.values, cp.ndarray)


@needs_cuda
def test_to_device_cpu_to_cuda():
    x = sg.Tensor([1.0, 2.0, 3.0])
    y = x.to_device("cuda:0")
    assert y.device == "cuda:0"
    import cupy as cp
    assert isinstance(y.values, cp.ndarray)
    assert np.allclose(y.values.get(), x.values)


@needs_cuda
def test_to_device_cuda_to_cpu():
    x = sg.Tensor([1.0, 2.0, 3.0], device="cuda:0")
    y = x.to_device("cpu")
    assert y.device == "cpu"
    assert isinstance(y.values, np.ndarray)
    assert np.allclose(y.values, [1.0, 2.0, 3.0])


@needs_cuda
def test_factory_zeros_cuda():
    t = sg.zeros((3, 4), device="cuda:0")
    assert t.device == "cuda:0"
    import cupy as cp
    assert isinstance(t.values, cp.ndarray)
    assert t.values.shape == (3, 4)
    assert cp.all(t.values == 0)


@needs_cuda
def test_factory_ones_cuda():
    t = sg.ones((2, 3), device="cuda:0")
    assert t.device == "cuda:0"
    import cupy as cp
    assert cp.all(t.values == 1)


@needs_cuda
def test_factory_normal_cuda():
    t = sg.normal((100,), device="cuda:0")
    assert t.device == "cuda:0"
    assert t.values.shape == (100,)


@needs_cuda
def test_factory_uniform_cuda():
    t = sg.uniform((100,), low=0, high=1, device="cuda:0")
    assert t.device == "cuda:0"
    import cupy as cp
    assert cp.all(t.values >= 0) and cp.all(t.values <= 1)


@needs_cuda
def test_cuda_add():
    a = sg.Tensor([1.0, 2.0, 3.0], device="cuda:0")
    b = sg.Tensor([4.0, 5.0, 6.0], device="cuda:0")
    c = a + b
    assert c.device == "cuda:0"
    assert np.allclose(c.values.get(), [5.0, 7.0, 9.0])


@needs_cuda
def test_cuda_matmul():
    data1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    data2 = np.array([[5.0, 6.0], [7.0, 8.0]])
    a = sg.Tensor(data1, device="cuda:0")
    b = sg.Tensor(data2, device="cuda:0")
    c = a @ b
    assert c.device == "cuda:0"
    expected = data1 @ data2
    assert np.allclose(c.values.get(), expected)


@needs_cuda
def test_cuda_backward():
    import torch
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    a = sg.Tensor(data, device="cuda:0")
    b = sg.Tensor(data, device="cuda:0")
    c = a @ b
    c.backward()
    at = torch.from_numpy(data).requires_grad_(True)
    bt = torch.from_numpy(data).requires_grad_(True)
    ct = at @ bt
    ct.sum().backward()
    assert np.allclose(a.grad.get(), at.grad.numpy(), atol=1e-5)
    assert np.allclose(b.grad.get(), bt.grad.numpy(), atol=1e-5)


@needs_cuda
def test_cuda_activation_backward():
    import torch
    from simplegrad.functions import relu
    data = np.array([-1.0, 0.5, 2.0])
    a = sg.Tensor(data, device="cuda:0")
    out = relu(a)
    out.backward()
    at = torch.from_numpy(data).requires_grad_(True)
    out_t = torch.relu(at)
    out_t.sum().backward()
    assert np.allclose(a.grad.get(), at.grad.numpy(), atol=1e-5)


@needs_cuda
def test_cuda_device_mismatch_raises():
    a = sg.Tensor([1.0, 2.0], device="cuda:0")
    b = sg.Tensor([3.0, 4.0], device="cpu")
    with pytest.raises(RuntimeError, match="same device"):
        _ = a + b


@needs_cuda
def test_module_to_device():
    model = nn.Linear(4, 2)
    model.to_device("cuda:0")
    import cupy as cp
    for param in model._get_parameters().values():
        assert param.device == "cuda:0"
        assert isinstance(param.values, cp.ndarray)


@needs_cuda
def test_module_to_device_and_back():
    model = nn.Linear(4, 2)
    model.to_device("cuda:0")
    model.to_device("cpu")
    for param in model._get_parameters().values():
        assert param.device == "cpu"
        assert isinstance(param.values, np.ndarray)


@needs_cuda
def test_set_default_device_cuda():
    set_default_device("cuda:0")
    assert get_default_device() == "cuda:0"
    x = sg.Tensor([1.0, 2.0])
    assert x.device == "cuda:0"
    import cupy as cp
    assert isinstance(x.values, cp.ndarray)
