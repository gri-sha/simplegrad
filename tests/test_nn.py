"""Tests for neural network layers: Linear, Conv2d, MaxPool2d, Embedding, Sequential."""

import numpy as np
import pytest

pytestmark = pytest.mark.usefixtures("device")
import simplegrad as sg
from .utils import gradcheck, fwdcheck


def _test_conv2d_helper(
    in_channels,
    out_channels,
    kernel_size,
    input_shape,
    stride=1,
    pad_width=0,
    pad_mode="constant",
    pad_value=0,
    use_bias=True,
):
    conv = sg.nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        pad_width=pad_width,
        pad_mode=pad_mode,
        pad_value=pad_value,
        use_bias=use_bias,
        dtype="float64",
    )

    x = sg.Tensor(np.random.randn(*input_shape).astype(np.float64), dtype="float64")

    inputs = [x, conv.weight]
    if use_bias:
        inputs.append(conv.bias)

    gradcheck(lambda: conv(x), inputs)


def test_conv2d_basic():
    """Test Conv2D layer forward and backward pass with padding=1, stride=1"""
    _test_conv2d_helper(
        in_channels=2,
        out_channels=3,
        kernel_size=(3, 3),
        input_shape=(1, 2, 5, 5),
        stride=1,
        pad_width=1,
        use_bias=True,
    )


def test_conv2d_stride_2():
    """Test Conv2D layer with stride=2"""
    _test_conv2d_helper(
        in_channels=2,
        out_channels=3,
        kernel_size=(3, 3),
        input_shape=(1, 2, 6, 6),
        stride=2,
        pad_width=1,
        use_bias=True,
    )


def test_conv2d_no_padding():
    """Test Conv2D layer with no padding"""
    _test_conv2d_helper(
        in_channels=1,
        out_channels=2,
        kernel_size=(3, 3),
        input_shape=(1, 1, 5, 5),
        stride=1,
        pad_width=0,
        use_bias=True,
    )


def test_conv2d_asymmetric_padding():
    """Test Conv2D layer with asymmetric padding (top, bottom, left, right)"""
    _test_conv2d_helper(
        in_channels=2,
        out_channels=3,
        kernel_size=(3, 3),
        input_shape=(1, 2, 5, 5),
        stride=1,
        pad_width=(1, 2, 1, 2),
        use_bias=True,
    )


def test_conv2d_5x5_kernel():
    """Test Conv2D layer with larger 5x5 kernel"""
    _test_conv2d_helper(
        in_channels=1,
        out_channels=2,
        kernel_size=(5, 5),
        input_shape=(1, 1, 8, 8),
        stride=1,
        pad_width=2,
        use_bias=True,
    )


def test_conv2d_stride_2_padding_1():
    """Test Conv2D layer with stride=2 and padding=1"""
    _test_conv2d_helper(
        in_channels=3,
        out_channels=4,
        kernel_size=(3, 3),
        input_shape=(1, 3, 8, 8),
        stride=2,
        pad_width=1,
        use_bias=True,
    )


def test_conv2d_no_bias():
    """Test Conv2D layer without bias"""
    _test_conv2d_helper(
        in_channels=2,
        out_channels=3,
        kernel_size=(3, 3),
        input_shape=(1, 2, 5, 5),
        stride=1,
        pad_width=1,
        use_bias=False,
    )


def test_conv2d_batch():
    """Test Conv2D layer with batch size > 1"""
    _test_conv2d_helper(
        in_channels=2,
        out_channels=3,
        kernel_size=(3, 3),
        input_shape=(2, 2, 5, 5),
        stride=1,
        pad_width=1,
        use_bias=True,
    )


def test_conv2d_channel_mixing():
    # each output channel must sum over input channels through its own kernel row
    # a bug in im2col reshape would mix channels across the wrong axis
    x_data = np.ones((1, 2, 3, 3), dtype=np.float64)
    x_data[:, 1] = 2.0
    x = sg.Tensor(x_data, dtype="float64")

    # weight (out_ch=2, in_ch=2, 1, 1): channel selector kernels
    w_data = np.array([[[[1.0]], [[0.0]]], [[[0.0]], [[1.0]]]], dtype=np.float64)
    w = sg.Tensor(w_data, dtype="float64")

    out = sg.conv2d(x, w, bias=None, pad_width=0)
    assert out.shape == (1, 2, 3, 3)
    expected = np.ones((1, 2, 3, 3), dtype=np.float64)
    expected[:, 1] = 2.0
    fwdcheck(out, expected)


def test_conv2d_asymmetric_padding_output_shape():
    # pad_width=(top,bottom,left,right) mapping must produce the correct spatial dims
    x = sg.Tensor(np.ones((1, 1, 3, 3), dtype=np.float64), dtype="float64")
    w = sg.Tensor(np.ones((1, 1, 1, 1), dtype=np.float64), dtype="float64")
    # padded H = 3+1+2=6, W = 3+1+2=6; with 1×1 kernel and stride=1: H_out=6, W_out=6
    out = sg.conv2d(x, w, bias=None, pad_width=(1, 2, 1, 2))
    assert out.shape == (1, 1, 6, 6), f"expected (1,1,6,6) got {out.shape}"


def test_conv2d_identity_kernel():
    # 3×3 kernel with 1 in center and 0 elsewhere, pad=1 → output == input
    rng = np.random.default_rng(42)
    x_data = rng.standard_normal((1, 1, 5, 5)).astype(np.float64)
    x = sg.Tensor(x_data.copy(), dtype="float64")

    k_data = np.zeros((1, 1, 3, 3), dtype=np.float64)
    k_data[0, 0, 1, 1] = 1.0
    k = sg.Tensor(k_data, dtype="float64")

    out = sg.conv2d(x, k, bias=None, pad_width=1)
    assert out.shape == (1, 1, 5, 5)
    fwdcheck(out, x_data)


def test_norm_output_statistics():
    # core mathematical guarantee: after normalization mean≈0 and std≈1 along normalized dims
    rng = np.random.default_rng(7)
    x_data = rng.standard_normal((4, 8)).astype(np.float64)
    x = sg.Tensor(x_data.copy(), dtype="float64")
    out = sg.norm(x, dims=[-1])
    assert np.allclose(out.values.mean(axis=-1), 0.0, atol=1e-5), "mean should be ≈ 0"
    assert np.allclose(out.values.std(axis=-1, ddof=0), 1.0, atol=1e-4), "std should be ≈ 1"


def test_max_pool2d():
    """Test MaxPool2d layer forward and backward pass"""
    pool = sg.nn.MaxPool2d(kernel_size=2, stride=2)
    x = sg.Tensor(
        np.random.randn(2, 3, 8, 8).astype(np.float64),
        dtype="float64",
    )
    gradcheck(lambda: pool(x), [x])


def test_linear_layer():
    """Test single linear layer forward and backward pass"""
    linear = sg.nn.Linear(4, 3, dtype="float64")
    x = sg.Tensor(np.random.randn(2, 4).astype(np.float64), dtype="float64")

    gradcheck(lambda: linear(x), [x, linear.weight, linear.bias])


def test_sequential_network():
    """Test forward and backward pass through 3-layer neural network"""
    model = sg.nn.Sequential(
        sg.nn.Linear(5, 4, dtype="float64"),
        sg.nn.ReLU(),
        sg.nn.Linear(4, 4, dtype="float64"),
        sg.nn.ReLU(),
        sg.nn.Linear(4, 2, dtype="float64"),
        sg.nn.Softmax(dim=1),
    )

    x = sg.Tensor(np.random.randn(3, 5).astype(np.float64), dtype="float64")
    l0, l2, l4 = model.modules[0], model.modules[2], model.modules[4]

    gradcheck(
        lambda: model(x),
        [x, l0.weight, l0.bias, l2.weight, l2.bias, l4.weight, l4.bias],
    )


def _test_embedding_helper(num_embeddings, embedding_dim, input_shape, dtype="float64"):
    embedding = sg.nn.Embedding(num_embeddings, embedding_dim, dtype=dtype)
    indices = np.random.randint(0, num_embeddings, size=input_shape)
    x = sg.Tensor(indices, dtype="int32" if dtype == "float32" else "int64")

    # Integer indices cannot be perturbed; only the weight gradient is checked.
    eps = 1e-3 if dtype == "float32" else 1e-5
    atol = 1e-3 if dtype == "float32" else 1e-5

    gradcheck(lambda: embedding(x), [embedding.weight], eps=eps, atol=atol)


def test_embedding_1d():
    """Test Embedding layer with 1D input"""
    _test_embedding_helper(num_embeddings=10, embedding_dim=5, input_shape=(8,), dtype="float32")


def test_embedding_2d():
    """Test Embedding layer with 2D input (batch × sequence)"""
    _test_embedding_helper(num_embeddings=20, embedding_dim=8, input_shape=(4, 6), dtype="float64")


def test_embedding_5d():
    """Test Embedding layer with 5D input"""
    _test_embedding_helper(
        num_embeddings=15, embedding_dim=4, input_shape=(2, 3, 2, 3, 4), dtype="float32"
    )
