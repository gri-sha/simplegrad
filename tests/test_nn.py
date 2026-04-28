"""Tests for neural network layers: Linear, Conv2d, MaxPool2d, Embedding, Sequential."""

import numpy as np
import simplegrad as sg
from .utils import gradcheck


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
