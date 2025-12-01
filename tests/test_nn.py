import simplegrad as sg
import numpy as np
import torch
from .utils import compare2tensors


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
    """Helper function to test Conv2D layer with different configurations"""
    conv = sg.nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        pad_width=pad_width,
        pad_mode=pad_mode,
        pad_value=pad_value,
        use_bias=use_bias,
    )

    x = sg.Tensor(np.random.randn(*input_shape).astype(np.float64), dtype="float64")
    y = conv(x)
    print(y.shape)
    y.zero_grad()
    y.backward()

    # PyTorch equivalent
    x_pt = torch.from_numpy(x.values).requires_grad_(True).to(torch.float64)

    # Handle asymmetric padding for PyTorch
    if (
        isinstance(pad_width, tuple)
        and len(pad_width) == 4
        and pad_width != (0, 0, 0, 0)
    ):
        # Asymmetric padding: (top, bottom, left, right)
        x_padded = torch.from_numpy(x.values)
        x_padded = torch.nn.functional.pad(
            x_padded,
            # PyTorch F.pad order is (left, right, top, bottom)
            (pad_width[2], pad_width[3], pad_width[0], pad_width[1]),
            mode=pad_mode,
            value=pad_value,
        )
        x_padded = x_padded.requires_grad_(True).to(torch.float64)
        conv_pt = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            bias=use_bias,
        )
        x_pt_for_conv = x_padded
        is_asymmetric = True
    else:
        # Symmetric padding
        conv_padding = pad_width if isinstance(pad_width, int) else 0
        conv_pt = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=conv_padding,
            bias=use_bias,
        )
        x_pt_for_conv = x_pt
        is_asymmetric = False

    conv_pt.weight.data = (
        torch.from_numpy(conv.weight.values).to(torch.float64).requires_grad_(True)
    )
    if use_bias:
        conv_pt.bias.data = (
            torch.from_numpy(conv.bias.values.flatten())
            .to(torch.float64)
            .requires_grad_(True)
        )

    y_pt = conv_pt(x_pt_for_conv)
    print(y_pt.shape)
    loss_pt = y_pt.sum()
    loss_pt.backward()

    compare2tensors(sg=y, pt=y_pt)

    # For asymmetric padding, extract the inner gradient (without padding) from PyTorch
    if is_asymmetric:
        # Extract gradient for original input from padded gradient
        # pad_width is (top, bottom, left, right)
        top, bottom, left, right = pad_width
        h_end = (
            x_padded.grad.shape[2] - bottom if bottom > 0 else x_padded.grad.shape[2]
        )
        w_end = x_padded.grad.shape[3] - right if right > 0 else x_padded.grad.shape[3]
        x_pt_grad_inner = x_padded.grad[:, :, top:h_end, left:w_end]
        compare2tensors(sg=x.grad, pt=x_pt_grad_inner)
    else:
        compare2tensors(sg=x.grad, pt=x_pt.grad)

    compare2tensors(
        sg=conv.weight.grad,
        pt=conv_pt.weight.grad,
    )
    if use_bias:
        compare2tensors(
            sg=conv.bias.grad,
            pt=conv_pt.bias.grad.unsqueeze(0),
        )


def test_convolution_layer():
    """Test Conv2D layer forward and backward pass with padding=1, stride=1"""
    _test_conv2d_helper(
        in_channels=2,
        out_channels=4,
        kernel_size=(3, 3),
        input_shape=(1, 2, 5, 5),
        stride=1,
        pad_width=1,
        pad_mode="constant",
        pad_value=0,
        use_bias=True,
    )


def test_convolution_layer_stride_2():
    """Test Conv2D layer with stride=2"""
    _test_conv2d_helper(
        in_channels=2,
        out_channels=4,
        kernel_size=(3, 3),
        input_shape=(1, 2, 8, 8),
        stride=2,
        pad_width=1,
        pad_mode="constant",
        pad_value=0,
        use_bias=True,
    )


def test_convolution_layer_no_padding():
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


def test_convolution_layer_asymmetric_padding():
    """Test Conv2D layer with asymmetric padding (top, bottom, left, right)"""
    _test_conv2d_helper(
        in_channels=2,
        out_channels=3,
        kernel_size=(3, 3),
        input_shape=(1, 2, 6, 6),
        stride=1,
        pad_width=(1, 2, 1, 2),  # top=1, bottom=2, left=1, right=2
        pad_mode="constant",
        pad_value=0,
        use_bias=True,
    )


def test_convolution_layer_5x5_kernel():
    """Test Conv2D layer with larger 5x5 kernel"""
    _test_conv2d_helper(
        in_channels=1,
        out_channels=3,
        kernel_size=(5, 5),
        input_shape=(2, 1, 10, 10),
        stride=1,
        pad_width=2,
        pad_mode="constant",
        pad_value=0,
        use_bias=True,
    )


def test_convolution_layer_stride_2_padding_1():
    """Test Conv2D layer with stride=2 and padding=1"""
    _test_conv2d_helper(
        in_channels=3,
        out_channels=8,
        kernel_size=(3, 3),
        input_shape=(2, 3, 16, 16),
        stride=2,
        pad_width=1,
        pad_mode="constant",
        pad_value=0,
        use_bias=True,
    )


def test_convolution_layer_no_bias():
    """Test Conv2D layer without bias"""
    _test_conv2d_helper(
        in_channels=2,
        out_channels=4,
        kernel_size=(3, 3),
        input_shape=(1, 2, 5, 5),
        stride=1,
        pad_width=1,
        use_bias=False,
    )


def test_convolution_layer_batch_processing():
    """Test Conv2D layer with larger batch size"""
    _test_conv2d_helper(
        in_channels=3,
        out_channels=16,
        kernel_size=(3, 3),
        input_shape=(8, 3, 7, 7),
        stride=1,
        pad_width=1,
        pad_mode="constant",
        pad_value=0,
        use_bias=True,
    )


# Test MaxPool2d layer
def test_max_pool2d():
    """Test MaxPool2d layer forward and backward pass"""
    batch_size, channels, height, width = 2, 3, 8, 8
    kernel_size = 2
    stride = 2

    pool = sg.nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    x = sg.Tensor(
        np.random.randn(batch_size, channels, height, width).astype(np.float64),
        dtype="float64",
    )
    y = pool(x)
    y.zero_grad()
    y.backward()

    # PyTorch equivalent
    x_pt = torch.from_numpy(x.values).requires_grad_(True).to(torch.float64)
    pool_pt = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    y_pt = pool_pt(x_pt)
    loss_pt = y_pt.sum()
    loss_pt.backward()

    compare2tensors(sg=y, pt=y_pt)
    compare2tensors(sg=x.grad, pt=x_pt.grad)


# Test Linear layer
def test_linear_layer():
    """Test single linear layer forward and backward pass"""
    in_features, out_features = 4, 3
    linear = sg.nn.Linear(in_features, out_features, dtype="float64")

    x = sg.Tensor(np.random.randn(2, in_features).astype(np.float64), dtype="float64")
    y = linear(x)
    y.zero_grad()
    y.backward()

    # PyTorch equivalent
    x_pt = torch.from_numpy(x.values).requires_grad_(True).to(torch.float64)
    linear_pt = torch.nn.Linear(in_features, out_features, bias=True)
    linear_pt.weight.data = (
        torch.from_numpy(linear.weight.values.T).to(torch.float64).requires_grad_(True)
    )
    linear_pt.bias.data = (
        torch.from_numpy(linear.bias.values.flatten())
        .to(torch.float64)
        .requires_grad_(True)
    )

    y_pt = linear_pt(x_pt)
    loss_pt = y_pt.sum()
    loss_pt.backward()

    compare2tensors(sg=y, pt=y_pt)
    compare2tensors(sg=x.grad, pt=x_pt.grad)
    compare2tensors(sg=linear.weight.grad, pt=linear_pt.weight.grad.T)
    compare2tensors(sg=linear.bias.grad, pt=linear_pt.bias.grad.unsqueeze(0))


def test_neural_network():
    """Test forward and backward pass through 3-layer neural network"""
    input_size, hidden_size, output_size = 5, 4, 2
    batch_size = 3

    # Create SimpleNet model
    model = sg.nn.Sequential(
        sg.nn.Linear(input_size, hidden_size, dtype="float64"),
        sg.nn.ReLU(),
        sg.nn.Linear(hidden_size, hidden_size, dtype="float64"),
        sg.nn.ReLU(),
        sg.nn.Linear(hidden_size, output_size, dtype="float64"),
        sg.nn.Softmax(dim=1),
    )

    # Create input
    x = sg.Tensor(
        np.random.randn(batch_size, input_size).astype(np.float64), dtype="float64"
    )

    # Forward pass
    output = model(x)
    output.zero_grad()
    output.backward()

    # PyTorch equivalent
    x_pt = torch.from_numpy(x.values).requires_grad_(True)

    # Create PyTorch model with same weights
    model_pt = torch.nn.Sequential(
        torch.nn.Linear(input_size, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, output_size),
        torch.nn.Softmax(dim=1),
    )

    # Copy weights
    model_pt[0].weight.data = (
        torch.from_numpy(model.modules[0].weight.values.T)
        .to(torch.float64)
        .requires_grad_(True)
    )
    model_pt[0].bias.data = (
        torch.from_numpy(model.modules[0].bias.values.flatten())
        .to(torch.float64)
        .requires_grad_(True)
    )
    model_pt[2].weight.data = (
        torch.from_numpy(model.modules[2].weight.values.T)
        .to(torch.float64)
        .requires_grad_(True)
    )
    model_pt[2].bias.data = (
        torch.from_numpy(model.modules[2].bias.values.flatten())
        .to(torch.float64)
        .requires_grad_(True)
    )
    model_pt[4].weight.data = (
        torch.from_numpy(model.modules[4].weight.values.T)
        .to(torch.float64)
        .requires_grad_(True)
    )
    model_pt[4].bias.data = (
        torch.from_numpy(model.modules[4].bias.values.flatten())
        .to(torch.float64)
        .requires_grad_(True)
    )

    # Forward pass
    print(model_pt)
    output_pt = model_pt(x_pt)
    loss_pt = output_pt.sum()
    loss_pt.backward()

    # Compare outputs
    compare2tensors(sg=output, pt=output_pt)

    # Compare gradients
    compare2tensors(sg=x.grad, pt=x_pt.grad)
    compare2tensors(sg=model.modules[0].weight.grad, pt=model_pt[0].weight.grad.T)
    compare2tensors(
        sg=model.modules[0].bias.grad, pt=model_pt[0].bias.grad.unsqueeze(0)
    )
    compare2tensors(sg=model.modules[2].weight.grad, pt=model_pt[2].weight.grad.T)
    compare2tensors(
        sg=model.modules[2].bias.grad, pt=model_pt[2].bias.grad.unsqueeze(0)
    )
    compare2tensors(sg=model.modules[4].weight.grad, pt=model_pt[4].weight.grad.T)
    compare2tensors(
        sg=model.modules[4].bias.grad, pt=model_pt[4].bias.grad.unsqueeze(0)
    )


def test_neural_network_with_optimizer():
    """Test neural network training with SGD optimizer"""
    input_size, hidden_size, output_size = 5, 4, 2
    batch_size = 3

    class SimpleNet(sg.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.fc1 = sg.nn.Linear(input_size, hidden_size)
            self.fc2 = sg.nn.Linear(hidden_size, hidden_size)
            self.fc3 = sg.nn.Linear(hidden_size, output_size)

        def forward(self, x):
            x = sg.relu(self.fc1(x))
            x = sg.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    # Create model
    model = SimpleNet(input_size, hidden_size, output_size)
    optimizer = sg.optim.SGD(model, lr=0.01, momentum=0.9)

    # Create input and target
    x = sg.Tensor(
        np.random.randn(batch_size, input_size).astype(np.float64), dtype="float64"
    )
    target = np.random.randn(batch_size, output_size).astype(np.float64)

    # Forward pass
    output = model(x)
    loss = sg.mse_loss(output, sg.Tensor(target, dtype="float64"))

    # Backward pass
    loss.backward()

    # Store old weights
    old_fc1_weights = model.fc1.weight.values.copy()

    # Optimization step
    optimizer.step()

    # Check that weights were updated
    assert not np.allclose(
        old_fc1_weights, model.fc1.weight.values
    ), "Weights should be updated"

    # Zero gradients
    optimizer.zero_grad()

    # Check gradients are zeroed
    for name, param in model.parameters().items():
        assert np.allclose(
            param.grad, 0
        ), f"Gradient for {name} should be zero after zero_grad()"
