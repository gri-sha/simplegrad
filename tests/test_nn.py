import simplegrad as sg
import numpy as np
import torch
from .utils import compare2tensors


# Test Linear layer
def test_linear_layer():
    """Test single linear layer forward and backward pass"""
    in_features, out_features = 4, 3
    linear = sg.nn.Linear(
        in_features, out_features, init_dtype="float64", init_multiplier=0.1
    )

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


# Test neural network with 3 linear layers


def test_neural_network():
    """Test forward and backward pass through 3-layer neural network"""
    input_size, hidden_size, output_size = 5, 4, 2
    batch_size = 3

    # Create SimpleNet model
    model = sg.nn.Sequential(
        sg.nn.Linear(
            input_size, hidden_size, init_dtype="float64", init_multiplier=0.1
        ),
        sg.nn.ReLU(),
        sg.nn.Linear(
            hidden_size, hidden_size, init_dtype="float64", init_multiplier=0.1
        ),
        sg.nn.ReLU(),
        sg.nn.Linear(
            hidden_size, output_size, init_dtype="float64", init_multiplier=0.1
        ),
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
    compare2tensors(sg=model.modules[0].bias.grad, pt=model_pt[0].bias.grad.unsqueeze(0))
    compare2tensors(sg=model.modules[2].weight.grad, pt=model_pt[2].weight.grad.T)
    compare2tensors(sg=model.modules[2].bias.grad, pt=model_pt[2].bias.grad.unsqueeze(0))
    compare2tensors(sg=model.modules[4].weight.grad, pt=model_pt[4].weight.grad.T)
    compare2tensors(sg=model.modules[4].bias.grad, pt=model_pt[4].bias.grad.unsqueeze(0))


def test_neural_network_with_optimizer():
    """Test neural network training with SGD optimizer"""
    input_size, hidden_size, output_size = 5, 4, 2
    batch_size = 3

    class SimpleNet(sg.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.fc1 = sg.nn.Linear(input_size, hidden_size, init_multiplier=0.1)
            self.fc2 = sg.nn.Linear(hidden_size, hidden_size, init_multiplier=0.1)
            self.fc3 = sg.nn.Linear(hidden_size, output_size, init_multiplier=0.1)

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
