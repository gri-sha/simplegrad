"""Tests for optimizers: SGD, Adam."""

import numpy as np
import simplegrad as sg


def test_sgd_updates_weights():
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

    model = SimpleNet(input_size, hidden_size, output_size)
    optimizer = sg.opt.SGD(model, lr=0.01, momentum=0.9)

    x = sg.Tensor(np.random.randn(batch_size, input_size).astype(np.float64), dtype="float64")
    target = np.random.randn(batch_size, output_size).astype(np.float64)

    output = model(x)
    loss = sg.mse_loss(output, sg.Tensor(target, dtype="float64"))
    loss.backward()

    old_fc1_weights = model.fc1.weight.values.copy()
    optimizer.step()

    assert not np.allclose(old_fc1_weights, model.fc1.weight.values), "Weights should be updated"

    optimizer.zero_grad()

    for name, param in model.parameters().items():
        assert np.allclose(param.grad, 0), f"Gradient for {name} should be zero after zero_grad()"
