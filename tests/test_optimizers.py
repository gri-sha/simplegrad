"""Tests for optimizers: SGD, Adam."""

import numpy as np
import pytest
import simplegrad as sg


class TwoLayerNet(sg.Module):
    """Small two-layer network used across multiple optimizer tests."""

    def __init__(self):
        super().__init__()
        self.fc1 = sg.nn.Linear(4, 8)
        self.fc2 = sg.nn.Linear(8, 2)

    def forward(self, x):
        return self.fc2(sg.relu(self.fc1(x)))


def _forward_backward(model, batch_size=3):
    """Run one forward+backward pass and return the loss."""
    x = sg.Tensor(np.random.randn(batch_size, 4).astype(np.float64), dtype="float64")
    target = sg.Tensor(np.random.randn(batch_size, 2).astype(np.float64), dtype="float64")
    loss = sg.mse_loss(model(x), target)
    loss.backward()
    return loss


# SGD — basic behaviour


def test_sgd_updates_weights():
    model = TwoLayerNet()
    optimizer = sg.opt.SGD(model, lr=0.01, momentum=0.9)
    _forward_backward(model)
    old = model.fc1.weight.values.copy()
    optimizer.step()
    assert not np.allclose(old, model.fc1.weight.values), "Weights should be updated after step"


def test_sgd_zero_grad():
    model = TwoLayerNet()
    optimizer = sg.opt.SGD(model, lr=0.01)
    _forward_backward(model)
    optimizer.step()
    optimizer.zero_grad()
    for name, param in model.parameters().items():
        assert np.allclose(param.grad, 0), f"Gradient for {name} should be zero after zero_grad()"


# SGD — parameter groups


def test_sgd_param_groups_independent_lr():
    """Parameters in different groups should be updated with their respective lr."""
    model = TwoLayerNet()
    optimizer = sg.opt.SGD(
        lr=0.1,
        param_groups=[
            {"label": "head", "params": model.fc1, "lr": 0.1},
            {"label": "tail", "params": model.fc2, "lr": 0.0},
        ],
    )
    _forward_backward(model)
    before_fc1 = model.fc1.weight.values.copy()
    before_fc2 = model.fc2.weight.values.copy()
    optimizer.step()
    assert not np.allclose(before_fc1, model.fc1.weight.values), "fc1 should be updated (lr=0.1)"
    assert np.allclose(before_fc2, model.fc2.weight.values), "fc2 should be frozen (lr=0.0)"


def test_sgd_param_groups_auto_labels():
    """Groups without explicit labels receive auto-assigned labels group_0, group_1, ..."""
    model = TwoLayerNet()
    optimizer = sg.opt.SGD(
        lr=0.01,
        param_groups=[
            {"params": model.fc1},
            {"params": model.fc2},
        ],
    )
    labels = [g["label"] for g in optimizer.param_groups]
    assert labels == ["group_0", "group_1"]


def test_sgd_param_groups_different_momentum():
    """Each group should use its own momentum value."""
    model = TwoLayerNet()
    optimizer = sg.opt.SGD(
        lr=0.01,
        momentum=0.0,
        param_groups=[
            {"label": "high_mom", "params": model.fc1, "momentum": 0.9},
            {"label": "no_mom", "params": model.fc2, "momentum": 0.0},
        ],
    )
    assert optimizer.param_groups[0]["momentum"] == 0.9
    assert optimizer.param_groups[1]["momentum"] == 0.0


# SGD — state


def test_sgd_state_structure():
    """state() should include step_count and per-group hyperparams and velocities."""
    model = TwoLayerNet()
    optimizer = sg.opt.SGD(model, lr=0.01, momentum=0.9)
    _forward_backward(model)
    optimizer.step()
    s = optimizer.state()
    assert s["step_count"] == 1
    assert len(s["param_groups"]) == 1
    group = s["param_groups"][0]
    assert group["label"] == "default"
    assert group["lr"] == 0.01
    assert group["momentum"] == 0.9
    assert group["dampening"] == 0
    assert set(group["velocities"].keys()) == set(model.parameters().keys())
    for v in group["velocities"].values():
        assert isinstance(v, np.ndarray)


def test_sgd_state_velocities_are_copies():
    """Velocities in state() should be independent copies, not live references."""
    model = TwoLayerNet()
    optimizer = sg.opt.SGD(model, lr=0.01, momentum=0.9)
    _forward_backward(model)
    optimizer.step()
    s = optimizer.state()
    key = next(iter(s["param_groups"][0]["velocities"]))
    snapshot = s["param_groups"][0]["velocities"][key].copy()
    _forward_backward(model)
    optimizer.step()
    assert np.allclose(
        s["param_groups"][0]["velocities"][key], snapshot
    ), "Velocity snapshot should not change after another step"


# Adam — basic behaviour


def test_adam_updates_weights():
    model = TwoLayerNet()
    optimizer = sg.opt.Adam(model, lr=1e-3)
    _forward_backward(model)
    old = model.fc1.weight.values.copy()
    optimizer.step()
    assert not np.allclose(old, model.fc1.weight.values), "Weights should be updated after step"


def test_adam_zero_grad():
    model = TwoLayerNet()
    optimizer = sg.opt.Adam(model, lr=1e-3)
    _forward_backward(model)
    optimizer.step()
    optimizer.zero_grad()
    for name, param in model.parameters().items():
        assert np.allclose(param.grad, 0), f"Gradient for {name} should be zero after zero_grad()"


# Adam — parameter groups


def test_adam_param_groups_independent_lr():
    model = TwoLayerNet()
    optimizer = sg.opt.Adam(
        lr=1e-3,
        param_groups=[
            {"label": "head", "params": model.fc1, "lr": 1e-3},
            {"label": "tail", "params": model.fc2, "lr": 0.0},
        ],
    )
    _forward_backward(model)
    before_fc1 = model.fc1.weight.values.copy()
    before_fc2 = model.fc2.weight.values.copy()
    optimizer.step()
    assert not np.allclose(before_fc1, model.fc1.weight.values), "fc1 should be updated (lr=1e-3)"
    assert np.allclose(before_fc2, model.fc2.weight.values), "fc2 should be frozen (lr=0.0)"


def test_adam_param_groups_different_betas():
    """Each group should store its own beta_1 / beta_2 overrides."""
    model = TwoLayerNet()
    optimizer = sg.opt.Adam(
        lr=1e-3,
        param_groups=[
            {"label": "default_betas", "params": model.fc1},
            {"label": "custom_betas", "params": model.fc2, "beta_1": 0.8, "beta_2": 0.99},
        ],
    )
    assert optimizer.param_groups[0]["beta_1"] == 0.9
    assert optimizer.param_groups[1]["beta_1"] == 0.8
    assert optimizer.param_groups[1]["beta_2"] == 0.99


# Adam — state


def test_adam_state_structure():
    """state() should include step_count and per-group hyperparams and moment arrays."""
    model = TwoLayerNet()
    optimizer = sg.opt.Adam(model, lr=1e-3, beta_1=0.9, beta_2=0.999, eps=1e-8)
    _forward_backward(model)
    optimizer.step()
    s = optimizer.state()
    assert s["step_count"] == 1
    assert len(s["param_groups"]) == 1
    group = s["param_groups"][0]
    assert group["label"] == "default"
    assert group["lr"] == 1e-3
    assert group["beta_1"] == 0.9
    assert group["beta_2"] == 0.999
    assert group["eps"] == 1e-8
    param_keys = set(model.parameters().keys())
    assert set(group["moments1"].keys()) == param_keys
    assert set(group["moments2"].keys()) == param_keys
    for arr in list(group["moments1"].values()) + list(group["moments2"].values()):
        assert isinstance(arr, np.ndarray)


def test_adam_state_moments_are_copies():
    """Moment arrays in state() should be independent copies, not live references."""
    model = TwoLayerNet()
    optimizer = sg.opt.Adam(model, lr=1e-3)
    _forward_backward(model)
    optimizer.step()
    s = optimizer.state()
    key = next(iter(s["param_groups"][0]["moments1"]))
    snapshot = s["param_groups"][0]["moments1"][key].copy()
    _forward_backward(model)
    optimizer.step()
    assert np.allclose(
        s["param_groups"][0]["moments1"][key], snapshot
    ), "Moment snapshot should not change after another step"


# set_param


def test_set_param_lr_all_groups():
    model = TwoLayerNet()
    optimizer = sg.opt.SGD(
        lr=0.01,
        param_groups=[
            {"label": "a", "params": model.fc1},
            {"label": "b", "params": model.fc2},
        ],
    )
    optimizer.set_param("lr", 0.001)
    assert optimizer.lr == 0.001
    for g in optimizer.param_groups:
        assert g["lr"] == 0.001


def test_set_param_lr_single_group():
    model = TwoLayerNet()
    optimizer = sg.opt.SGD(
        lr=0.01,
        param_groups=[
            {"label": "a", "params": model.fc1},
            {"label": "b", "params": model.fc2},
        ],
    )
    optimizer.set_param("lr", 0.001, group="a")
    assert optimizer.param_groups[0]["lr"] == 0.001
    assert optimizer.param_groups[1]["lr"] == 0.01
    assert optimizer.lr == 0.01, "self.lr should only update when all groups are changed"


def test_set_param_non_lr_hyperparam():
    model = TwoLayerNet()
    optimizer = sg.opt.SGD(model, lr=0.01, momentum=0.9)
    optimizer.set_param("momentum", 0.5)
    assert optimizer.param_groups[0]["momentum"] == 0.5


def test_set_param_unknown_key_raises():
    model = TwoLayerNet()
    optimizer = sg.opt.SGD(model, lr=0.01)
    with pytest.raises(KeyError):
        optimizer.set_param("nonexistent", 1.0)
