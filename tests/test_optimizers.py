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


def _forward_backward(model, batch_size=3, seed=42):
    """Run one forward+backward pass and return the loss."""
    rng = np.random.default_rng(seed)
    x = sg.Tensor(rng.standard_normal((batch_size, 4)).astype(np.float64), dtype="float64")
    target = sg.Tensor(rng.standard_normal((batch_size, 2)).astype(np.float64), dtype="float64")
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
    """Each group uses its own momentum buffer, verified via the update formula."""

    class _Scalar(sg.Module):
        def __init__(self, val):
            super().__init__()
            self.w = sg.Tensor(np.array([float(val)], dtype=np.float64), dtype="float64")

        def forward(self, x):
            return self.w

    m1, m2 = _Scalar(1.0), _Scalar(1.0)
    optimizer = sg.opt.SGD(
        lr=0.1,
        momentum=0.0,
        param_groups=[
            {"label": "high_mom", "params": m1, "momentum": 0.9},
            {"label": "no_mom", "params": m2, "momentum": 0.0},
        ],
    )
    assert optimizer.param_groups[0]["momentum"] == 0.9
    assert optimizer.param_groups[1]["momentum"] == 0.0

    # apply constant gradient g=1.0 for 2 steps, then check hand-computed expected values
    # high_mom: v1=-0.1, w=0.9;  v2=0.9*(-0.1)-0.1=-0.19, w=0.71
    # no_mom:   v1=-0.1, w=0.9;  v2=0.0*(-0.1)-0.1=-0.10, w=0.80
    for _ in range(2):
        m1.w.grad = np.array([1.0], dtype=np.float64)
        m2.w.grad = np.array([1.0], dtype=np.float64)
        optimizer.step()

    np.testing.assert_allclose(m1.w.values, [0.71], atol=1e-10)
    np.testing.assert_allclose(m2.w.values, [0.80], atol=1e-10)


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


# numerical correctness


class _Scalar(sg.Module):
    """Single-parameter model for testing update formulas against hand-computed values."""

    def __init__(self, val: float = 1.0):
        super().__init__()
        self.w = sg.Tensor(np.array([val], dtype=np.float64), dtype="float64")

    def forward(self, x):
        return self.w * x


def test_adam_one_step_formula():
    # theta=1, grad=0.5, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8
    # m1=0.05, v1=0.00025; m_hat=0.5, v_hat=0.25
    # update = 0.01 * 0.5 / (sqrt(0.25) + 1e-8) = 0.01
    # theta_new = 1.0 - 0.01 = 0.99
    model = _Scalar(1.0)
    optimizer = sg.opt.Adam(model, lr=0.01, beta_1=0.9, beta_2=0.999, eps=1e-8)
    model.w.grad = np.array([0.5], dtype=np.float64)
    optimizer.step()
    np.testing.assert_allclose(model.w.values, [0.99], atol=1e-10)


def test_adamw_one_step_formula():
    # same Adam step as above: theta → 0.99
    # weight decay is applied to the post-Adam param: 0.99 -= lr * wd * 0.99 = 0.99 * (1 - 0.001) = 0.98901
    model = _Scalar(1.0)
    optimizer = sg.opt.AdamW(model, lr=0.01, beta_1=0.9, beta_2=0.999, eps=1e-8, weight_decay=0.1)
    model.w.grad = np.array([0.5], dtype=np.float64)
    optimizer.step()
    np.testing.assert_allclose(model.w.values, [0.99 * (1 - 0.01 * 0.1)], atol=1e-8)


def test_adamw_weight_decay_shrinks_params():
    # with zero gradient, weight decay alone must reduce |param| each step
    model = _Scalar(2.0)
    optimizer = sg.opt.AdamW(model, lr=0.1, weight_decay=0.5)
    before = model.w.values.copy()
    model.w.grad = np.zeros(1, dtype=np.float64)
    optimizer.step()
    assert model.w.values[0] < before[0]


# convergence


def test_sgd_converges_on_simple_linear():
    # y = X @ W_true: exact linear relationship, loss must go to near 0
    rng = np.random.default_rng(20)
    W_true = rng.standard_normal((4, 2)).astype(np.float64)
    x_data = rng.standard_normal((16, 4)).astype(np.float64)
    y_data = x_data @ W_true
    x = sg.Tensor(x_data, dtype="float64")
    y = sg.Tensor(y_data, dtype="float64")

    class _Lin(sg.Module):
        def __init__(self):
            super().__init__()
            self.fc = sg.nn.Linear(4, 2, dtype="float64")

        def forward(self, inp):
            return self.fc(inp)

    model = _Lin()
    optimizer = sg.opt.SGD(model, lr=0.1)

    loss0 = float(sg.mse_loss(model(x), y).values.sum())
    for _ in range(200):
        loss = sg.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    assert float(loss.values.sum()) < loss0 * 0.001


# zero_grad edge case


def test_zero_grad_before_any_backward():
    # zero_grad must not raise when called before any backward pass
    model = TwoLayerNet()
    optimizer = sg.opt.SGD(model, lr=0.01)
    optimizer.zero_grad()
    for param in model.parameters().values():
        assert np.all(param.grad == 0)
