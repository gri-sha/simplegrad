"""Tests for learning rate schedulers."""

import numpy as np
import pytest
import simplegrad as sg


def _make_optimizer(lr=0.1):
    """Create a simple optimizer for scheduler tests."""
    model = sg.nn.Linear(4, 2)
    return sg.opt.SGD(model, lr=lr)


# ExponentialLR — construction cases


def test_exponential_lr_case1_start_end_total():
    """Case 1: start_lr, end_lr, total_steps -> gamma computed."""
    opt = _make_optimizer(lr=0.1)
    scheduler = sg.sch.ExponentialLR(opt, start_lr=0.1, end_lr=0.01, total_steps=10)
    assert scheduler.start_lr == 0.1
    assert scheduler.end_lr == 0.01
    assert scheduler.total_steps == 10
    assert np.isclose(scheduler.gamma, (0.01 / 0.1) ** (1.0 / 10))


def test_exponential_lr_case2_start_end_gamma():
    """Case 2: start_lr, end_lr, gamma -> total_steps computed."""
    opt = _make_optimizer(lr=0.1)
    scheduler = sg.sch.ExponentialLR(opt, start_lr=0.1, end_lr=0.01, gamma=0.9)
    expected_steps = int(round(np.log(0.01 / 0.1) / np.log(0.9)))
    assert scheduler.start_lr == 0.1
    assert scheduler.end_lr == 0.01
    assert scheduler.gamma == 0.9
    assert scheduler.total_steps == expected_steps


def test_exponential_lr_case3_start_total_gamma():
    """Case 3: start_lr, total_steps, gamma -> end_lr computed."""
    opt = _make_optimizer(lr=0.1)
    scheduler = sg.sch.ExponentialLR(opt, start_lr=0.1, total_steps=10, gamma=0.9)
    assert scheduler.start_lr == 0.1
    assert scheduler.total_steps == 10
    assert scheduler.gamma == 0.9
    assert np.isclose(scheduler.end_lr, 0.1 * (0.9**10))


def test_exponential_lr_case4_end_total_gamma():
    """Case 4: end_lr, total_steps, gamma -> start_lr computed."""
    opt = _make_optimizer(lr=0.1)
    scheduler = sg.sch.ExponentialLR(opt, end_lr=0.01, total_steps=10, gamma=0.9)
    assert scheduler.end_lr == 0.01
    assert scheduler.total_steps == 10
    assert scheduler.gamma == 0.9
    assert np.isclose(scheduler.start_lr, 0.01 / (0.9**10))


def test_exponential_lr_case5_start_gamma_infinite():
    """Case 5: start_lr, gamma -> infinite total_steps."""
    opt = _make_optimizer(lr=0.1)
    scheduler = sg.sch.ExponentialLR(opt, start_lr=0.1, gamma=0.9)
    assert scheduler.start_lr == 0.1
    assert scheduler.gamma == 0.9
    assert scheduler.total_steps == float("inf")


def test_exponential_lr_all_four_raises():
    """Providing all four parameters should raise ValueError."""
    opt = _make_optimizer()
    with pytest.raises(ValueError):
        sg.sch.ExponentialLR(opt, start_lr=0.1, end_lr=0.01, total_steps=10, gamma=0.9)


def test_exponential_lr_invalid_combination_raises():
    """Providing an invalid combination should raise ValueError."""
    opt = _make_optimizer()
    with pytest.raises(ValueError):
        sg.sch.ExponentialLR(opt, end_lr=0.01)


# ExponentialLR — step behaviour


def test_exponential_lr_step_updates_lr():
    """Step should decay the learning rate by gamma each time."""
    opt = _make_optimizer(lr=0.1)
    scheduler = sg.sch.ExponentialLR(opt, start_lr=0.1, gamma=0.5, total_steps=5)

    scheduler.step()
    assert np.isclose(opt.lr, 0.1)

    scheduler.step()
    assert np.isclose(opt.lr, 0.05)

    scheduler.step()
    assert np.isclose(opt.lr, 0.025)


def test_exponential_lr_step_stops_after_total_steps():
    """After total_steps, the learning rate should stop changing."""
    opt = _make_optimizer(lr=0.1)
    scheduler = sg.sch.ExponentialLR(opt, start_lr=0.1, gamma=0.5, total_steps=3)

    scheduler.step()
    scheduler.step()
    scheduler.step()
    scheduler.step()

    assert np.isclose(opt.lr, 0.025)


def test_exponential_lr_integration_with_optimizer():
    """Scheduler should correctly control an optimizer during training."""
    model = sg.nn.Linear(4, 2)
    optimizer = sg.opt.SGD(model, lr=0.1)
    scheduler = sg.sch.ExponentialLR(optimizer, start_lr=0.1, gamma=0.5, total_steps=5)

    x = sg.Tensor(np.random.randn(2, 4).astype(np.float64), dtype="float64")
    target = sg.Tensor(np.random.randn(2, 2).astype(np.float64), dtype="float64")

    for _ in range(4):
        loss = sg.mse_loss(model(x), target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    assert np.isclose(optimizer.lr, 0.1 * (0.5**3))


# CosineAnnealingLR — construction


def test_cosine_annealing_lr_default_lr_max():
    """If lr_max is not provided, it should default to the optimizer's current lr."""
    opt = _make_optimizer(lr=0.1)
    scheduler = sg.sch.CosineAnnealingLR(opt, T_0=10)
    assert scheduler.lr_max == 0.1
    assert scheduler.lr_min == 0.0
    assert scheduler.T_0 == 10
    assert scheduler.T_mult == 1


def test_cosine_annealing_lr_sets_lr_max():
    """If lr_max is provided, the optimizer's lr should be set to it immediately."""
    opt = _make_optimizer(lr=0.1)
    scheduler = sg.sch.CosineAnnealingLR(opt, T_0=10, lr_max=0.5)
    assert scheduler.lr_max == 0.5
    assert np.isclose(opt.lr, 0.5)


def test_cosine_annealing_lr_custom_params():
    """Custom lr_min and T_mult should be stored correctly."""
    opt = _make_optimizer(lr=0.1)
    scheduler = sg.sch.CosineAnnealingLR(opt, T_0=5, T_mult=2, lr_min=0.01, lr_max=0.2)
    assert scheduler.T_0 == 5
    assert scheduler.T_mult == 2
    assert scheduler.lr_min == 0.01
    assert scheduler.lr_max == 0.2


# CosineAnnealingLR — step behaviour


def test_cosine_annealing_lr_step_shape():
    """LR should start at lr_max, decrease toward lr_min, then restart."""
    opt = _make_optimizer(lr=0.1)
    scheduler = sg.sch.CosineAnnealingLR(opt, T_0=4, lr_max=0.1, lr_min=0.0)

    # Step 0: t_cur=0, cos(0)=1 -> lr = lr_max
    scheduler.step()
    assert np.isclose(opt.lr, 0.1)

    # Step 1: t_cur=1, cos(pi/4)
    scheduler.step()
    expected = 0.5 * 0.1 * (1 + np.cos(np.pi / 4))
    assert np.isclose(opt.lr, expected)
    assert opt.lr < 0.1
    assert opt.lr > 0.0

    # Step 2: t_cur=2, cos(pi/2)=0 -> lr = mid
    scheduler.step()
    assert np.isclose(opt.lr, 0.05)

    # Step 3: t_cur=3, cos(3pi/4)
    scheduler.step()
    expected = 0.5 * 0.1 * (1 + np.cos(3 * np.pi / 4))
    assert np.isclose(opt.lr, expected)
    assert opt.lr < 0.05
    assert opt.lr > 0.0

    # Step 4: restarted, t_cur=0 -> lr = lr_max again
    scheduler.step()
    assert np.isclose(opt.lr, 0.1)


def test_cosine_annealing_lr_warm_restart_T_mult():
    """With T_mult > 1, the period length should grow after each restart."""
    opt = _make_optimizer(lr=0.1)
    scheduler = sg.sch.CosineAnnealingLR(opt, T_0=2, T_mult=2, lr_max=0.1, lr_min=0.0)

    # First period: T_i=2, t_cur = 0, 1
    scheduler.step()  # t=0, lr=0.1
    assert np.isclose(opt.lr, 0.1)
    scheduler.step()  # t=1, cos(pi/2)=0, lr=0.05, then restart
    assert np.isclose(opt.lr, 0.05)
    assert scheduler.T_i == 4  # doubled after restart

    # Second period: T_i=4, t_cur = 0, 1, 2, 3
    scheduler.step()  # t=0, lr=0.1
    assert np.isclose(opt.lr, 0.1)
    scheduler.step()  # t=1, cos(pi/4)
    assert np.isclose(opt.lr, 0.5 * 0.1 * (1 + np.cos(np.pi / 4)))
    scheduler.step()  # t=2, cos(pi/2)=0, lr=0.05
    assert np.isclose(opt.lr, 0.05)
    scheduler.step()  # t=3, cos(3pi/4)
    assert np.isclose(opt.lr, 0.5 * 0.1 * (1 + np.cos(3 * np.pi / 4)))
    scheduler.step()  # restarted, t=0, lr=0.1
    assert np.isclose(opt.lr, 0.1)
    assert scheduler.T_i == 8  # doubled again


def test_cosine_annealing_lr_integration_with_optimizer():
    """Scheduler should correctly control an optimizer during training."""
    model = sg.nn.Linear(4, 2)
    optimizer = sg.opt.SGD(model, lr=0.1)
    scheduler = sg.sch.CosineAnnealingLR(optimizer, T_0=3, lr_max=0.1, lr_min=0.0)

    x = sg.Tensor(np.random.randn(2, 4).astype(np.float64), dtype="float64")
    target = sg.Tensor(np.random.randn(2, 2).astype(np.float64), dtype="float64")

    lrs = []
    for _ in range(7):
        loss = sg.mse_loss(model(x), target)
        loss.backward()
        optimizer.step()
        lrs.append(optimizer.lr)
        scheduler.step()
        optimizer.zero_grad()

    # lrs[0] is initial lr before any scheduler step
    assert np.isclose(lrs[0], 0.1)
    # lrs[1] is after first scheduler step (t_cur=0)
    assert np.isclose(lrs[1], 0.1)
    # lrs[2] is after second scheduler step (t_cur=1)
    assert np.isclose(lrs[2], 0.5 * 0.1 * (1 + np.cos(np.pi / 3)))
    # lrs[3] is after third scheduler step (t_cur=2)
    assert np.isclose(lrs[3], 0.5 * 0.1 * (1 + np.cos(2 * np.pi / 3)))
    # lrs[4] is after restart (t_cur=0 again)
    assert np.isclose(lrs[4], 0.1)


# ReduceLROnPlateauLR — construction


def test_reduce_lr_on_plateau_constructor():
    """All constructor parameters should be stored correctly."""
    opt = _make_optimizer(lr=0.1)
    scheduler = sg.sch.ReduceLROnPlateauLR(
        opt,
        factor=0.5,
        patience=5,
        min_lr=1e-5,
        threshold=1e-3,
        threshold_mode="abs",
        cooldown=2,
        maximize_metric=True,
        verbose=True,
    )
    assert scheduler.factor == 0.5
    assert scheduler.patience == 5
    assert scheduler.min_lr == 1e-5
    assert scheduler.threshold == 1e-3
    assert scheduler.threshold_mode == "abs"
    assert scheduler.cooldown == 2
    assert scheduler.maximize_metric is True
    assert scheduler.verbose is True


def test_reduce_lr_on_plateau_invalid_factor():
    """factor >= 1.0 should raise ValueError."""
    opt = _make_optimizer()
    with pytest.raises(ValueError):
        sg.sch.ReduceLROnPlateauLR(opt, factor=1.0)


def test_reduce_lr_on_plateau_invalid_patience():
    """Negative patience should raise ValueError."""
    opt = _make_optimizer()
    with pytest.raises(ValueError):
        sg.sch.ReduceLROnPlateauLR(opt, factor=0.5, patience=-1)


def test_reduce_lr_on_plateau_invalid_threshold():
    """Non-positive threshold should raise ValueError."""
    opt = _make_optimizer()
    with pytest.raises(ValueError):
        sg.sch.ReduceLROnPlateauLR(opt, factor=0.5, threshold=0)


def test_reduce_lr_on_plateau_invalid_threshold_mode():
    """Invalid threshold_mode should raise ValueError."""
    opt = _make_optimizer()
    with pytest.raises(ValueError):
        sg.sch.ReduceLROnPlateauLR(opt, factor=0.5, threshold_mode="invalid")


# ReduceLROnPlateauLR — step behaviour


def test_reduce_lr_on_plateau_reduces_lr_on_plateau():
    """LR should be reduced after patience steps without improvement."""
    opt = _make_optimizer(lr=0.1)
    scheduler = sg.sch.ReduceLROnPlateauLR(opt, factor=0.5, patience=2, verbose=False)

    scheduler.step(10.0)  # best = 10.0
    assert np.isclose(opt.lr, 0.1)
    assert scheduler.num_bad_steps == 0

    scheduler.step(9.0)  # improvement
    assert np.isclose(opt.lr, 0.1)
    assert scheduler.num_bad_steps == 0

    scheduler.step(8.5)  # improvement (still going down)
    assert np.isclose(opt.lr, 0.1)

    scheduler.step(8.5)  # no improvement (equal to best, not better)
    assert scheduler.num_bad_steps == 1

    scheduler.step(8.5)  # no improvement again -> trigger reduction
    assert np.isclose(opt.lr, 0.05)  # 0.1 * 0.5


def test_reduce_lr_on_plateau_respects_min_lr():
    """LR should not drop below min_lr."""
    opt = _make_optimizer(lr=0.1)
    scheduler = sg.sch.ReduceLROnPlateauLR(opt, factor=0.5, patience=1, min_lr=0.03, verbose=True)

    scheduler.step(1.0)  # best = 1.0
    scheduler.step(1.0)  # plateau (no improvement) -> reduce to 0.05
    assert np.isclose(opt.lr, 0.05)

    scheduler.step(0.5)  # improvement resets bad counter, best = 0.5
    scheduler.step(0.5)  # plateau -> reduce to max(0.025, 0.03) = 0.03
    assert np.isclose(opt.lr, 0.03)  # clamped to min_lr


def test_reduce_lr_on_plateau_cooldown():
    """After a reduction, cooldown period should prevent immediate monitoring."""
    opt = _make_optimizer(lr=0.1)
    scheduler = sg.sch.ReduceLROnPlateauLR(opt, factor=0.5, patience=1, cooldown=2, verbose=False)

    scheduler.step(1.0)  # best = 1.0
    scheduler.step(1.0)  # plateau -> reduce to 0.05, cooldown_remaining = 2
    assert opt.lr == 0.05
    assert scheduler.cooldown_remaining == 2

    scheduler.step(0.8)  # in cooldown, not monitored
    assert scheduler.cooldown_remaining == 1
    assert scheduler.num_bad_steps == 0

    scheduler.step(0.8)  # cooldown ends
    assert scheduler.cooldown_remaining == 0


def test_reduce_lr_on_plateau_maximize_metric():
    """With maximize_metric=True, higher values are better."""
    opt = _make_optimizer(lr=0.1)
    scheduler = sg.sch.ReduceLROnPlateauLR(
        opt, factor=0.5, patience=2, maximize_metric=True, verbose=False
    )

    scheduler.step(0.5)  # best = 0.5
    scheduler.step(0.6)  # improvement
    assert scheduler.num_bad_steps == 0
    scheduler.step(0.55)  # no improvement
    scheduler.step(0.55)  # no improvement -> reduce
    assert np.isclose(opt.lr, 0.05)


def test_reduce_lr_on_plateau_threshold_rel():
    """With threshold_mode='rel', threshold is relative to best value."""
    opt = _make_optimizer(lr=0.1)
    scheduler = sg.sch.ReduceLROnPlateauLR(
        opt,
        factor=0.5,
        patience=1,
        threshold=0.1,
        threshold_mode="rel",
        maximize_metric=False,
        verbose=False,
    )

    scheduler.step(10.0)  # best = 10.0, improvement requires metric < 10.0 * (1 - 0.1) = 9.0
    scheduler.step(10.0)  # 10.0 >= 9.0, not improvement -> reduce to 0.05, best = 10.0
    assert scheduler.num_bad_steps == 0
    scheduler.step(5.0)  # 5.0 < 9.0, improvement -> bad = 0, best = 5.0
    scheduler.step(4.4)  # 4.4 < 5.0 * 0.9 = 4.5, improvement -> bad = 0
    assert np.isclose(opt.lr, 0.05)


def test_reduce_lr_on_plateau_threshold_abs():
    """With threshold_mode='abs', threshold is an absolute delta from best."""
    opt = _make_optimizer(lr=0.1)
    scheduler = sg.sch.ReduceLROnPlateauLR(
        opt,
        factor=0.5,
        patience=1,
        threshold=0.5,
        threshold_mode="abs",
        maximize_metric=False,
        verbose=False,
    )

    scheduler.step(10.0)  # best = 10.0, improvement requires metric < 10.0 - 0.5 = 9.5
    scheduler.step(10.0)  # 10.0 >= 9.5, not improvement -> reduce to 0.05, best = 10.0
    assert scheduler.num_bad_steps == 0
    scheduler.step(9.0)  # 9.0 < 9.5, improvement -> bad = 0, best = 9.0
    scheduler.step(8.4)  # 8.4 < 9.0 - 0.5 = 8.5, improvement -> bad = 0
    assert np.isclose(opt.lr, 0.05)


def test_reduce_lr_on_plateau_verbose(capsys):
    """When verbose=True, a message should be printed on reduction."""
    opt = _make_optimizer(lr=0.1)
    scheduler = sg.sch.ReduceLROnPlateauLR(opt, factor=0.5, patience=1, verbose=True)

    scheduler.step(10.0)
    scheduler.step(9.0)  # improvement
    scheduler.step(8.5)  # no improvement
    scheduler.step(8.5)  # no improvement -> reduce
    captured = capsys.readouterr()
    assert "reducing learning rate" in captured.out.lower()


def test_reduce_lr_on_plateau_no_reduction_if_factor_too_small():
    """If the decay is smaller than machine epsilon, no reduction occurs."""
    opt = _make_optimizer(lr=0.1)
    scheduler = sg.sch.ReduceLROnPlateauLR(opt, factor=0.999999, patience=1, verbose=False)

    scheduler.step(10.0)
    scheduler.step(9.0)
    scheduler.step(8.5)
    scheduler.step(8.5)  # would reduce to ~0.0999999
    # The difference is < 1e-12 so no reduction should occur
    assert np.isclose(opt.lr, 0.1)


def test_reduce_lr_on_plateau_integration_with_optimizer():
    """Scheduler should correctly control an optimizer during training."""
    model = sg.nn.Linear(4, 2)
    optimizer = sg.opt.SGD(model, lr=0.1)
    scheduler = sg.sch.ReduceLROnPlateauLR(optimizer, factor=0.5, patience=2, verbose=False)

    x = sg.Tensor(np.random.randn(2, 4).astype(np.float64), dtype="float64")
    target = sg.Tensor(np.random.randn(2, 2).astype(np.float64), dtype="float64")

    # loss improves for 3 steps then plateaus — 2 consecutive flat steps trigger reduction
    losses = [0.5, 0.4, 0.3, 0.3, 0.3, 0.3]

    for i, target_loss in enumerate(losses):
        optimizer.zero_grad()
        output = model(x)
        fake_loss = sg.Tensor(np.array([[target_loss]] * 2, dtype=np.float64), dtype="float64")
        loss = sg.mse_loss(output, fake_loss)
        loss.backward()
        optimizer.step()
        scheduler.step(target_loss)
        optimizer.zero_grad()

    # steps 4-5 (both 0.3) are bad steps; bad=2 >= patience=2 -> reduce to 0.05
    assert np.isclose(optimizer.lr, 0.05)
