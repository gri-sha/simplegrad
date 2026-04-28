"""Tests for lazy execution mode."""

import numpy as np
import simplegrad as sg
from simplegrad.core import Tensor, is_lazy, lazy, mode
from simplegrad.core.autograd import _create_op_result

# mode infrastructure


def test_default_mode_is_eager():
    assert not is_lazy()


def test_lazy_context_manager_activates_lazy_mode():
    with lazy():
        assert is_lazy()
    assert not is_lazy()


def test_lazy_context_manager_restores_on_exception():
    try:
        with lazy():
            raise ValueError("boom")
    except ValueError:
        pass
    assert not is_lazy()


def test_set_mode_lazy():
    mode("lazy")
    assert is_lazy()
    mode("eager")
    assert not is_lazy()


def test_set_mode_invalid_raises():
    try:
        mode("turbo")
        assert False, "Should have raised"
    except ValueError:
        pass
    finally:
        mode("eager")


# Tensor lazy fields and _create_op_result


def test_tensor_has_forward_fn_field():
    t = Tensor([1.0, 2.0])
    assert hasattr(t, "_forward_fn")
    assert t._forward_fn is None


def test_tensor_lazy_init_via_deferred():
    t = Tensor.deferred(lambda: np.array([1.0, 2.0]), shape=(2,))
    assert t.values is None
    assert t.shape == (2,)
    assert t._forward_fn is not None


def test_create_op_result_eager_mode():
    result = _create_op_result(lambda: np.array([3.0, 4.0]), shape=(2,), dtype="float32")
    assert result.values is not None
    np.testing.assert_array_equal(result.values, [3.0, 4.0])


def test_create_op_result_lazy_mode():
    with lazy():
        result = _create_op_result(lambda: np.array([3.0, 4.0]), shape=(2,), dtype="float32")
    assert result.values is None
    assert result.shape == (2,)
    assert result._forward_fn is not None


def test_tensor_str_shows_unrealized_in_lazy_mode():
    with lazy():
        t = Tensor.deferred(lambda: np.array([1.0]), shape=(1,))
    assert "unrealized" in str(t).lower()


# realize() and auto-realize in backward()


def test_realize_executes_pending_fn():
    called = []

    def forward_fn():
        called.append(1)
        return np.array([5.0, 6.0])

    with lazy():
        t = Tensor.deferred(forward_fn, shape=(2,))

    assert t.values is None
    t.realize()
    assert t.values is not None
    np.testing.assert_array_equal(t.values, [5.0, 6.0])
    assert len(called) == 1


def test_realize_returns_self():
    with lazy():
        t = Tensor.deferred(lambda: np.array([1.0]), shape=(1,))
    result = t.realize()
    assert result is t


def test_realize_on_eager_tensor_is_noop():
    t = Tensor([1.0, 2.0])
    t.realize()
    np.testing.assert_array_equal(t.values, [1.0, 2.0])


def test_realize_executes_graph_in_topo_order():
    x = Tensor([1.0, 2.0], comp_grad=True)
    with lazy():
        mid = Tensor.deferred(lambda: x.values * 2, shape=x.shape, dtype=x.dtype)
        mid.prev = {x}
        out = Tensor.deferred(lambda: mid.values + 1, shape=mid.shape, dtype=mid.dtype)
        out.prev = {mid}

    out.realize()
    np.testing.assert_array_almost_equal(out.values, [3.0, 5.0])


# ce_loss backward fix


def test_ce_loss_backward_correct_after_lazy_realize():
    from simplegrad.functions.losses import ce_loss

    x = Tensor([[1.0, 2.0, 3.0]], comp_grad=True)
    y = Tensor([[0.0, 0.0, 1.0]])

    loss_eager = ce_loss(x, y, dim=-1)
    loss_eager.backward()
    eager_grad = x.grad.copy()
    x.grad = None

    with lazy():
        loss_lazy = ce_loss(x, y, dim=-1)
    loss_lazy.backward()
    np.testing.assert_array_almost_equal(x.grad, eager_grad)


# lazy operator overloads


def test_lazy_add_tensor():
    a = Tensor([[1.0, 2.0]], comp_grad=True)
    b = Tensor([[3.0, 4.0]], comp_grad=True)
    with lazy():
        out = a + b
    assert out.values is None
    assert out.shape == (1, 2)
    out.realize()
    np.testing.assert_array_equal(out.values, [[4.0, 6.0]])


def test_lazy_matmul():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], comp_grad=True)
    b = Tensor([[1.0], [0.0]], comp_grad=True)
    with lazy():
        out = a @ b
    assert out.values is None
    assert out.shape == (2, 1)
    out.realize()
    np.testing.assert_array_equal(out.values, [[1.0], [3.0]])


def test_lazy_add_backward():
    a = Tensor([[1.0, 2.0]], comp_grad=True)
    b = Tensor([[3.0, 4.0]], comp_grad=True)
    with lazy():
        out = a + b
    out.backward()
    np.testing.assert_array_equal(a.grad, [[1.0, 1.0]])
    np.testing.assert_array_equal(b.grad, [[1.0, 1.0]])


def test_lazy_transpose():
    a = Tensor([[1.0, 2.0, 3.0]], comp_grad=True)
    with lazy():
        out = a.T
    assert out.shape == (3, 1)
    out.realize()
    np.testing.assert_array_equal(out.values, [[1.0], [2.0], [3.0]])


# lazy activations and math functions


def test_lazy_relu():
    from simplegrad.functions.activations import relu

    x = Tensor([-1.0, 0.0, 2.0], comp_grad=True)
    with lazy():
        out = relu(x)
    assert out.values is None
    assert out.shape == (3,)
    out.realize()
    np.testing.assert_array_equal(out.values, [0.0, 0.0, 2.0])


def test_lazy_exp():
    from simplegrad.functions.math import exp

    x = Tensor([0.0, 1.0], comp_grad=True)
    with lazy():
        out = exp(x)
    assert out.values is None
    out.realize()
    np.testing.assert_array_almost_equal(out.values, [1.0, np.e])


def test_lazy_activation_backward():
    from simplegrad.functions.activations import relu

    x = Tensor([-1.0, 2.0], comp_grad=True)
    with lazy():
        out = relu(x)
    out.backward()
    np.testing.assert_array_equal(x.grad, [0.0, 1.0])


# lazy reduction ops


def test_lazy_sum_no_dim():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], comp_grad=True)
    with lazy():
        out = sg.sum(x)
    assert out.values is None
    assert out.shape == (1, 1)
    out.realize()
    np.testing.assert_array_equal(out.values, [[10.0]])


def test_lazy_sum_with_dim():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], comp_grad=True)
    with lazy():
        out = sg.sum(x, dim=1)
    assert out.shape == (2, 1)
    out.realize()
    np.testing.assert_array_equal(out.values, [[3.0], [7.0]])


def test_lazy_mean():
    x = Tensor([[2.0, 4.0]], comp_grad=True)
    with lazy():
        out = sg.mean(x)
    out.realize()
    np.testing.assert_array_almost_equal(out.values, [[3.0]])


# lazy transform ops


def test_lazy_reshape():
    from simplegrad.functions.tranform import reshape

    x = Tensor([[1.0, 2.0], [3.0, 4.0]], comp_grad=True)
    with lazy():
        out = reshape(x, (4, 1))
    assert out.values is None
    assert out.shape == (4, 1)
    out.realize()
    assert out.values.shape == (4, 1)


def test_lazy_flatten():
    from simplegrad.functions.tranform import flatten

    x = Tensor([[[1.0, 2.0], [3.0, 4.0]]], comp_grad=True)  # shape (1, 2, 2)
    with lazy():
        out = flatten(x, start_dim=1)
    assert out.shape == (1, 4)
    out.realize()
    assert out.values.shape == (1, 4)


# lazy conv2d and pooling


def test_lazy_conv2d():
    from simplegrad.functions.conv import conv2d

    x = Tensor(np.ones((1, 1, 4, 4), dtype="float32"), comp_grad=True)
    w = Tensor(np.ones((1, 1, 2, 2), dtype="float32"), comp_grad=True)
    with lazy():
        out = conv2d(x, w)
    assert out.values is None
    assert out.shape == (1, 1, 3, 3)
    out.realize()
    assert out.values.shape == (1, 1, 3, 3)


def test_lazy_max_pool2d():
    from simplegrad.functions.pooling import max_pool2d

    x = Tensor(np.ones((1, 1, 4, 4), dtype="float32"), comp_grad=True)
    with lazy():
        out = max_pool2d(x, kernel_size=2, stride=2)
    assert out.values is None
    assert out.shape == (1, 1, 2, 2)
    out.realize()
    assert out.values.shape == (1, 1, 2, 2)


# lazy Dropout and Embedding


def test_lazy_dropout_eval_mode():
    from simplegrad.nn.dropout import Dropout

    x = Tensor([[1.0, 2.0, 3.0]], comp_grad=True)
    drop = Dropout(p=0.5)
    drop.set_eval_mode()
    with lazy():
        out = drop.forward(x)
    assert out.values is None
    assert out.shape == (1, 3)
    out.realize()
    np.testing.assert_array_equal(out.values, x.values)


def test_lazy_dropout_train_mode():
    from simplegrad.nn.dropout import Dropout

    x = Tensor(np.ones((2, 4), dtype="float32"), comp_grad=True)
    drop = Dropout(p=0.5)
    drop.set_train_mode()
    with lazy():
        out = drop.forward(x)
    assert out.values is None
    out.backward()
    assert x.grad is not None


def test_lazy_embedding():
    from simplegrad.nn.embedding import Embedding

    emb = Embedding(num_embeddings=10, embedding_dim=4)
    idx = Tensor(np.array([[1, 2], [3, 4]]), dtype="int32")
    with lazy():
        out = emb.forward(idx)
    assert out.values is None
    assert out.shape == (2, 2, 4)
    out.realize()
    assert out.values.shape == (2, 2, 4)


# end-to-end tests


def test_lazy_linear_forward_matches_eager():
    from simplegrad.nn.linear import Linear

    np.random.seed(42)
    layer = Linear(4, 2)
    x = Tensor(np.random.randn(3, 4).astype("float32"), comp_grad=True)

    out_eager = layer.forward(x)
    eager_values = out_eager.values.copy()

    with lazy():
        out_lazy = layer.forward(x)
    assert out_lazy.values is None
    out_lazy.realize()
    np.testing.assert_array_almost_equal(out_lazy.values, eager_values)


def test_lazy_full_forward_and_backward():
    from simplegrad.nn.linear import Linear
    from simplegrad.functions.activations import softmax
    from simplegrad.functions.losses import ce_loss

    np.random.seed(0)
    layer = Linear(4, 3)
    x = Tensor(np.random.randn(2, 4).astype("float32"), comp_grad=True)
    y = Tensor(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype="float32"))

    logits_e = layer.forward(x)
    probs_e = softmax(logits_e, dim=1)
    loss_e = ce_loss(probs_e, y, dim=1)
    loss_e.backward()
    eager_weight_grad = layer.weight.grad.copy()
    for p in layer.parameters().values():
        p.grad = None

    with lazy():
        logits_l = layer.forward(x)
        probs_l = softmax(logits_l, dim=1)
        loss_l = ce_loss(probs_l, y, dim=1)

    loss_l.backward()
    np.testing.assert_array_almost_equal(layer.weight.grad, eager_weight_grad, decimal=5)


def test_lazy_dead_code_elimination():
    call_log = []

    def expensive_fn():
        call_log.append("called")
        return np.array([99.0])

    x = Tensor([1.0, 2.0], comp_grad=True)

    with lazy():
        used = x + Tensor([1.0, 1.0])
        unused = Tensor.deferred(expensive_fn, shape=(1,))  # not in used's graph

    used.realize()
    assert "called" not in call_log, "Unreachable tensor should not be realized"


def test_lazy_context_does_not_affect_outer_scope():
    x = Tensor([1.0])
    with lazy():
        y = x + Tensor([2.0])
    z = x + Tensor([3.0])
    assert z.values is not None
    np.testing.assert_array_equal(z.values, [4.0])
