"""Tests for activation functions: relu, tanh, sigmoid, elu, gelu, softmax."""

import numpy as np
import pytest
import simplegrad as sg
from .utils import gradcheck

# input that exercises both positive and negative regions without hitting
# discontinuities (relu boundary, elu boundary) exactly at any element
_DATA = np.array([[0.5, -1.2, 0.3], [-0.8, 2.1, -0.4]], dtype=np.float64)


# elementwise activations (all shape-preserving) — gradients only

def test_elementwise_activations():
    """Gradient check for all shape-preserving activations on a shared input."""
    cases = [
        ("relu",          lambda x: sg.relu(x)),
        ("tanh",          lambda x: sg.tanh(x)),
        ("sigmoid",       lambda x: sg.sigmoid(x)),
        ("elu(alpha=1)",  lambda x: sg.elu(x)),
        ("elu(alpha=0.5)",lambda x: sg.elu(x, alpha=0.5)),
        ("gelu(erf)",     lambda x: sg.gelu(x, mode="erf")),
        ("gelu(tanh)",    lambda x: sg.gelu(x, mode="tanh")),
    ]
    for name, fn in cases:
        x = sg.Tensor(_DATA.copy(), dtype="float64")
        gradcheck(lambda fn=fn, x=x: fn(x), [x])


def test_elu_invalid_mode_raises():
    """gelu with an unknown mode should raise ValueError."""
    x = sg.Tensor(_DATA.copy(), dtype="float64")
    with pytest.raises(ValueError):
        sg.gelu(x, mode="unknown")


# softmax — output shape + gradients, tested separately across dims

@pytest.mark.parametrize("shape,dim", [
    ((4,),       0),
    ((4,),      -1),
    ((3, 4),     0),
    ((3, 4),     1),
    ((3, 4),    -1),
    ((2, 3, 4),  0),
    ((2, 3, 4),  1),
    ((2, 3, 4),  2),
    ((2, 3, 4), -1),
    ((2, 3, 4), -2),
])
def test_softmax_shape_preserved(shape, dim):
    """Softmax must return the same shape as the input."""
    x = sg.Tensor(np.random.randn(*shape).astype(np.float64), dtype="float64")
    out = sg.softmax(x, dim=dim)
    assert out.shape == x.shape, f"shape mismatch for input {shape}, dim={dim}"


@pytest.mark.parametrize("shape,dim", [
    ((4,),       0),
    ((3, 4),     0),
    ((3, 4),     1),
    ((3, 4),    -1),
    ((2, 3, 4),  2),
    ((2, 3, 4), -1),
    ((2, 3, 4), -2),
])
def test_softmax_gradients(shape, dim):
    """Gradient check for softmax across various shapes and dims."""
    x = sg.Tensor(np.random.randn(*shape).astype(np.float64), dtype="float64")
    gradcheck(lambda: sg.softmax(x, dim=dim), [x])


def test_softmax_sums_to_one():
    """Softmax output along the normalised dimension must sum to 1."""
    x = sg.Tensor(np.random.randn(3, 4).astype(np.float64), dtype="float64")
    out = sg.softmax(x, dim=1)
    assert np.allclose(out.values.sum(axis=1), 1.0), "rows should sum to 1"
