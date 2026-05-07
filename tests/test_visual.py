"""Tests for computation graph group tagging and cluster rendering."""

from simplegrad.core import Tensor
from simplegrad.core.compound_ops import graph_group, compound_op


def test_group_is_none_by_default():
    t = Tensor([1.0, 2.0])
    assert t.group is None


def test_graph_group_tags_tensors_created_inside():
    with graph_group("MyOp"):
        t = Tensor([1.0, 2.0])
    name, gid = t.group
    assert name == "MyOp"
    assert isinstance(gid, int)


def test_group_is_none_after_context_exits():
    with graph_group("MyOp"):
        pass
    t = Tensor([1.0])
    assert t.group is None


def test_multiple_calls_produce_different_ids():
    with graph_group("Op"):
        a = Tensor([1.0])
    with graph_group("Op"):
        b = Tensor([1.0])
    assert a.group[1] != b.group[1]


def test_compound_op_output_is_not_grouped():
    @compound_op
    def my_func(x):
        return Tensor(x.values * 2)

    x = Tensor([1.0])
    out = my_func(x)
    assert out.group is None


def test_compound_op_preserves_function_metadata():
    @compound_op
    def my_func(x):
        """My docstring."""
        return x

    assert my_func.__name__ == "my_func"
    assert my_func.__doc__ == "My docstring."


def test_compound_op_different_calls_different_ids():
    @compound_op
    def my_func(x):
        inner = x * 2
        return inner + 0

    x = Tensor([1.0], comp_grad=True)
    a = my_func(x)
    b = my_func(x)
    a_internal = [t for t in _collect(a) if t is not x and t is not a]
    b_internal = [t for t in _collect(b) if t is not x and t is not b]
    assert a_internal[0].group[1] != b_internal[0].group[1]


def _collect(t, seen=None):
    if seen is None:
        seen = set()
    if id(t) in seen:
        return []
    seen.add(id(t))
    result = [t]
    for p in t.prev:
        result.extend(_collect(p, seen))
    return result


def test_softmax_is_single_node():
    from simplegrad.functions.activations import softmax

    x = Tensor([[1.0, 2.0, 3.0]], comp_grad=True)
    out = softmax(x, dim=1)
    # softmax is now a single Function node — no internal intermediates
    assert out.oper == "Softmax"
    assert out.group is None
    internal = [t for t in _collect(out) if t is not x and t is not out]
    assert len(internal) == 0


def test_softmax_graph_contains_node():
    from simplegrad.functions.activations import softmax
    from simplegrad.visual.inline_comp_graph import graph as render_graph

    x = Tensor([[1.0, 2.0, 3.0]], comp_grad=True, label="x")
    out = softmax(x, dim=1)
    g = render_graph(out)
    assert "Softmax" in g.source


def test_relu_graph_has_no_cluster():
    from simplegrad.functions.activations import relu
    from simplegrad.visual.inline_comp_graph import graph as render_graph

    x = Tensor([1.0, -1.0, 2.0], comp_grad=True, label="x")
    out = relu(x)
    g = render_graph(out)
    assert "cluster_" not in g.source


def test_two_softmax_calls_produce_two_nodes():
    from simplegrad.functions.activations import softmax
    from simplegrad.visual.inline_comp_graph import graph as render_graph

    x = Tensor([[1.0, 2.0, 3.0]], comp_grad=True, label="x")
    a = softmax(x, dim=1)
    b = softmax(a, dim=1)
    g = render_graph(b)
    assert g.source.count("label=Softmax") == 2
