"""Graphviz-based computation graph visualization for inline (notebook) use."""

import graphviz
from ..core import Tensor


def _node_signature(tensor: Tensor) -> str:
    if tensor.label:
        return f"Tensor '{tensor.label}' | shape: {tensor.shape} | comp_grad: {tensor.comp_grad}"
    else:
        return f"shape: {tensor.shape} | comp_grad: {tensor.comp_grad}"


_TENSOR_BASE_ATTRS = {
    "shape": "record",
    "style": "rounded,filled",
    "fontname": "monospace",
    "fontsize": "10",
    "fontcolor": "#151718",
}

_OP_NODE_ATTRS = {
    "shape": "box",
    "style": "rounded,filled",
    "fillcolor": "#e8e8e8",
    "fontcolor": "#151718",
    "fontname": "monospace",
    "fontsize": "10",
}


def _collect_nodes(tensor: Tensor, visited: set = None) -> list:
    """Traverse the computation graph and return all tensors reachable from ``tensor``.

    Args:
        tensor: Root tensor to start traversal from.
        visited: Set of already-visited tensor ids (used internally for recursion).
    """
    if visited is None:
        visited = set()
    if id(tensor) in visited:
        return []
    visited.add(id(tensor))
    result = [tensor]
    for t in tensor.prev:
        result.extend(_collect_nodes(t, visited))
    return result


def _render_tensor_node(tensor: Tensor, target) -> None:
    """Add the tensor node and its operation node to target, with an op→tensor edge.

    Args:
        tensor: The tensor to render.
        target: A ``graphviz.Digraph`` or subgraph context to add nodes into.
    """
    attrs = {**_TENSOR_BASE_ATTRS, "fillcolor": "#34b87e" if tensor.is_leaf else "#feba14"}
    target.node(tensor._str_id, _node_signature(tensor), **attrs)
    if tensor.oper is not None:
        oper_id = tensor._str_id + tensor.oper
        target.node(oper_id, tensor.oper, **_OP_NODE_ATTRS)
        target.edge(oper_id, tensor._str_id, arrowhead="vee")


def _render_tensor_body(tensor: Tensor, target) -> None:
    """Add only the tensor node to target (no op node, no edge).

    Args:
        tensor: The tensor whose node to render.
        target: A ``graphviz.Digraph`` or subgraph context.
    """
    attrs = {**_TENSOR_BASE_ATTRS, "fillcolor": "#34b87e" if tensor.is_leaf else "#feba14"}
    target.node(tensor._str_id, _node_signature(tensor), **attrs)


def _render_op_node(tensor: Tensor, target) -> None:
    """Add only the operation node for tensor to target (no tensor node, no edge).

    Args:
        tensor: The tensor whose operation node to render.
        target: A ``graphviz.Digraph`` or subgraph context.
    """
    if tensor.oper is None:
        return
    oper_id = tensor._str_id + tensor.oper
    target.node(oper_id, tensor.oper, **_OP_NODE_ATTRS)


def _add_graph_edges(tensor: Tensor, graph, visited: set = None) -> None:
    """Add directed edges from input tensors to operation nodes throughout the graph.

    Edges are always added to the root graph so they render correctly across
    cluster boundaries.

    Args:
        tensor: Current tensor in the traversal.
        graph: The root ``graphviz.Digraph`` to add edges into.
        visited: Set of already-visited tensor ids.
    """
    if visited is None:
        visited = set()
    if id(tensor) in visited:
        return
    visited.add(id(tensor))
    for t in tensor.prev:
        graph.edge(t._str_id, tensor._str_id + tensor.oper, arrowhead="vee")
    for t in tensor.prev:
        _add_graph_edges(t, graph, visited)


def graph(tensor: Tensor, path: str | None = None) -> graphviz.Digraph:
    """Render the computation graph of a tensor as an SVG diagram.

    Functions decorated with ``@compound_op`` are enclosed in a labelled
    black-border rectangle. The internal intermediate tensors and the final
    operation node of the compound op are inside the rectangle; the output
    tensor node itself sits outside, connecting to the cluster via an edge.

    Node colors:
        - Green (#34b87e): leaf tensors (inputs / parameters)
        - Yellow (#feba14): intermediate tensors
        - Light grey (#e8e8e8): operation nodes

    Args:
        tensor: The output tensor whose computation graph to visualize.
        path: If provided, save the SVG to this file path (without extension).

    Returns:
        A ``graphviz.Digraph`` object. Displays inline in Jupyter notebooks.
    """
    try:
        graphviz.version()
    except graphviz.backend.ExecutableNotFound:
        raise RuntimeError(
            "Graphviz system binaries not found. For installation check: https://graphviz.org/download/"
        ) from None

    g = graphviz.Digraph(
        format="svg",
        graph_attr={
            "rankdir": "LR",
            "nodesep": "0.5",
            "ranksep": "0.7",
            "bgcolor": "#f5f5f5",
        },
    )
    g.strict = True

    all_tensors = _collect_nodes(tensor)

    # clusters[gid] = (gname, regular_tensors, compound_output_tensors)
    clusters: dict[int, tuple[str, list, list]] = {}
    ungrouped: list = []
    compound_outputs: list = []

    for t in all_tensors:
        if t.group is not None:
            gname, gid = t.group
            if gid not in clusters:
                clusters[gid] = (gname, [], [])
            clusters[gid][1].append(t)
        elif t._op_group is not None:
            gname, gid = t._op_group
            if gid not in clusters:
                clusters[gid] = (gname, [], [])
            clusters[gid][2].append(t)
            compound_outputs.append(t)
        else:
            ungrouped.append(t)

    for gid, (gname, tensors, outputs) in clusters.items():
        with g.subgraph(name=f"cluster_{gid}") as c:
            c.attr(
                label=gname,
                labelloc="t",
                labeljust="l",
                color="#151718",
                style="rounded",
                fontname="monospace",
                fontsize="10",
            )
            for t in tensors:
                _render_tensor_node(t, c)
            for t in outputs:
                _render_op_node(t, c)

    for t in ungrouped:
        _render_tensor_node(t, g)

    for t in compound_outputs:
        _render_tensor_body(t, g)
        if t.oper is not None:
            g.edge(t._str_id + t.oper, t._str_id, arrowhead="vee")

    _add_graph_edges(tensor, g)

    if path:
        g.render(filename=path, format="svg", cleanup=True)
    return g
