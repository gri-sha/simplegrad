import graphviz
from simplegrad.core import Tensor


def _node_signature(tensor: Tensor) -> str:
    if tensor.label:
        return f"Tensor '{tensor.label}' | shape: {tensor.shape} | comp_grad: {tensor.comp_grad}"
    else:
        return f"shape: {tensor.shape} | comp_grad: {tensor.comp_grad}"

def _add_graph_vertices(tensor: Tensor, graph):
    node_attrs = {"shape": "record", "style": "rounded,filled"}
    if tensor.is_leaf:
        node_attrs["fillcolor"] = "lightblue"
        node_attrs["fontname"] = "monospace"
        node_attrs["fontsize"] = "10"
    else:
        node_attrs["fillcolor"] = "white"
        node_attrs["fontname"] = "monospace"
        node_attrs["fontsize"] = "10"
    graph.node(tensor._str_id, _node_signature(tensor), **node_attrs)
    if tensor.oper is not None:
        oper_id = tensor._str_id + tensor.oper
        graph.node(
            oper_id,
            tensor.oper,
            shape="box",
            style="rounded,filled",
            fillcolor="lightgrey",
            fontname="monospace",
            fontsize="10",
        )
        graph.edge(oper_id, tensor._str_id, arrowhead="vee")
        for t in tensor.prev:
            _add_graph_vertices(t, graph)


def _add_graph_edges(tensor, graph):
    for t in tensor.prev:
        graph.edge(t._str_id, tensor._str_id + tensor.oper, arrowhead="vee")
    for sc in tensor.prev:
        _add_graph_edges(sc, graph)


def graph(tensor, path=None):
    g = graphviz.Digraph(
        format="svg",
        graph_attr={
            "rankdir": "LR",
            "nodesep": "0.5",
            "ranksep": "0.7",
            "bgcolor": "white",
        },
    )
    g.strict = True
    _add_graph_vertices(tensor, graph=g)
    _add_graph_edges(tensor, graph=g)
    if path:
        g.render(filename=path, format="svg", cleanup=True)
    return g

def _build_graph_data(tensor: Tensor) -> dict:
    """Build a JSON-serializable graph structure for D3.js visualization."""
    nodes = []
    edges = []
    visited = set()

    def traverse(t: Tensor):
        if t._str_id in visited:
            return
        visited.add(t._str_id)

        # Add tensor node
        nodes.append(
            {
                "id": t._str_id,
                "type": "tensor",
                "label": t.label or "",
                "shape": list(t.shape),
                "comp_grad": t.comp_grad,
                "is_leaf": t.is_leaf,
            }
        )

        # Add operation node if exists
        if t.oper is not None:
            oper_id = t._str_id + "_" + t.oper
            nodes.append(
                {
                    "id": oper_id,
                    "type": "operation",
                    "label": t.oper,
                }
            )
            edges.append(
                {
                    "source": oper_id,
                    "target": t._str_id,
                }
            )

            # Connect parent tensors to operation
            for parent in t.prev:
                edges.append(
                    {
                        "source": parent._str_id,
                        "target": oper_id,
                    }
                )
                traverse(parent)

    traverse(tensor)

    return {
        "nodes": nodes,
        "edges": edges,
    }
