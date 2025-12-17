from simplegrad.core import Tensor

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