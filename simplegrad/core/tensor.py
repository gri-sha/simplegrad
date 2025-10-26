import numpy as np
import graphviz
from simplegrad.dtypes import as_array

class Tensor:
    def __init__(
        self, values=None, comp_grad=True, label=None, dtype=None, column=False
    ):
        if values is None:
            values = np.array([])
        self.values = as_array(values, dtype)

        if self.values.ndim == 1:
            self.values = (
                self.values.reshape(-1, 1) if column else self.values.reshape(1, -1)
            )

        self.shape = self.values.shape
        self.label = label
        self.prev = set()
        self.oper = None  # Operation name (no spaces allowed)
        self.comp_grad = comp_grad
        self.grad = None
        self.backward_step = lambda: None

    @staticmethod
    def ones(shape, comp_grad=True, label=None):
        return Tensor(np.ones(shape), comp_grad, label)

    @staticmethod
    def zeros(shape, comp_grad=True, label=None):
        return Tensor(np.zeros(shape), comp_grad, label)

    def __eq__(self, other):
        return id(self) == id(other)

    def __hash__(self):
        return hash(id(self))

    def __getitem__(self, idxs):
        return (
            self.values.__getitem__(idxs),
            self.grad.__getitem__(idxs) if self.grad is not None else None,
        )

    def __iter__(self):
        return self.values.__iter__()

    def __str__(self):
        return f"Tensor '{self.label}', shape: {self.shape}:\nvalues:\n{self.values}\ngrad:\n{self.grad}"

    def flatten(self):
        return self.values.flatten()

    def zero_grad(self):
        if self.comp_grad:
            self.grad = np.zeros(self.shape)
        for t in self.prev:
            t.zero_grad()

    def backward(self):
        topo_order = []
        visited = set()

        def build_topo(tensor):
            if tensor not in visited:
                visited.add(tensor)
                for t in tensor.prev:
                    build_topo(t)
                topo_order.append(tensor)

        build_topo(self)

        self.grad = np.ones(self.values.shape)
        for t in reversed(topo_order):
            t.backward_step()

    def _reduce_broadcasted_dims(self, delta):
        while delta.ndim > self.grad.ndim:
            delta = delta.sum(axis=0)
        for i, (d, g) in enumerate(zip(delta.shape, self.grad.shape)):
            if d != g:
                delta = delta.sum(axis=i, keepdims=True)
        return delta

    def __add__(self, other):
        if isinstance(other, (int, float)):
            out = Tensor(self.values + other)
            out.prev = {self}
            out.oper = f"+({other:.2f})"

            def backward_step():
                if self.comp_grad:
                    self.grad += self._reduce_broadcasted_dims(delta=out.grad)

            out.backward_step = backward_step
            return out

        elif isinstance(other, Tensor):
            if self == other:
                return self * 2

            out = Tensor(self.values + other.values)
            out.prev = {self, other}
            out.oper = "+"

            def backward_step():
                if self.comp_grad:
                    self.grad += self._reduce_broadcasted_dims(delta=out.grad)
                if other.comp_grad:
                    other.grad += other._reduce_broadcasted_dims(delta=out.grad)

            out.backward_step = backward_step
            return out

        else:
            raise ValueError(f"Wrong operand type: {type(other)}")

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            out = Tensor(self.values * other)
            out.prev = {self}
            out.oper = f"*({other:.2f})"

            def backward_step():
                if self.comp_grad:
                    self.grad += out.grad * other

            out.backward_step = backward_step
            return out

        elif isinstance(other, Tensor):
            if self == other:
                return self**2

            out = Tensor(self.values * other.values)
            out.prev = {self, other}
            out.oper = "*"

            def backward_step():
                if self.comp_grad:
                    self.grad += self._reduce_broadcasted_dims(
                        delta=out.grad * other.values
                    )

                if other.comp_grad:
                    other.grad += other._reduce_broadcasted_dims(
                        delta=out.grad * self.values
                    )

            out.backward_step = backward_step
            return out

        else:
            raise ValueError(f"Wrong operand type: {type(other)}")

    def __pow__(self, other):
        if isinstance(other, (float, int)):
            if np.any(self.values < 0) and not float(other).is_integer():
                raise ValueError(
                    f"Invalid: {self.label if self.label else 'Tensor'} ** {other} would be complex."
                )
            out = Tensor(self.values**other)
            out.prev = {self}
            out.oper = f"^{other:.2f}"

            def backward_step():
                if self.comp_grad:
                    self.grad += out.grad * other * (self.values ** (other - 1))

            out.backward_step = backward_step

            return out
        else:
            raise ValueError("Only 'float' or 'int' exponents are supported")

    def __sub__(self, other):
        return self + other * -1

    def __truediv__(self, other):
        return self * other**-1

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return other + self * -1

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        return other * self**-1

    def __neg__(self):
        return self * -1

    # identifier for graphviz
    @property
    def _str_id(self):
        return str(id(self))

    @property
    def _node_signature(self):
        if self.label:
            return f"Tensor '{self.label}' | shape: {self.shape} | comp_grad: {self.comp_grad}"
        else:
            return f"shape: {self.shape} | comp_grad: {self.comp_grad}"

    def _add_graph_vertices(self, graph):
        graph.node(self._str_id, self._node_signature, shape="record")
        if self.oper is not None:
            oper_id = self._str_id + self.oper  # spaces are not allowed in he ids
            graph.node(oper_id, self.oper, shape="oval")
            graph.edge(oper_id, self._str_id)
            for t in self.prev:
                t._add_graph_vertices(graph)

    def _add_graph_edges(self, graph):
        for t in self.prev:
            graph.edge(t._str_id, self._str_id + self.oper)
        for sc in self.prev:
            sc._add_graph_edges(graph)

    def display_graph(self, path=None):
        g = graphviz.Digraph(format="svg", graph_attr={"rankdir": "LR"})
        g.strict = True
        self._add_graph_vertices(graph=g)
        self._add_graph_edges(graph=g)
        if path:
            g.render(filename=path, format="svg", cleanup=True)
        return g
