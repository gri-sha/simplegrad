from __future__ import annotations  # make all annotations lazy
import numpy as np
from contextlib import contextmanager
from simplegrad.dtypes import as_array, convert_to_dtype

_COMP_GRAD = True


@contextmanager
def no_grad():
    global _COMP_GRAD
    prev_comp_grad = _COMP_GRAD
    _COMP_GRAD = False
    try:
        # print(_COMP_GRAD)
        yield
    finally:
        _COMP_GRAD = prev_comp_grad


def _should_compute_grad(*inputs) -> bool:
    """
    Determine if gradient computation is needed for an operation.

    Args:
        *inputs: Any number of inputs (Tensors, scalars, etc.)

    Returns:
        False if:
            - Global _COMP_GRAD is False (inside no_grad() context)
            - No input Tensor has comp_grad=True
        True if:
            - At least one input Tensor has comp_grad=True
    """
    # Check global flag first
    if not _COMP_GRAD:
        return False

    # Check if any input tensor requires gradient
    for inp in inputs:
        if isinstance(inp, Tensor) and inp.comp_grad:
            return True

    return False


class Tensor:
    def __init__(
        self,
        values: np.ndarray | list | None = None,
        comp_grad: bool = None,  # None means "use global _COMP_GRAD"
        label: str | None = None,
        dtype: str | None = None,
    ) -> None:
        self.dtype = dtype if dtype is not None else "float32"
        if values is None:
            values = np.array([])
        self.values = as_array(values, self.dtype)

        self.shape = self.values.shape
        self.label = label
        self.prev = set()
        self.oper = None  # Operation name (no spaces allowed)
        self.comp_grad = comp_grad if comp_grad is not None else _COMP_GRAD  # Check global at runtime
        self.is_leaf = True
        self.grad = None
        self.backward_step = lambda: None

    def convert_to(self, dtype: str, inplace: bool = True) -> "Tensor" | None:
        if inplace:
            self.dtype = dtype
            self.values = convert_to_dtype(self.values, dtype)
            if self.grad is not None:
                self.grad = convert_to_dtype(self.grad, dtype)
        else:
            new_tensor = Tensor(
                values=convert_to_dtype(self.values, dtype),
                comp_grad=self.comp_grad,
                label=self.label,
            )
            if self.grad is not None:
                new_tensor.grad = convert_to_dtype(self.grad, dtype)
            return new_tensor

    def __len__(self) -> int:
        return len(self.values)

    def __eq__(self, other: any) -> bool:
        return id(self) == id(other)

    def __hash__(self) -> int:
        return hash(id(self))

    def __getitem__(self, idxs) -> tuple:
        return (
            self.values.__getitem__(idxs),
            self.grad.__getitem__(idxs) if self.grad is not None else None,
        )

    def __iter__(self):
        return self.values.__iter__()

    def __str__(self):
        # Determine grad info based on comp_grad and is_leaf
        if self.grad is not None:
            grad_info = f"\ngrad:\n{self.grad}"
        else:
            grad_info = "\ngrad: None"

        return f"Tensor '{self.label}'\nshape: {self.shape}\nis_leaf: {self.is_leaf}\ndtype: {self.dtype}\ncomp_grad: {self.comp_grad}\nvalues:\n{self.values}{grad_info}"

    def zero_grad(self):
        # Initialize gradients only on leaf nodes
        if self.comp_grad and self.is_leaf:
            self.grad = np.zeros(self.shape)
        for t in self.prev:
            t.zero_grad()

    def _check_can_backward(self):
        if not self.comp_grad:
            raise RuntimeError(f"Cannot call backward() on tensor {self.label or ''} with comp_grad=False.")
        if self.grad is not None and not self.is_leaf:
            raise RuntimeError("backward() can only be called once on non-leaf tensors, or you need to use retain_grad()")
        if self.values.size == 0:
            raise RuntimeError("Cannot call backward() on an empty tensor")

    def backward(self) -> None:
        self._check_can_backward()

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

        # clean up the gradients of non-leaf nodes
        for t in topo_order:
            if not t.is_leaf and t != self:
                t.grad = None

    def _init_grad_if_needed(self) -> None:
        if self.grad is None:
            self.grad = np.zeros(self.shape)

    def _reduce_broadcasted_dims(self, delta: np.ndarray) -> np.ndarray:
        while delta.ndim > self.grad.ndim:
            delta = delta.sum(axis=0)
        for i, (d, g) in enumerate(zip(delta.shape, self.grad.shape)):
            if d != g:
                delta = delta.sum(axis=i, keepdims=True)
        return delta

    def __add__(self, other: int | float | "Tensor") -> "Tensor":
        if isinstance(other, (int, float)):
            out = Tensor(self.values + other)
            out.prev = {self}
            out.oper = f"+({other:.2f})"
            out.comp_grad = _should_compute_grad(self)
            out.is_leaf = False

            if out.comp_grad:
                out.backward_step = lambda: add_const_backward(self, other, out)
            return out

        elif isinstance(other, Tensor):
            if self == other:
                return self * 2

            out = Tensor(self.values + other.values)
            out.prev = {self, other}
            out.oper = "+"
            out.comp_grad = _should_compute_grad(self, other)
            out.is_leaf = False

            if out.comp_grad:
                out.backward_step = lambda: _add_backward(self, other, out)
            return out

        else:
            raise ValueError(f"Wrong operand type: {type(other)}")

    def __mul__(self, other: int | float | "Tensor") -> "Tensor":
        if isinstance(other, (int, float)):
            out = Tensor(self.values * other)
            out.prev = {self}
            out.oper = f"*({other:.2f})"
            out.comp_grad = _should_compute_grad(self)  # Scalars don't have comp_grad attribute
            out.is_leaf = False

            if out.comp_grad:
                out.backward_step = lambda: mul_const_backward(self, other, out)
            return out

        elif isinstance(other, Tensor):
            if self == other:
                return self**2

            out = Tensor(self.values * other.values)
            out.prev = {self, other}
            out.oper = "*"
            out.comp_grad = _should_compute_grad(self, other)
            out.is_leaf = False

            if out.comp_grad:
                out.backward_step = lambda: _mul_backward(self, other, out)
            return out

        else:
            raise ValueError(f"Wrong operand type: {type(other)}")

    def __pow__(self, other: int | float) -> "Tensor":
        if isinstance(other, (float, int)):
            if np.any(self.values < 0) and not float(other).is_integer():
                raise ValueError(f"Invalid: {self.label if self.label else 'Tensor'} ** {other} would be complex.")
            out = Tensor(self.values**other)
            out.prev = {self}
            out.oper = f"^{other:.2f}"
            out.comp_grad = _should_compute_grad(self)
            out.is_leaf = False

            if out.comp_grad:
                out.backward_step = lambda: _pow_backward(self, other, out)

            return out
        else:
            raise ValueError(f"Only 'float' or 'int' exponents are supported, got {type(other)}")

    def __matmul__(self, other: "Tensor") -> "Tensor":
        if isinstance(other, Tensor):
            out = Tensor(self.values @ other.values)
            out.prev = {self, other}
            out.oper = "@"
            out.comp_grad = _should_compute_grad(self, other)
            out.is_leaf = False

            if out.comp_grad:
                out.backward_step = lambda: _matmul_backward(self, other, out)
            return out
        else:
            raise ValueError(f"Only 'Tensor' operands are supported for matmul, got {type(other)}")

    @property
    def T(self) -> "Tensor":
        out = Tensor(self.values.T)
        out.prev = {self}
        out.oper = "T"
        out.comp_grad = _should_compute_grad(self)
        out.is_leaf = False

        if out.comp_grad:
            out.backward_step = lambda: _T_backward(self, out)
        return out

    def __sub__(self, other: int | float | "Tensor") -> "Tensor":
        return self + other * -1

    def __truediv__(self, other: int | float | "Tensor") -> "Tensor":
        return self * other**-1

    def __radd__(self, other: int | float | "Tensor") -> "Tensor":
        return self + other

    def __rsub__(self, other: int | float | "Tensor") -> "Tensor":
        return other + self * -1

    def __rmul__(self, other: int | float | "Tensor") -> "Tensor":
        return self * other

    def __rtruediv__(self, other: int | float | "Tensor") -> "Tensor":
        return other * self**-1

    def __neg__(self) -> "Tensor":
        return self * -1

    # identifier for graphviz
    @property
    def _str_id(self) -> str:
        return str(id(self))


def _add_backward(x: Tensor, y: Tensor, out: Tensor) -> None:
    if x.comp_grad:
        x._init_grad_if_needed()
        x.grad += x._reduce_broadcasted_dims(delta=out.grad)
    if y.comp_grad:
        y._init_grad_if_needed()
        y.grad += y._reduce_broadcasted_dims(delta=out.grad)


def add_const_backward(x: Tensor, const: int | float, out: Tensor) -> None:
    if x.comp_grad:
        x._init_grad_if_needed()
        x.grad += x._reduce_broadcasted_dims(delta=out.grad)


def _mul_backward(x: Tensor, y: Tensor, out: Tensor) -> None:
    if x.comp_grad:
        x._init_grad_if_needed()
        x.grad += x._reduce_broadcasted_dims(delta=out.grad * y.values)
    if y.comp_grad:
        y._init_grad_if_needed()
        y.grad += y._reduce_broadcasted_dims(delta=out.grad * x.values)


def mul_const_backward(x: Tensor, const: int | float, out: Tensor) -> None:
    if x.comp_grad:
        x._init_grad_if_needed()
        x.grad += out.grad * const


def _pow_backward(x: Tensor, const: int | float, out: Tensor) -> None:
    if x.comp_grad:
        x._init_grad_if_needed()
        x.grad += out.grad * const * (x.values ** (const - 1))


def _matmul_backward(x: Tensor, y: Tensor, out: Tensor) -> None:
    if x.comp_grad:
        x._init_grad_if_needed()
        x.grad += x._reduce_broadcasted_dims(
            # smart way to transpose matrices in the batches
            np.matmul(out.grad, y.values.swapaxes(-1, -2))
        )
    if y.comp_grad:
        y._init_grad_if_needed()
        y.grad += y._reduce_broadcasted_dims(np.matmul(x.values.swapaxes(-1, -2), out.grad))


def _T_backward(x: Tensor, out: Tensor) -> None:
    if x.comp_grad:
        x._init_grad_if_needed()
        x.grad += out.grad.T
