import numpy as np
from contextlib import contextmanager
from typing import Union, Optional
from simplegrad.dtypes import as_array, convert_to_dtype

_COMP_GRAD = True


@contextmanager
def no_grad():
    global _COMP_GRAD
    prev_comp_grad = _COMP_GRAD
    _COMP_GRAD = False
    try:
        print(_COMP_GRAD)
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
        values: Optional[Union[np.ndarray, list]] = None,
        comp_grad: bool = None,  # None means "use global _COMP_GRAD"
        label: Optional[str] = None,
        dtype: Optional[str] = None,
        # column: Optional[bool] = False,
    ) -> None:
        self.dtype = dtype if dtype is not None else "float32"
        if values is None:
            values = np.array([])
        self.values = as_array(values, self.dtype)

        # if self.values.ndim == 0:
        #     self.values = self.values.reshape(1, 1)
        # elif self.values.ndim == 1:
        #     self.values = self.values.reshape(-1, 1) if column else self.values.reshape(1, -1)
        self.shape = self.values.shape
        self.label = label
        self.prev = set()
        self.oper = None  # Operation name (no spaces allowed)
        self.comp_grad = comp_grad if comp_grad is not None else _COMP_GRAD  # Check global at runtime
        self.is_leaf = True
        self.grad = None
        self.backward_step = lambda: None

    def reshape(self, new_shape: tuple[int], inplace: bool = True, include_grad: bool = True) -> Optional["Tensor"]:
        if inplace:
            self.values = self.values.reshape(new_shape)
            self.shape = self.values.shape
            if self.grad is not None and include_grad:
                self.grad = self.grad.reshape(new_shape)
        else:
            new_tensor = Tensor(
                values=self.values.reshape(new_shape),
                comp_grad=self.comp_grad,
                label=self.label,
            )
            return new_tensor

    def convert_to_dtype(self, dtype: str, inplace: bool = True) -> Optional["Tensor"]:
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

    def backward(self):
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

    def _init_grad_if_needed(self):
        if self.grad is None:
            self.grad = np.zeros(self.shape)

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
            out.comp_grad = _should_compute_grad(self)
            out.is_leaf = False

            if out.comp_grad:

                def backward_step():
                    if self.comp_grad:
                        self._init_grad_if_needed()
                        self.grad += self._reduce_broadcasted_dims(delta=out.grad)

                out.backward_step = backward_step
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

                def backward_step():
                    if self.comp_grad:
                        self._init_grad_if_needed()
                        self.grad += self._reduce_broadcasted_dims(delta=out.grad)
                    if other.comp_grad:
                        other._init_grad_if_needed()
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
            out.comp_grad = _should_compute_grad(self)  # Scalars don't have comp_grad attribute
            out.is_leaf = False

            if out.comp_grad:

                def backward_step():
                    if self.comp_grad:
                        self._init_grad_if_needed()
                        self.grad += out.grad * other

                out.backward_step = backward_step
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

                def backward_step():
                    if self.comp_grad:
                        self._init_grad_if_needed()
                        self.grad += self._reduce_broadcasted_dims(delta=out.grad * other.values)

                    if other.comp_grad:
                        other._init_grad_if_needed()
                        other.grad += other._reduce_broadcasted_dims(delta=out.grad * self.values)

                out.backward_step = backward_step
            return out

        else:
            raise ValueError(f"Wrong operand type: {type(other)}")

    def __pow__(self, other):
        if isinstance(other, (float, int)):
            if np.any(self.values < 0) and not float(other).is_integer():
                raise ValueError(f"Invalid: {self.label if self.label else 'Tensor'} ** {other} would be complex.")
            out = Tensor(self.values**other)
            out.prev = {self}
            out.oper = f"^{other:.2f}"
            out.comp_grad = _should_compute_grad(self)
            out.is_leaf = False

            if out.comp_grad:

                def backward_step():
                    if self.comp_grad:
                        self._init_grad_if_needed()
                        self.grad += out.grad * other * (self.values ** (other - 1))

                out.backward_step = backward_step

            return out
        else:
            raise ValueError(f"Only 'float' or 'int' exponents are supported, got {type(other)}")

    def __matmul__(self, other):
        if isinstance(other, Tensor):
            out = Tensor(self.values @ other.values)
            out.prev = {self, other}
            out.oper = "@"
            out.comp_grad = _should_compute_grad(self, other)
            out.is_leaf = False

            if out.comp_grad:

                def backward_step():
                    if self.comp_grad:
                        self._init_grad_if_needed()
                        self.grad += self._reduce_broadcasted_dims(
                            # smart way to transpose matrices in the batches
                            np.matmul(out.grad, other.values.swapaxes(-1, -2))
                        )
                    if other.comp_grad:
                        other._init_grad_if_needed()
                        other.grad += other._reduce_broadcasted_dims(np.matmul(self.values.swapaxes(-1, -2), out.grad))

                out.backward_step = backward_step
            return out
        else:
            raise ValueError(f"Only 'Tensor' operands are supported for matmul, got {type(other)}")

    @property
    def T(self):
        out = Tensor(self.values.T)
        out.prev = {self}
        out.oper = "T"
        out.comp_grad = _should_compute_grad(self)
        out.is_leaf = False

        if out.comp_grad:

            def backward_step():
                if self.comp_grad:
                    self._init_grad_if_needed()
                    self.grad += out.grad.T

            out.backward_step = backward_step
        return out

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
