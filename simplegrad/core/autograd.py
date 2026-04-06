from __future__ import annotations  # make all annotations lazy
import numpy as np
from contextlib import contextmanager
from typing import Callable
from .dtypes import as_array, convert_to_dtype
from .compound_ops import get_current_group

_COMP_GRAD = True
_LAZY_MODE: bool = False


def get_comp_grad():
    """Return the current value of the global flag controlling gradient computation."""
    return _COMP_GRAD


@contextmanager
def no_grad():
    """Context manager that disables gradient computation globally."""
    global _COMP_GRAD
    prev_comp_grad = _COMP_GRAD
    _COMP_GRAD = False
    try:
        yield
    finally:
        _COMP_GRAD = prev_comp_grad


def _should_compute_grad(*inputs) -> bool:
    """Return True if at least one input tensor requires gradients and the global flag is set."""
    if not _COMP_GRAD:
        return False

    for inp in inputs:
        if isinstance(inp, Tensor) and inp.comp_grad:
            return True
    return False


def is_lazy() -> bool:
    """Return True if lazy execution mode is currently active.

    In lazy mode, tensor operations build a deferred computation graph
    instead of executing numpy immediately. Actual computation is
    triggered by calling ``.realize()`` on a tensor or by calling
    ``.backward()``, which auto-realizes before running backprop.

    Returns:
        True if lazy mode is active, False for eager mode (the default).
    """
    return _LAZY_MODE


@contextmanager
def lazy():
    """Context manager that activates lazy execution mode for the enclosed block.

    Inside the block, tensor operations record their computation without
    running numpy. Call ``.realize()`` on the output tensor (or call
    ``.backward()``) to execute the full graph.

    Example:
        >>> with sg.lazy():
        ...     out = model(x)
        ... out.realize()
    """
    global _LAZY_MODE
    prev = _LAZY_MODE
    _LAZY_MODE = True
    try:
        yield
    finally:
        _LAZY_MODE = prev


def set_mode(mode: str) -> None:
    """Set the global execution mode persistently.

    Prefer the ``lazy()`` context manager for scoped control. Use
    ``set_mode`` only when you need the mode to persist across multiple
    function calls or modules.

    Args:
        mode: Either ``"eager"`` (default, execute numpy immediately) or
            ``"lazy"`` (defer execution until ``.realize()`` is called).

    Raises:
        ValueError: If ``mode`` is not ``"eager"`` or ``"lazy"``.

    Example:
        >>> sg.set_mode("lazy")
        >>> out = model(x)
        >>> out.realize()
        >>> sg.set_mode("eager")
    """
    global _LAZY_MODE
    if mode == "lazy":
        _LAZY_MODE = True
    elif mode == "eager":
        _LAZY_MODE = False
    else:
        raise ValueError(f"Unknown mode '{mode}'. Expected 'eager' or 'lazy'.")


def _create_op_result(forward_fn, shape: tuple, dtype: str):
    """Create the output tensor for an operation, respecting the current execution mode.

    In eager mode, calls ``forward_fn()`` immediately and wraps the numpy result
    in a new Tensor. In lazy mode, creates a shell Tensor with ``values=None``
    and stores ``forward_fn`` to be executed later by ``.realize()``.

    This is the single function that all op implementations call instead of
    writing ``Tensor(np.some_op(x.values))`` directly. Keeping all mode-dispatch
    logic here means op functions never need to branch on eager vs lazy themselves.

    Args:
        forward_fn: Zero-argument callable returning a numpy ndarray. Captures
            input tensor objects (not their values), so it stays valid until
            ``.realize()`` populates those tensors' ``.values``.
        shape: The output tensor's shape. Must be correct — this is what
            ``tensor.shape`` will return before ``.realize()`` is called.
        dtype: Output dtype string (e.g. ``"float32"``).

    Returns:
        A new Tensor. In eager mode, ``tensor.values`` is populated. In lazy
        mode, ``tensor.values`` is ``None`` and ``tensor._forward_fn`` holds
        ``forward_fn``.
    """
    if is_lazy():
        return Tensor.deferred(forward_fn, shape=shape, dtype=dtype)
    return Tensor(forward_fn(), dtype=dtype)


class Context:
    """Stores intermediate values computed during a forward pass for reuse in backward.

    Every op that needs to carry state from forward to backward should create a
    ``Context``, write to it inside the forward lambda, and read from it inside
    the backward function. This pattern works in both eager and lazy mode: in
    eager mode the forward lambda runs immediately; in lazy mode it runs at
    ``.realize()`` time — either way, the backward always runs after the forward,
    so ``ctx`` attributes are always populated by the time they are read.

    Attributes are set freely with dot notation — use whatever names are
    meaningful for the op.

    Example:
        >>> ctx = Context()
        >>> ctx.mask = np.random.rand(*x.values.shape) >= p
        >>> ctx.mask  # available in backward
    """

    pass


class Function:
    """Base class for differentiable operations.

    Subclass this and implement ``forward`` and ``backward`` as static methods.
    Call ``cls.apply(*inputs)`` to run the op — it handles creating the output
    tensor, wiring the computation graph, and setting up gradient accumulation.

    ``forward`` computes and returns the numpy result (and saves anything needed
    for backward into ``ctx``). ``backward`` receives the upstream gradient and
    returns one gradient array per Tensor input — pure computation, no
    accumulation. The ``apply`` method handles accumulating those gradients into
    ``.grad`` via ``+=``, including broadcast dimension reduction.

    Class attributes:
        oper: Short label shown on graph nodes. Defaults to the class name.
        differentiable: Set to False for ops like argmax that have no gradient.
    """

    oper: str = ""
    differentiable: bool = True

    @classmethod
    def apply(cls, *inputs: object, oper: str | None = None) -> "Tensor":
        """Run the op, build the graph node, and wire up the backward step.

        Args:
            *inputs: Tensor and non-Tensor arguments forwarded to ``forward``
                and ``output_shape``. Non-Tensor inputs are ignored during
                gradient accumulation.
            oper: Optional label override for the graph node. Falls back to
                ``cls.oper`` then ``cls.__name__``.

        Returns:
            Output tensor wired into the computation graph.
        """

        ctx = Context()
        tensor_inputs = [t for t in inputs if isinstance(t, Tensor)]
        out = _create_op_result(
            lambda: cls.forward(ctx, *inputs),
            shape=cls.output_shape(*inputs),
            dtype=tensor_inputs[0].dtype,
        )
        out.prev = set(tensor_inputs)
        out.comp_grad = _should_compute_grad(*tensor_inputs) and cls.differentiable
        out.is_leaf = False
        out.oper = oper if oper is not None else (cls.oper or cls.__name__)
        if out.comp_grad:
            out.backward_step = lambda: cls._accumulate(ctx, out, tensor_inputs)
        return out

    @classmethod
    def _accumulate(cls, ctx, out, tensor_inputs: list) -> None:
        """Call backward and accumulate the returned gradients into each input."""
        grads = cls.backward(ctx, out.grad)
        if not isinstance(grads, tuple):
            grads = (grads,)
        for inp, grad in zip(tensor_inputs, grads):
            if inp.comp_grad and grad is not None:
                inp._init_grad_if_needed()
                inp.grad += inp._reduce_broadcasted_dims(grad)

    @staticmethod
    def output_shape(*inputs) -> tuple:
        """Infer the output shape from inputs without executing the op.

        The default returns the shape of the first Tensor input (correct for
        element-wise ops). Override for ops where the output shape differs.
        """

        for inp in inputs:
            if isinstance(inp, Tensor):
                return inp.shape
        raise ValueError("No Tensor input found")

    @staticmethod
    def forward(ctx, *inputs) -> np.ndarray:
        """Compute the forward pass. Save anything needed for backward into ctx."""
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output: np.ndarray) -> np.ndarray | tuple:
        """Compute gradients. Return one array per Tensor input (None = no grad)."""
        raise NotImplementedError


class Tensor:
    """N-dimensional array with automatic differentiation support.

    Wraps a numpy array and records operations into a dynamic computation graph.
    Call `.backward()` on a scalar output to propagate gradients back to all
    leaf tensors with `comp_grad=True`.

    Args:
        values: Initial data. Converted to a numpy array of the given dtype.
        comp_grad: Enable gradient tracking. Defaults to the global `_COMP_GRAD` flag.
        label: Optional name shown in computation graph visualizations.
        dtype: Data type string (e.g. ``"float32"``). Defaults to ``"float32"``.
    """

    def __init__(
        self,
        values: np.ndarray | list | None = None,
        comp_grad: bool = None,
        label: str | None = None,
        dtype: str | None = None,
    ) -> None:
        self.dtype = dtype if dtype is not None else "float32"
        if values is None:
            values = np.array([])
        self.values = as_array(values, self.dtype)
        self.shape = self.values.shape
        self._forward_fn = None

        self.label = label
        self.prev = set()
        self.oper = None
        self.comp_grad = comp_grad if comp_grad is not None else get_comp_grad()
        self.is_leaf = True
        self.grad = None
        self.backward_step = lambda: None
        self.group: tuple[str, int] | None = get_current_group()

    @classmethod
    def deferred(
        cls, forward_fn: Callable[[], np.ndarray], shape: tuple, dtype: str = "float32"
    ) -> "Tensor":
        """Create an unrealized tensor that defers computation to ``.realize()``.

        Used internally by ``_create_op_result`` when lazy mode is active. The
        tensor is a shell — ``values`` is ``None`` and ``shape`` is known —
        until ``.realize()`` walks the graph and calls ``forward_fn``.

        Args:
            forward_fn: Zero-argument callable that returns a numpy ndarray.
                Captures input tensors by reference, so it stays valid until
                ``.realize()`` populates their ``.values``.
            shape: Output shape. Available immediately without executing the op.
            dtype: Data type string. Defaults to ``"float32"``.

        Returns:
            An unrealized Tensor with ``values=None``.
        """
        t = cls.__new__(cls)
        t.dtype = dtype
        t.values = None
        t.shape = shape
        t._forward_fn = forward_fn
        t.label = None
        t.prev = set()
        t.oper = None
        t.comp_grad = get_comp_grad()
        t.is_leaf = True
        t.grad = None
        t.backward_step = lambda: None
        t.group = get_current_group()
        return t

    def convert_to(self, dtype: str, inplace: bool = True) -> "Tensor" | None:
        """Convert tensor values (and gradients) to a different dtype.

        Args:
            dtype: Target dtype string (e.g. ``"float64"``).
            inplace: Modify this tensor in-place if True, else return a new tensor.

        Returns:
            New Tensor if ``inplace=False``, else None.
        """
        if self.values is None:
            raise RuntimeError(
                "Cannot convert dtype of an unrealized tensor. Call .realize() first."
            )
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
        if self.values is None:
            raise RuntimeError("Cannot get length of an unrealized tensor. Call .realize() first.")
        return len(self.values)

    def __eq__(self, other: any) -> bool:
        return id(self) == id(other)

    def __hash__(self) -> int:
        return hash(id(self))

    def __getitem__(self, idxs) -> tuple:
        if self.values is None:
            raise RuntimeError("Cannot index an unrealized tensor. Call .realize() first.")
        return (
            self.values.__getitem__(idxs),
            self.grad.__getitem__(idxs) if self.grad is not None else None,
        )

    def __iter__(self):
        if self.values is None:
            raise RuntimeError("Cannot iterate over an unrealized tensor. Call .realize() first.")
        return self.values.__iter__()

    def __str__(self):
        if self.values is None:
            return (
                f"Tensor '{self.label}' [unrealized]\n"
                f"shape: {self.shape}\n"
                f"is_leaf: {self.is_leaf}\n"
                f"dtype: {self.dtype}\n"
                f"comp_grad: {self.comp_grad}"
            )
        if self.grad is not None:
            grad_info = f"\ngrad:\n{self.grad}"
        else:
            grad_info = "\ngrad: None"
        return (
            f"Tensor '{self.label}'\nshape: {self.shape}\nis_leaf: {self.is_leaf}\n"
            f"dtype: {self.dtype}\ncomp_grad: {self.comp_grad}\nvalues:\n{self.values}{grad_info}"
        )

    def zero_grad(self):
        """Zero gradients on all leaf tensors in the computation graph."""
        if self.comp_grad and self.is_leaf:
            self.grad = np.zeros(self.shape)
        for t in self.prev:
            t.zero_grad()

    def realize(self) -> "Tensor":
        """Execute all pending forward computations in the computation graph.

        Walks the graph in topological order (inputs before outputs) and
        executes any stored ``_forward_fn`` callables, filling in
        ``tensor.values`` for every unrealized tensor. After this call,
        ``self.values`` and the values of every upstream tensor are guaranteed
        to be non-None.

        In eager mode this is a no-op — all tensors are already realized.

        Returns:
            ``self``, so you can chain: ``loss = model(x).realize()``.
        """
        topo: list[Tensor] = []
        visited: set[int] = set()

        def _build_topo(t: Tensor) -> None:
            if id(t) in visited:
                return
            visited.add(id(t))
            for parent in t.prev:
                _build_topo(parent)
            topo.append(t)

        _build_topo(self)

        for t in topo:
            if t._forward_fn is not None:
                t.values = t._forward_fn()
                t.shape = t.values.shape
                t._forward_fn = None

        return self

    def _check_can_backward(self):
        """Raise if backward() cannot be called on this tensor."""
        if not self.comp_grad:
            raise RuntimeError(
                f"Cannot call backward() on tensor {self.label or ''} with comp_grad=False."
            )
        if self.grad is not None and not self.is_leaf:
            raise RuntimeError(
                "backward() can only be called once on non-leaf tensors, or you need to use retain_grad()"
            )
        if self.values is not None and self.values.size == 0:
            raise RuntimeError("Cannot call backward() on an empty tensor")

    def backward(self) -> None:
        """Run backpropagation from this tensor.

        If the tensor is unrealized (lazy mode), calls ``.realize()``
        automatically before running backprop. This means you can always call
        ``.backward()`` directly without a separate ``.realize()`` step.

        Computes gradients for all leaf tensors in the graph with ``comp_grad=True``.
        This tensor's gradient is initialized to ones (assumes scalar loss).
        Non-leaf gradients are freed after the backward pass.

        Raises:
            RuntimeError: If ``comp_grad=False``, tensor is empty, or backward has
                already been called on this non-leaf tensor.
        """
        self.realize()
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

        for t in topo_order:
            if not t.is_leaf and t != self:
                t.grad = None

    def _init_grad_if_needed(self) -> None:
        """Initialize gradient array to zeros if not yet set."""
        if self.grad is None:
            self.grad = np.zeros(self.shape)

    def _reduce_broadcasted_dims(self, delta: np.ndarray) -> np.ndarray:
        """Sum gradient over broadcast dimensions to match this tensor's shape."""
        while delta.ndim > self.grad.ndim:
            delta = delta.sum(axis=0)
        for i, (d, g) in enumerate(zip(delta.shape, self.grad.shape)):
            if d != g:
                delta = delta.sum(axis=i, keepdims=True)
        return delta

    def __add__(self, other: int | float | "Tensor") -> "Tensor":
        """Add a scalar or tensor element-wise."""
        if isinstance(other, (int, float)):
            return _AddScalar.apply(self, other, oper=f"+({other:.2f})")
        if not isinstance(other, Tensor):
            raise ValueError(f"Wrong operand type: {type(other)}")
        if self == other:
            return self * 2
        return _Add.apply(self, other)

    def __mul__(self, other: int | float | "Tensor") -> "Tensor":
        """Multiply by a scalar or tensor element-wise."""
        if isinstance(other, (int, float)):
            return _MulScalar.apply(self, other, oper=f"*({other:.2f})")
        if not isinstance(other, Tensor):
            raise ValueError(f"Wrong operand type: {type(other)}")
        if self == other:
            return self**2
        return _Mul.apply(self, other)

    def __pow__(self, other: int | float) -> "Tensor":
        """Raise to a scalar power element-wise."""
        if not isinstance(other, (float, int)):
            raise ValueError(f"Only 'float' or 'int' exponents are supported, got {type(other)}")
        if not is_lazy() and np.any(self.values < 0) and not float(other).is_integer():
            raise ValueError(
                f"Invalid: {self.label if self.label else 'Tensor'} ** {other} would be complex."
            )
        return _Pow.apply(self, other, oper=f"^{other:.2f}")

    def __matmul__(self, other: "Tensor") -> "Tensor":
        """Matrix multiply with another tensor."""
        if not isinstance(other, Tensor):
            raise ValueError(f"Only 'Tensor' operands are supported for matmul, got {type(other)}")
        return _Matmul.apply(self, other)

    @property
    def T(self) -> "Tensor":
        """Transpose of the tensor."""
        return _Transpose.apply(self)

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

    @property
    def _str_id(self) -> str:
        return str(id(self))


class _Add(Function):
    oper = "+"

    @staticmethod
    def output_shape(x, y) -> tuple:
        return np.broadcast_shapes(x.shape, y.shape)

    @staticmethod
    def forward(ctx: Context, x, y) -> np.ndarray:
        return x.values + y.values

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> tuple:
        return grad_output, grad_output


class _AddScalar(Function):
    oper = "+"

    @staticmethod
    def forward(ctx: Context, x, c) -> np.ndarray:
        return x.values + c

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> np.ndarray:
        return grad_output


class _Mul(Function):
    oper = "*"

    @staticmethod
    def output_shape(x, y) -> tuple:
        return np.broadcast_shapes(x.shape, y.shape)

    @staticmethod
    def forward(ctx: Context, x, y) -> np.ndarray:
        ctx.x_values = x.values
        ctx.y_values = y.values
        return x.values * y.values

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> tuple:
        return grad_output * ctx.y_values, grad_output * ctx.x_values


class _MulScalar(Function):
    oper = "*"

    @staticmethod
    def forward(ctx: Context, x, c) -> np.ndarray:
        ctx.c = c
        return x.values * c

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * ctx.c


class _Pow(Function):
    oper = "^"

    @staticmethod
    def forward(ctx: Context, x, n) -> np.ndarray:
        ctx.x_values = x.values
        ctx.n = n
        return x.values**n

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * ctx.n * (ctx.x_values ** (ctx.n - 1))


class _Matmul(Function):
    oper = "@"

    @staticmethod
    def output_shape(x, y) -> tuple:
        return (*x.shape[:-2], x.shape[-2], y.shape[-1])

    @staticmethod
    def forward(ctx: Context, x, y) -> np.ndarray:
        ctx.x_values = x.values
        ctx.y_values = y.values
        return x.values @ y.values

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> tuple:
        # For z = x @ y:
        # dz/dx = grad_output @ y^T
        # dz/dy = x^T @ grad_output
        # In batched matmul, we treat only the last two axes as matrix axes and all preceding axes as batch axes.
        # Therefore we use swapaxes(-1, -2) to transpose each matrix independently while leaving batch dimensions unchanged.
        grad_x = np.matmul(grad_output, ctx.y_values.swapaxes(-1, -2))  #
        grad_y = np.matmul(ctx.x_values.swapaxes(-1, -2), grad_output)
        return grad_x, grad_y


class _Transpose(Function):
    oper = "T"

    @staticmethod
    def output_shape(x) -> tuple:
        return x.shape[::-1]

    @staticmethod
    def forward(ctx: Context, x) -> np.ndarray:
        return x.values.T

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> np.ndarray:
        return grad_output.T
