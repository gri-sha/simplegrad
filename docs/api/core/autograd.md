# Tensor & Autograd

`Tensor` is the central data structure of simplegrad. It wraps a NumPy array and records operations in a computation graph so that gradients can be computed via reverse-mode automatic differentiation. Every mathematical operation on a `Tensor` returns a new `Tensor` and, when `comp_grad=True`, wires itself into the graph for backpropagation.

```python
import simplegrad as sg

x = sg.Tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x * x).sum()
y.backward()
print(x.grad)  # [2. 4. 6.]
```

::: simplegrad.core.autograd.Tensor
    options:
      members: false
      show_root_heading: true
      heading_level: 2
      docstring_section_style: list

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `.values` | `ndarray \| None` | The underlying NumPy or CuPy array. `None` until `.realize()` is called in lazy mode. |
| `.shape` | `tuple` | Shape of the tensor. Available even before `.realize()`. |
| `.dtype` | `str` | Data type string, e.g. `"float32"`. |
| `.device` | `str` | Device string, e.g. `"cpu"` or `"cuda:0"`. |
| `.comp_grad` | `bool` | Whether gradient tracking is enabled for this tensor. |
| `.is_leaf` | `bool` | `True` if the tensor was not produced by an operation (e.g. a parameter or input). |
| `.grad` | `ndarray \| None` | Accumulated gradient array after `.backward()`. `None` until backprop runs. |
| `.label` | `str \| None` | Optional name used in computation graph visualizations. |

## Methods

| Method | Description |
|--------|-------------|
| [`.backward()`](tensor/backward.md) | Run backpropagation from this tensor, computing gradients for all leaves. |
| [`.realize()`](tensor/realize.md) | Execute all pending forward computations in lazy mode. |
| [`.zero_grad()`](tensor/zero_grad.md) | Zero gradients on all leaf tensors in the computation graph. |
| [`.convert_to()`](tensor/convert_to.md) | Convert tensor values to a different dtype. |
| [`.to_device()`](tensor/to_device.md) | Copy this tensor to a target device, returning a new leaf tensor. |
| [`.deferred()`](tensor/deferred.md) | Class method — create an unrealized tensor that defers computation. |
| [`.T`](tensor/T.md) | Property — transpose of the tensor. |

## Arithmetic operators

| Operator | Expression | Description |
|----------|------------|-------------|
| `__add__` | `a + b` | Element-wise addition with a scalar or tensor. |
| `__sub__` | `a - b` | Element-wise subtraction. |
| `__mul__` | `a * b` | Element-wise multiplication with a scalar or tensor. |
| `__truediv__` | `a / b` | Element-wise division. |
| `__pow__` | `a ** n` | Element-wise power with a scalar exponent. |
| `__matmul__` | `a @ b` | Matrix multiplication. |
| `__neg__` | `-a` | Element-wise negation. |

---

## Execution mode

Simplegrad supports two execution modes — **eager** (default) and **lazy**. In eager mode every operation runs NumPy immediately. In lazy mode operations are recorded into a graph and only executed when `.realize()` or `.backward()` is called, enabling whole-graph optimisations. The `no_grad` context manager disables gradient tracking entirely, which is useful during inference to save memory and computation.

```python
import simplegrad as sg

# Disable gradient tracking during inference
with sg.no_grad():
    out = model(x)

# Defer execution until .realize()
with sg.lazy():
    out = model(x)
out.realize()
```

::: simplegrad.core.autograd.no_grad
    options:
      show_root_heading: true
      heading_level: 3
      docstring_section_style: list

::: simplegrad.core.autograd.lazy
    options:
      show_root_heading: true
      heading_level: 3
      docstring_section_style: list

::: simplegrad.core.autograd.mode
    options:
      show_root_heading: true
      heading_level: 3
      docstring_section_style: list

::: simplegrad.core.autograd.is_lazy
    options:
      show_root_heading: true
      heading_level: 3
      docstring_section_style: list
