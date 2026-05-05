# Function & Context

`Function` is the base class for every differentiable operation in simplegrad. You subclass it, implement `forward` (runs NumPy) and `backward` (returns local gradients), and call `cls.apply(...)` to execute. The `apply` method wires the result into the computation graph automatically. `Context` is a simple namespace used to shuttle intermediate values computed in `forward` through to `backward`.

```python
import simplegrad as sg
from simplegrad.core import Function, Context
import numpy as np

class _Square(Function):
    oper = "Square"

    @staticmethod
    def forward(ctx: Context, x) -> np.ndarray:
        ctx.x_values = x.values
        return x.values ** 2

    @staticmethod
    def backward(ctx: Context, grad):
        return 2 * ctx.x_values * grad

x = sg.Tensor([2.0, 3.0], requires_grad=True)
y = _Square.apply(x)
y.sum().backward()
print(x.grad)  # [4. 6.]
```

## Function

::: simplegrad.core.autograd.Function
    options:
      members: false
      show_root_heading: true
      heading_level: 3
      docstring_section_style: list

### Class attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `.oper` | `str` | Short label shown on computation graph nodes. Defaults to the class name. |
| `.differentiable` | `bool` | Set to `False` for ops with no gradient (e.g. `argmax`). |

### Methods

| Method | Description |
|--------|-------------|
| [`.apply()`](function/apply.md) | Run the op, build the graph node, and wire up the backward step. |
| [`.forward()`](function/forward.md) | Compute the forward pass. Save anything needed for backward into `ctx`. |
| [`.backward()`](function/backward.md) | Compute gradients. Return one array per Tensor input. |
| [`.output_shape()`](function/output_shape.md) | Infer the output shape from inputs without executing the op. |

## Context

::: simplegrad.core.autograd.Context
    options:
      members: false
      show_root_heading: true
      heading_level: 3
      docstring_section_style: list

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `.device` | `str` | Device string of the operation's tensors (e.g. `"cpu"`). Set automatically by `apply`. |
| `.backend` | `module` | Compute module — `numpy` or `cupy`. Alias as `xp = ctx.backend` in ops. |
