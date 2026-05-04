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

::: simplegrad.core.autograd.Function

::: simplegrad.core.autograd.Context
