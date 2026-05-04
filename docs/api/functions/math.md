# Math Functions

These are element-wise mathematical functions with full gradient support. Each one wraps a NumPy primitive inside a `Function` subclass so that gradients flow correctly through the operation during backpropagation. They are useful when building custom loss functions or architectures that require explicit mathematical transformations.

```python
import simplegrad as sg
from simplegrad.functions.math import log, exp

x = sg.Tensor([1.0, 2.0, 3.0], requires_grad=True)
y = log(x) + exp(x)
y.sum().backward()
print(x.grad)
```

::: simplegrad.functions.math.log

::: simplegrad.functions.math.exp

::: simplegrad.functions.math.sin

::: simplegrad.functions.math.cos

::: simplegrad.functions.math.tan
