# Factory Functions

Factory functions create `Tensor` objects with common initializations without requiring raw NumPy arrays. They are re-exported at the top-level `simplegrad` namespace, so you can call `sg.zeros(...)`, `sg.ones(...)`, etc. directly. All factories accept an optional `device` argument and respect the global default dtype.

```python
import simplegrad as sg

w = sg.normal((4, 8), requires_grad=True)   # Gaussian-initialised weights
b = sg.zeros((4,), requires_grad=True)      # zero bias
x = sg.uniform((2, 8), low=-1, high=1)     # random input
```

::: simplegrad.core.factory.zeros

::: simplegrad.core.factory.ones

::: simplegrad.core.factory.normal

::: simplegrad.core.factory.uniform

::: simplegrad.core.factory.full
