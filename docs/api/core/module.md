# Module

`Module` is the base class for all neural network layers in simplegrad. Subclass it and implement `forward()` to define a custom layer. Parameter tensors and sub-modules are discovered automatically via attribute introspection, so you can call `.parameters()` or `.zero_grad()` on any composite model without wiring anything by hand.

```python
import simplegrad as sg
from simplegrad.core import Module

class MyLayer(Module):
    def __init__(self):
        super().__init__()
        self.weight = sg.normal((4, 4), requires_grad=True)

    def forward(self, x):
        return x @ self.weight

layer = MyLayer()
out = layer(sg.ones((2, 4)))
```

::: simplegrad.core.module.Module
