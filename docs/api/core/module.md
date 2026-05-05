# Module

`Module` is the base class for all neural network layers in simplegrad. Subclass it and implement `forward()` to define a custom layer. Parameter tensors and sub-modules are discovered automatically via attribute introspection, so you can call `.parameters()` or `.summary()` on any composite model without wiring anything by hand.

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
    options:
      members: false
      show_root_heading: true
      heading_level: 2
      docstring_section_style: list

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `.label` | `str` | Display name of the module. Defaults to the class name. |
| `.eval_mode` | `bool` | `True` when in evaluation mode (set via `.set_eval_mode()`). |

## Methods

| Method | Description |
|--------|-------------|
| [`.forward()`](module/forward.md) | Define the forward pass. Must be implemented by subclasses. |
| [`.parameters()`](module/parameters.md) | Return all parameter tensors in this module and its sub-modules. |
| [`.submodules()`](module/submodules.md) | Return all direct sub-modules as a named dict. |
| [`.to_device()`](module/to_device.md) | Move all parameters to the target device in-place. |
| [`.summary()`](module/summary.md) | Print a table of all parameters, their shapes, and total parameter count. |
| [`.set_train_mode()`](module/set_train_mode.md) | Switch this module and all sub-modules to training mode. |
| [`.set_eval_mode()`](module/set_eval_mode.md) | Switch this module and all sub-modules to evaluation mode. |
