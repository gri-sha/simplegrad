# MaxPool2d

`MaxPool2d` is a `Module` wrapper around the functional `max_pool2d` op. It down-samples spatial feature maps by retaining the maximum activation in each pooling window, reducing computational cost and providing a degree of spatial invariance. It holds no learnable parameters and is typically placed after a `Conv2d` layer in a CNN.

```python
import simplegrad as sg
import simplegrad.nn as nn

pool = nn.MaxPool2d(kernel_size=2, stride=2)
x = sg.normal((1, 16, 28, 28), requires_grad=True)
out = pool(x)  # shape: (1, 16, 14, 14)
```

::: simplegrad.nn.pooling.MaxPool2d
    options:
      members: false
      show_root_heading: true
      heading_level: 2
      docstring_section_style: list

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `.kernel_size` | `int \| tuple` | Size of the pooling window. |
| `.stride` | `int \| tuple` | Step size of the sliding window. |
| `.padding` | `int \| tuple` | Zero-padding added to both spatial dimensions. |

## Methods

| Method | Description |
|--------|-------------|
| [`.forward()`](pooling/forward.md) | Apply max-pooling to the input feature map. |

Inherits all methods from [Module](../core/module.md): `.parameters()`, `.submodules()`, `.to_device()`, `.summary()`, `.set_train_mode()`, `.set_eval_mode()`.
