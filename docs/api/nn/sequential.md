# Sequential

`Sequential` chains a list of `Module` layers and calls them in order during the forward pass, passing the output of each layer as the input to the next. It is the fastest way to build simple feed-forward architectures without writing a custom `Module` subclass. All nested parameters are discovered automatically and can be passed to an optimizer.

```python
import simplegrad as sg
import simplegrad.nn as nn

model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)
out = model(sg.normal((32, 784)))
```

::: simplegrad.nn.sequential.Sequential
    options:
      members: false
      show_root_heading: true
      heading_level: 2
      docstring_section_style: list

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `.modules` | `list[Module]` | Ordered list of layers passed at construction. |

## Methods

| Method | Description |
|--------|-------------|
| [`.forward()`](sequential/forward.md) | Pass `x` through each layer in order and return the final output. |

Inherits all methods from [Module](../core/module.md): `.parameters()`, `.submodules()`, `.to_device()`, `.summary()`, `.set_train_mode()`, `.set_eval_mode()`.
