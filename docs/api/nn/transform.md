# Flatten

`Flatten` is a `Module` wrapper that collapses spatial dimensions into a single feature vector, bridging convolutional layers and fully connected layers. It calls the functional `flatten` op under the hood and holds no learnable parameters. Use it inside a `Sequential` model to avoid writing a custom `forward` just for reshaping.

```python
import simplegrad.nn as nn

model = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Flatten(),                # (N, 16, 28, 28) -> (N, 12544)
    nn.Linear(12544, 10),
)
```

::: simplegrad.nn.transform.Flatten
    options:
      members: false
      show_root_heading: true
      heading_level: 2
      docstring_section_style: list

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `.start_dim` | `int` | First dimension to flatten. Defaults to `1` (preserves batch dim). |
| `.end_dim` | `int` | Last dimension to flatten. Defaults to `-1` (all remaining). |

## Methods

| Method | Description |
|--------|-------------|
| [`.forward()`](flatten/forward.md) | Flatten the input tensor from `start_dim` to `end_dim`. |

Inherits all methods from [Module](../core/module.md): `.parameters()`, `.submodules()`, `.to_device()`, `.summary()`, `.set_train_mode()`, `.set_eval_mode()`.
