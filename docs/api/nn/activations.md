# Activation Layers

Activation layers are `Module` wrappers around the functional activation ops, making it convenient to compose them inside `Sequential` or custom `Module` subclasses. Each layer holds no trainable parameters — it simply calls the corresponding function from `simplegrad.functions.activations` in its `forward` method.

```python
import simplegrad as sg
import simplegrad.nn as nn

model = nn.Sequential(
    nn.Linear(16, 32),
    nn.ReLU(),
    nn.Linear(32, 10),
)
out = model(sg.ones((4, 16)))
```

All activation layers inherit from [Module](../core/module.md) and share its methods (`.parameters()`, `.to_device()`, `.set_train_mode()`, etc.).

---

## ReLU

::: simplegrad.nn.activation_layers.ReLU
    options:
      members: false
      show_root_heading: true
      heading_level: 3
      docstring_section_style: list

| Method | Description |
|--------|-------------|
| [`.forward()`](relu/forward.md) | Apply `max(0, x)` element-wise. |

---

## ELU

::: simplegrad.nn.activation_layers.ELU
    options:
      members: false
      show_root_heading: true
      heading_level: 3
      docstring_section_style: list

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `.alpha` | `float` | Scale for the negative saturation region. Defaults to `1.0`. |

### Methods

| Method | Description |
|--------|-------------|
| [`.forward()`](elu/forward.md) | Apply ELU activation element-wise. |

---

## Tanh

::: simplegrad.nn.activation_layers.Tanh
    options:
      members: false
      show_root_heading: true
      heading_level: 3
      docstring_section_style: list

| Method | Description |
|--------|-------------|
| [`.forward()`](tanh/forward.md) | Apply `tanh(x)` element-wise. |

---

## Sigmoid

::: simplegrad.nn.activation_layers.Sigmoid
    options:
      members: false
      show_root_heading: true
      heading_level: 3
      docstring_section_style: list

| Method | Description |
|--------|-------------|
| [`.forward()`](sigmoid/forward.md) | Apply `1 / (1 + exp(-x))` element-wise. |

---

## GELU

::: simplegrad.nn.activation_layers.GELU
    options:
      members: false
      show_root_heading: true
      heading_level: 3
      docstring_section_style: list

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `.mode` | `str` | Approximation mode: `"erf"` (exact) or `"tanh"` (fast). Defaults to `"erf"`. |

### Methods

| Method | Description |
|--------|-------------|
| [`.forward()`](gelu/forward.md) | Apply GELU activation element-wise. |

---

## Softmax

::: simplegrad.nn.activation_layers.Softmax
    options:
      members: false
      show_root_heading: true
      heading_level: 3
      docstring_section_style: list

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `.dim` | `int \| None` | Axis along which softmax is computed. |

### Methods

| Method | Description |
|--------|-------------|
| [`.forward()`](softmax/forward.md) | Apply softmax along the configured axis. |
