# Dropout

`Dropout` randomly zeroes elements of its input tensor with probability `p` during training, then scales the remaining values by `1/(1-p)` to keep the expected magnitude constant. This regularisation technique prevents co-adaptation of neurons and reduces overfitting. During evaluation (after calling `model.set_eval_mode()`), the layer becomes a pass-through with no masking applied.

```python
import simplegrad as sg
import simplegrad.nn as nn

model = nn.Sequential(
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Dropout(p=0.3),
    nn.Linear(64, 10),
)
out = model(sg.ones((8, 128)))
```

::: simplegrad.nn.dropout.Dropout
    options:
      members: false
      show_root_heading: true
      heading_level: 2
      docstring_section_style: list

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `.p` | `float` | Probability of zeroing each element during training. |

## Methods

| Method | Description |
|--------|-------------|
| [`.forward()`](dropout/forward.md) | Apply dropout mask during training; pass through during evaluation. |

Inherits all methods from [Module](../core/module.md): `.parameters()`, `.submodules()`, `.to_device()`, `.summary()`, `.set_train_mode()`, `.set_eval_mode()`.
