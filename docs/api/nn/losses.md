# Loss Layers

Loss layers are `Module` wrappers around the functional loss ops, making it easy to slot a loss function into a model definition or training loop as a named module. `CELoss` wraps cross-entropy loss for classification and `MSELoss` wraps mean squared error for regression. Neither layer has trainable parameters.

```python
import simplegrad as sg
import simplegrad.nn as nn

criterion = nn.CELoss()
logits = sg.normal((8, 10), requires_grad=True)
targets = sg.Tensor([0, 1, 2, 3, 4, 5, 6, 7])
loss = criterion(logits, targets)
loss.backward()
```

All loss layers inherit from [Module](../core/module.md) and share its methods.

---

## CELoss

::: simplegrad.nn.loss_layers.CELoss
    options:
      members: false
      show_root_heading: true
      heading_level: 3
      docstring_section_style: list

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `.dim` | `int` | Axis over which softmax is computed. Defaults to `-1`. |
| `.reduction` | `str \| None` | `"mean"`, `"sum"`, or `None` for no reduction. |

### Methods

| Method | Description |
|--------|-------------|
| [`.forward()`](celoss/forward.md) | Compute cross-entropy loss from logits and integer targets. |

---

## MSELoss

::: simplegrad.nn.loss_layers.MSELoss
    options:
      members: false
      show_root_heading: true
      heading_level: 3
      docstring_section_style: list

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `.reduction` | `str \| None` | `"mean"`, `"sum"`, or `None` for no reduction. |

### Methods

| Method | Description |
|--------|-------------|
| [`.forward()`](mseloss/forward.md) | Compute mean squared error between predictions and targets. |
