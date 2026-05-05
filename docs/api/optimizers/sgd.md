# SGD

`SGD` (Stochastic Gradient Descent) updates each parameter by subtracting a fixed fraction of its gradient. It optionally supports momentum, which accumulates a velocity vector to dampen oscillations and accelerate convergence along consistent gradient directions. SGD is simpler than Adam and often preferred when training convolutional networks on vision tasks.

```python
import simplegrad as sg
import simplegrad.nn as nn
import simplegrad.optimizers as optim

model = nn.Linear(128, 10)
opt = optim.SGD(lr=0.01, momentum=0.9, model=model)

loss = model(sg.ones((4, 128))).sum()
loss.backward()
opt.step()
opt.zero_grad()
```

::: simplegrad.optimizers.sgd.SGD
    options:
      members: false
      show_root_heading: true
      heading_level: 2
      docstring_section_style: list

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `.lr` | `float` | Default learning rate. |
| `.step_count` | `int` | Number of optimization steps taken. |
| `.param_groups` | `list[dict]` | Parameter groups with `"lr"`, `"momentum"`, `"dampening"`, and `"params"`. |
| `.velocities` | `dict` | Per-parameter velocity arrays used by the momentum update. |

## Methods

| Method | Description |
|--------|-------------|
| [`.step()`](sgd/step.md) | Apply one SGD update step to all parameters. |
| [`.state()`](sgd/state.md) | Return the full optimizer state including velocities. |

Inherits `.zero_grad()`, `.reset_step_count()`, `.set_param()` from [Optimizer](../core/optimizer.md).
