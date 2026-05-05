# Adam

`Adam` (Adaptive Moment Estimation) maintains per-parameter first and second moment estimates of the gradients, allowing it to adapt the effective learning rate for each parameter individually. It combines the benefits of momentum and RMSProp and is the most widely used optimiser for training deep networks. The standard defaults (`beta_1=0.9`, `beta_2=0.999`) work well across a broad range of tasks.

```python
import simplegrad as sg
import simplegrad.nn as nn
import simplegrad.optimizers as optim

model = nn.Linear(128, 10)
opt = optim.Adam(lr=1e-3, model=model)

loss = model(sg.ones((4, 128))).sum()
loss.backward()
opt.step()
opt.zero_grad()
```

::: simplegrad.optimizers.adam.Adam
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
| `.param_groups` | `list[dict]` | Parameter groups with `"lr"`, `"beta_1"`, `"beta_2"`, `"eps"`, and `"params"`. |

## Methods

| Method | Description |
|--------|-------------|
| [`.step()`](adam/step.md) | Apply one Adam update step to all parameters. |
| [`.state()`](adam/state.md) | Return the full optimizer state including moment estimates. |

Inherits `.zero_grad()`, `.reset_step_count()`, `.set_param()` from [Optimizer](../core/optimizer.md).
