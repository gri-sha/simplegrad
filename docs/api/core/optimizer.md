# Optimizer

`Optimizer` is the abstract base class for all parameter update rules in simplegrad. Concrete subclasses (SGD, Adam) implement `step()` to apply a specific update rule. It accepts either a single `Module` (all parameters form one group) or a list of explicit parameter groups, enabling per-group learning rates and hyperparameters.

```python
import simplegrad as sg
import simplegrad.nn as nn
import simplegrad.optimizers as optim

model = nn.Linear(8, 4)
opt = optim.SGD(lr=0.01, model=model)

out = model(sg.ones((2, 8)))
loss = out.sum()
loss.backward()
opt.step()
opt.zero_grad()
```

::: simplegrad.core.optimizer.Optimizer
    options:
      members: false
      show_root_heading: true
      heading_level: 2
      docstring_section_style: list

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `.lr` | `float` | Default learning rate shared across all groups. |
| `.step_count` | `int` | Number of optimization steps taken so far. |
| `.param_groups` | `list[dict]` | List of parameter group dicts, each containing `"params"`, `"lr"`, and any extra hyperparameters. |

## Methods

| Method | Description |
|--------|-------------|
| [`.zero_grad()`](optimizer/zero_grad.md) | Zero gradients for all parameters across all groups. |
| [`.step()`](optimizer/step.md) | Perform a single optimization step. Implemented by subclasses. |
| [`.reset_step_count()`](optimizer/reset_step_count.md) | Reset the internal step counter to zero. |
| [`.set_param()`](optimizer/set_param.md) | Set a hyperparameter value for one or all parameter groups. |
| [`.state()`](optimizer/state.md) | Return the current optimizer state as a dict. |
