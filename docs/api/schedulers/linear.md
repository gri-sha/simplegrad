# Schedulers

Learning rate schedulers adjust the optimizer's learning rate over the course of training. Simplegrad provides function-based schedules (`LinearLR`, `ExponentialLR`, `CosineAnnealingLR`) that change the rate according to a fixed formula, and a metric-based schedule (`ReduceLROnPlateauLR`) that reduces the rate when a monitored metric stops improving. Call `scheduler.step()` once per epoch after the optimizer step.

```python
import simplegrad as sg
import simplegrad.nn as nn
import simplegrad.optimizers as optim
import simplegrad.schedulers as schedulers

model = nn.Linear(64, 10)
opt = optim.Adam(lr=1e-2, model=model)
scheduler = schedulers.CosineAnnealingLR(opt, T_max=50)

for epoch in range(50):
    # ... training loop ...
    scheduler.step()
```

All schedulers inherit from [Scheduler](../core/scheduler.md).

---

## LinearLR

::: simplegrad.schedulers.func_based.LinearLR
    options:
      members: false
      show_root_heading: true
      heading_level: 3
      docstring_section_style: list

### Methods

| Method | Description |
|--------|-------------|
| [`.step()`](linear_lr/step.md) | Linearly interpolate the learning rate for the current step. |

---

## ExponentialLR

::: simplegrad.schedulers.func_based.ExponentialLR
    options:
      members: false
      show_root_heading: true
      heading_level: 3
      docstring_section_style: list

### Methods

| Method | Description |
|--------|-------------|
| [`.step()`](exponential_lr/step.md) | Multiply the learning rate by `gamma` each step. |

---

## CosineAnnealingLR

::: simplegrad.schedulers.func_based.CosineAnnealingLR
    options:
      members: false
      show_root_heading: true
      heading_level: 3
      docstring_section_style: list

### Methods

| Method | Description |
|--------|-------------|
| [`.step()`](cosine_annealing_lr/step.md) | Update the learning rate following a cosine annealing schedule. |

---

## ReduceLROnPlateauLR

::: simplegrad.schedulers.metric_based.ReduceLROnPlateauLR
    options:
      members: false
      show_root_heading: true
      heading_level: 3
      docstring_section_style: list

### Methods

| Method | Description |
|--------|-------------|
| [`.step()`](reduce_lr_on_plateau/step.md) | Check the monitored metric and reduce the learning rate if on a plateau. |
