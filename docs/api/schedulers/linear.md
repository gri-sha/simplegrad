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

::: simplegrad.schedulers.func_based.LinearLR

::: simplegrad.schedulers.func_based.ExponentialLR

::: simplegrad.schedulers.func_based.CosineAnnealingLR

::: simplegrad.schedulers.metric_based.ReduceLROnPlateauLR
