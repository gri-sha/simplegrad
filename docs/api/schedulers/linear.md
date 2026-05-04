# Schedulers

Learning rate schedulers adjust the optimizer's learning rate over the course of training. Simplegrad provides function-based schedules (`LinearLR`, `ExponentialLR`, `CosineAnnealingLR`) that change the rate according to a fixed formula, and a metric-based schedule (`ReduceLROnPlateauLR`) that reduces the rate when a monitored metric stops improving. Call `scheduler.step()` once per epoch after the optimizer step.

```python
import simplegrad as sg
from simplegrad.optimizers import Adam
from simplegrad.schedulers import CosineAnnealingLR

model = sg.nn.Linear(64, 10)
opt = Adam(lr=1e-2, model=model)
scheduler = CosineAnnealingLR(opt, T_max=50)

for epoch in range(50):
    # ... training loop ...
    scheduler.step()
```

::: simplegrad.schedulers.func_based.LinearLR

::: simplegrad.schedulers.func_based.ExponentialLR

::: simplegrad.schedulers.func_based.CosineAnnealingLR

::: simplegrad.schedulers.metric_based.ReduceLROnPlateauLR
