# Scheduler

`Scheduler` is the abstract base class for learning rate schedules in simplegrad. Concrete subclasses (e.g. `LinearLR`, `CosineAnnealingLR`) implement `step()` to adjust the learning rate of an attached optimizer after each epoch or iteration. Calling `scheduler.step()` once per epoch is the typical usage pattern.

```python
import simplegrad as sg
import simplegrad.nn as nn
import simplegrad.optimizers as optim
import simplegrad.schedulers as schedulers

model = nn.Linear(8, 4)
opt = optim.SGD(lr=0.1, model=model)
scheduler = schedulers.LinearLR(opt, start_factor=1.0, end_factor=0.01, total_iters=10)

for epoch in range(10):
    # ... training loop ...
    scheduler.step()
```

::: simplegrad.core.scheduler.Scheduler
