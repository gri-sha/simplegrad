# Optimizer

`Optimizer` is the abstract base class for all parameter update rules in simplegrad. Concrete subclasses (SGD, Adam) implement `step()` to apply a specific update rule. It accepts either a single `Module` (all parameters form one group) or a list of explicit parameter groups, enabling per-group learning rates and hyperparameters.

```python
import simplegrad as sg
from simplegrad.optimizers import SGD

model = sg.nn.Linear(8, 4)
opt = SGD(lr=0.01, model=model)

out = model(sg.ones((2, 8)))
loss = out.sum()
loss.backward()
opt.step()
opt.zero_grad()
```

::: simplegrad.core.optimizer.Optimizer
