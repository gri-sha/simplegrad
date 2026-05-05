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
