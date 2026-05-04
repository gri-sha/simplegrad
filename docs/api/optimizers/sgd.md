# SGD

`SGD` (Stochastic Gradient Descent) updates each parameter by subtracting a fixed fraction of its gradient. It optionally supports momentum, which accumulates a velocity vector to dampen oscillations and accelerate convergence along consistent gradient directions. SGD is simpler than Adam and often preferred when training convolutional networks on vision tasks.

```python
import simplegrad as sg
from simplegrad.optimizers import SGD

model = sg.nn.Linear(128, 10)
opt = SGD(lr=0.01, momentum=0.9, model=model)

loss = model(sg.ones((4, 128))).sum()
loss.backward()
opt.step()
opt.zero_grad()
```

::: simplegrad.optimizers.sgd.SGD
