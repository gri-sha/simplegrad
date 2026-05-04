# Reduction Operations

Reduction operations collapse one or more axes of a tensor into a scalar or lower-dimensional tensor. `sum` and `mean` are differentiable and commonly used to produce a scalar loss from a batch of per-sample values. `argmax` and `argmin` return indices rather than values and are therefore not differentiable — they are used for computing accuracy metrics during evaluation.

```python
import simplegrad as sg
from simplegrad.functions.reduction import sum, mean

x = sg.normal((4, 10), requires_grad=True)
loss = mean(x)
loss.backward()

preds = sg.argmax(x, axis=1)  # predicted class indices
```

::: simplegrad.functions.reduction.sum

::: simplegrad.functions.reduction.mean

::: simplegrad.functions.reduction.trace

::: simplegrad.functions.reduction.argmax

::: simplegrad.functions.reduction.argmin
