# MaxPool2d

`MaxPool2d` is a `Module` wrapper around the functional `max_pool2d` op. It down-samples spatial feature maps by retaining the maximum activation in each pooling window, reducing computational cost and providing a degree of spatial invariance. It holds no learnable parameters and is typically placed after a `Conv2d` layer in a CNN.

```python
import simplegrad as sg
import simplegrad.nn as nn

pool = nn.MaxPool2d(kernel_size=2, stride=2)
x = sg.normal((1, 16, 28, 28), requires_grad=True)
out = pool(x)  # shape: (1, 16, 14, 14)
```

::: simplegrad.nn.pooling.MaxPool2d
