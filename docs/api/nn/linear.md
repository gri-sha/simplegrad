# Linear

`Linear` applies an affine transformation `y = x @ W.T + b` to its input. It is the fundamental building block of fully connected networks. The weight matrix is initialised with Kaiming uniform initialisation and the bias is set to zero by default. Both are registered as learnable parameters and updated by any `Optimizer`.

```python
import simplegrad as sg
import simplegrad.nn as nn

fc = nn.Linear(in_features=784, out_features=256)
x = sg.normal((32, 784))
out = fc(x)  # shape: (32, 256)
out.sum().backward()
```

::: simplegrad.nn.linear.Linear
