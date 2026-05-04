# Transform Functions

Shape-transform functions rearrange tensor data without changing its values, enabling tensors to flow between layers that expect different shapes. `flatten` collapses one or more dimensions into a single dimension (the typical operation before a `Linear` layer in a CNN), while `reshape` gives full control over the output shape. Both are differentiable — gradients are simply reshaped back during backpropagation.

```python
import simplegrad as sg
from simplegrad.functions.tranform import flatten

x = sg.normal((4, 16, 7, 7), requires_grad=True)  # (N, C, H, W)
out = flatten(x, start_dim=1)                       # (4, 784)
```

::: simplegrad.functions.tranform.flatten

::: simplegrad.functions.tranform.reshape
