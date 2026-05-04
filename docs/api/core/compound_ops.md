# Compound Ops

Compound ops are higher-level operations that are built by composing other simplegrad primitives rather than calling NumPy directly. The `@compound_op` decorator and `graph_group` context manager tag all tensors created during such a call with a shared group identifier, so the computation graph visualiser can draw a labelled cluster around them for easier inspection.

```python
import simplegrad as sg
from simplegrad.core.compound_ops import compound_op

@compound_op
def my_op(x):
    return (x * x).sum()

y = my_op(sg.Tensor([1.0, 2.0, 3.0], requires_grad=True))
sg.draw(y)  # nodes for my_op are grouped in the graph
```

::: simplegrad.core.compound_ops.compound_op

::: simplegrad.core.compound_ops.graph_group
