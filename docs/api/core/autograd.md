# Tensor & Autograd

`Tensor` is the central data structure of simplegrad. It wraps a NumPy array and records operations in a computation graph so that gradients can be computed via reverse-mode automatic differentiation. Every mathematical operation on a `Tensor` returns a new `Tensor` and, when `requires_grad=True`, wires itself into the graph for backpropagation.

```python
import simplegrad as sg

x = sg.Tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x * x).sum()
y.backward()
print(x.grad)  # [2. 4. 6.]
```

::: simplegrad.core.autograd.Tensor

---

## Execution mode

Simplegrad supports two execution modes — **eager** (default) and **lazy**. In eager mode every operation runs NumPy immediately. In lazy mode operations are recorded into a graph and only executed when `.realize()` or `.backward()` is called, enabling whole-graph optimisations. The `no_grad` context manager disables gradient tracking entirely, which is useful during inference to save memory and computation.

```python
import simplegrad as sg

# Disable gradient tracking during inference
with sg.no_grad():
    out = model(x)

# Defer execution until .realize()
with sg.lazy():
    out = model(x)
out.realize()
```

::: simplegrad.core.autograd.no_grad

::: simplegrad.core.autograd.lazy

::: simplegrad.core.autograd.mode

::: simplegrad.core.autograd.is_lazy
