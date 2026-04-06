![simplegrad](images/header.png)

A lightweight deep learning framework built on NumPy with automatic differentiation.

## Installation

```bash
pip install simplegrad
```

## Quick start

```python
import simplegrad as sg

x = sg.Tensor([[1.0, 2.0], [3.0, 4.0]], label="x")
w = sg.Tensor([[0.5], [-0.5]], label="w")

y = sg.mean(x @ w)
y.backward()

print(x.grad)  # d(mean(x @ w)) / dx
print(w.grad)  # d(mean(x @ w)) / dw
```

See the [Introduction](introduction.md) for a full walkthrough of the framework architecture, a training example, and guides to lazy mode and experiment tracking.
