![simplegrad](images/header.png)

A lightweight deep learning framework with automatic differentiation. Runs on NumPy (CPU) or CuPy (GPU/NVIDIA CUDA).

## Installation

```bash
pip install simplegrad[gpu] @ git+https://github.com/simplegrad/simplegrad.git
```

The SimpleBoard frontend is built automatically from npm sources during installation. If you do not have Node.js installed or want to skip the build, set `SIMPLEGRAD_NO_BUILD_WEB=1`:
```bash
SIMPLEGRAD_NO_BUILD_WEB=1 pip install simplegrad[gpu] @ git+https://github.com/simplegrad/simplegrad.git
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
