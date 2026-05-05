![simplegrad](images/header.png)

A lightweight deep learning framework with automatic differentiation. Runs on NumPy (CPU) or CuPy (GPU/NVIDIA CUDA).

## Installation

The package is not yet on PyPI — install from source:

```bash
# CPU only (NumPy backend)
pip install git+https://github.com/simplegrad/simplegrad.git

# GPU support (CuPy — requires CUDA 12.x)
pip install "simplegrad[gpu] @ git+https://github.com/simplegrad/simplegrad.git"
```

The SimpleBoard frontend is compiled from npm sources during installation. If Node.js is not available, skip the build with:

```bash
SIMPLEGRAD_NO_BUILD_WEB=1 pip install git+https://github.com/simplegrad/simplegrad.git
```

### Optional extras

| Extra | Installs | Use when |
|-------|----------|----------|
| `gpu` | `cupy-cuda12x` | Running on an NVIDIA GPU (CUDA 12.x) |
| `bench` | `torch` | Running the benchmarks in `benchmarks/` |
| `dev` | `pytest`, `black`, `mypy`, `ipykernel` | Contributing / running tests |
| `docs` | `mkdocs`, `mkdocs-material`, `mkdocstrings` | Building the documentation locally |

```bash
# Install multiple extras at once
pip install "simplegrad[gpu,dev] @ git+https://github.com/simplegrad/simplegrad.git"
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
