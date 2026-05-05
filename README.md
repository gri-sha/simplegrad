![header logo](./docs/images/header.png)

![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-013243?logo=numpy&logoColor=white)
![CuPy](https://img.shields.io/badge/cupy-76B900?logo=nvidia&logoColor=white)
![MkDocs](https://img.shields.io/badge/mkdocs-526CFE?logo=materialformkdocs&logoColor=white)
![pytest](https://img.shields.io/badge/pytest-0A9EDC?logo=pytest&logoColor=white)
![black](https://img.shields.io/badge/black-000000?logo=python&logoColor=white)

**`simplegrad`** is the simplest deep learning framework.

## Install

The package is not yet on PyPI — install from source:

```bash
# CPU only (NumPy backend)
pip install git+https://github.com/simplegrad/simplegrad.git

# GPU support (CuPy — requires CUDA 12.x)
pip install "simplegrad[gpu] @ git+https://github.com/simplegrad/simplegrad.git"
```

The SimpleBoard frontend is compiled from npm sources during installation. Skip it if Node.js is not available:

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

## What's inside

| Module | What it gives you |
|--------|-------------------|
| `core/` | Tensor, autograd engine, lazy execution |
| `functions/` | Math, activations, losses, conv, pooling |
| `nn/` | Linear, Conv2d, Embedding, Dropout, ... |
| `optimizers/` | SGD, Adam |
| `schedulers/` | Linear, Exponential, Cosine, ReduceOnPlateau |
| `track/` | SQLite-backed experiment tracking |
| `simpleboard/` | Web dashboard for tracked runs |

## Docs

Full API reference, architecture walkthrough, and training examples at the [documentation site](https://gri-sha.github.io/simplegrad/).
