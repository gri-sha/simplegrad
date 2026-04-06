![header logo](./assets/header.png)

![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-backed-013243?logo=numpy&logoColor=white)
![Docs](https://img.shields.io/badge/docs-mkdocs--material-526CFE?logo=materialformkdocs&logoColor=white)
![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)
![Code style](https://img.shields.io/badge/code%20style-black-000000)

**`simplegrad`** is a deep learning framework designed to be read. Every layer, every gradient, every optimizer — written from scratch on top of NumPy so you can trace exactly what happens when a neural network learns.

It's not a toy. It ships a full autograd engine, conv layers, optimizers, schedulers, lazy execution, experiment tracking, and a web dashboard. But the source stays small enough to understand in an afternoon.

## Install

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

## What's inside

| Module | What it gives you |
|---|---|
| `core/` | Tensor, autograd engine, lazy execution |
| `functions/` | Math, activations, losses, conv, pooling |
| `nn/` | Linear, Conv2d, Embedding, Dropout, ... |
| `optimizers/` | SGD, Adam |
| `schedulers/` | Linear, Exponential, Cosine, ReduceOnPlateau |
| `track/` | SQLite-backed experiment tracking |
| `simpleboard/` | Web dashboard for tracked runs |

## Docs

Full API reference, architecture walkthrough, and training examples at the [documentation site](https://gri-sha.github.io/simplegrad/).
