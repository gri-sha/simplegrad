# Simplegrad — Project Guide

## Project Overview

Simplegrad is a deep learning framework built around a clean, Pythonic API. It provides numpy-backed tensors with automatic differentiation, a full set of neural network primitives, and tight numpy interoperability. The project also includes **SimpleBoard** — a web application for experiment tracking and visualization.

The framework has an educational objective: users should be able to read the source code and understand how deep learning works from the ground up.

## Repository Structure

```
simplegrad/               # Main package
    core/                 # All base classes and the autograd engine
        autograd.py       # Tensor, Function, Context, lazy mode, tensor ops (_Add, _Mul, ...)
        compound_ops.py   # graph_group, compound_op decorators for computation graph clustering
        dtypes.py         # dtype utilities (as_array, convert_to_dtype, get_dtype_class, ...)
        factory.py        # Tensor factory functions (zeros, ones, normal, uniform, full)
        module.py         # Module base class for all neural network layers
        optimizer.py      # Optimizer base class
        scheduler.py      # Scheduler base class
        __init__.py       # Exports everything from core/
    functions/            # Differentiable math, activations, losses, pooling, conv
                          # All files import from ..core (one-way dependency)
    nn/                   # Neural network layers (Linear, Conv2d, Dropout, Embedding, ...)
    optimizers/           # SGD, Adam (import Optimizer from core)
    schedulers/           # LinearLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateauLR
    track/                # Experiment tracking (Tracker, SQLite backend)
    visual/               # Inline visualization (computation graphs, training plots)
    simpleboard/          # Web app for experiment tracking dashboard

tests/                    # Test suite (pytest)
docs/                     # MkDocs documentation source
    api/                  # Auto-generated API reference pages
examples/                 # Usage examples
experiments/              # Default output directory for tracked runs
assets/                   # Static assets
build_web.py              # Builds the SimpleBoard frontend
pyproject.toml            # Project metadata, dependencies, tool config
mkdocs.yml                # Documentation site configuration
```

### Dependency layers

The package follows a strict one-way import hierarchy:

```
core/          (no intra-simplegrad imports except core/module.py → core/autograd.py)
  ↑
functions/     (import from ..core only)
  ↑
nn/            (import from ..core and ..functions)
  ↑
optimizers/    (import from ..core)
schedulers/    (import from ..core)
```

Never import from a higher layer into a lower one — this causes circular imports.

## Code Style

### Language

All code, comments, variable names, docstrings, and commit messages must be written in **English**. No other languages anywhere in the repository. This is a strict rule.

### Comments

- Do not use separator lines in comments anywhere — this includes source files, test files, and scripts. No `===`, `---`, `———`, `***`, or similar decorators, even as section dividers between test groups.
- Use a plain single-line comment for section labels: `# lazy operator overloads`, not a decorated block.
- Keep comments concise and only where the logic is not self-evident.
- Never delete existing comments when editing a file. If you rewrite or replace a section of code, carry all original comments over into the new version.

### Type Annotations

Use modern Python union syntax throughout — never import `Optional`, `Union`, `List`, `Dict`, `Tuple`, or `Set` from `typing`.

| Instead of | Write |
|---|---|
| `Optional[int]` | `int \| None` |
| `Union[int, float]` | `int \| float` |
| `List[int]` | `list[int]` |
| `Dict[str, int]` | `dict[str, int]` |
| `Tuple[int, ...]` | `tuple[int, ...]` |
| `Set[str]` | `set[str]` |

Only import from `typing` when there is no built-in alternative: `Callable`, `TypeVar`, `Protocol`, `overload`, `TYPE_CHECKING`, etc.

### Docstrings

Every function and method that is part of the public API (callable by a user of the framework) must have a docstring in **Google style**.

Docstrings should be thorough. Since simplegrad has educational goals, explain not just what a function does but also the mathematical or conceptual meaning where relevant. Users reading the source should come away understanding the underlying principles.

Example of the expected style:

```python
def relu(x: Tensor) -> Tensor:
    """Applies the Rectified Linear Unit activation function element-wise.

    Computes max(0, x) for each element. ReLU is the most commonly used
    activation function in deep learning because it avoids the vanishing
    gradient problem present in sigmoid and tanh for large inputs, while
    remaining computationally cheap.

    Args:
        x: Input tensor of any shape.

    Returns:
        A tensor of the same shape as x, with all negative values set to zero.
        Gradients flow through only the positive elements during backpropagation.

    Example:
        >>> x = Tensor([-1.0, 0.0, 2.0])
        >>> relu(x)
        Tensor([0.0, 0.0, 2.0])
    """
```

Internal helper methods (prefixed with `_`) do not require docstrings unless their logic is non-obvious.

## Development

### Python Environment

This project uses a local virtual environment at `.venv/`. Never install packages into the global Python environment.

Always activate the venv before running anything:

```bash
source .venv/bin/activate
```

To reinstall the package inside the venv:

```bash
pip install -e ".[dev]"
```

### Formatting

Run Black before every commit:

```bash
black .
```

CI will fail on unformatted code, so always format locally first.

### Running Tests

Always activate the venv first, then run pytest:

```bash
source .venv/bin/activate
pytest
```

Tests live in `tests/`. Use `pytest tests/test_nn.py` or `pytest tests/test_functions.py` to run a specific file.

### Documentation

Preview docs locally:

```bash
mkdocs serve
```

Docs are deployed automatically to GitHub Pages on every push to `main` via the `docs.yml` workflow. To deploy manually:

```bash
mkdocs gh-deploy --force
```

### SimpleBoard Frontend

The SimpleBoard web app has a compiled frontend. After making changes to `simplegrad/simpleboard/app/`, rebuild it before committing:

```bash
python build_web.py
```

## Internals

### Numpy Backend

All tensor data is stored as a numpy `ndarray` in `tensor.values`. Every operation ultimately calls numpy under the hood. Do not introduce non-numpy code paths (e.g. plain Python lists or math) inside the core or functions packages — always operate on `.values` directly.

### Adding a New Differentiable Operation

All differentiable ops use the `Function` base class from `simplegrad.core.autograd`. Subclass it, implement `forward` and `backward` as static methods, and call `cls.apply(...)` to run the op. The base class handles graph wiring, gradient accumulation, and lazy/eager dispatch automatically.

```python
from simplegrad.core import Tensor, Function, Context
import numpy as np


class _MyOp(Function):
    oper = "MyOp"   # label shown in computation graph — no spaces

    @staticmethod
    def forward(ctx: Context, x: Tensor) -> np.ndarray:
        ctx.x_values = x.values       # save anything needed for backward
        return np.some_numpy_op(x.values)

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * <local_gradient_expression using ctx>


def my_op(x: Tensor) -> Tensor:
    """Public API function. Add a Google-style docstring."""
    return _MyOp.apply(x)
```

Key rules:
- `forward` returns a numpy array and saves intermediates to `ctx`. It must not write to `.grad`.
- `backward` returns one gradient array per Tensor input (or a tuple). It must not accumulate — the base class does that.
- Override `output_shape(*inputs)` if the output shape differs from the first input's shape (required for matmul, conv, reductions, etc.).
- For functions that compose multiple ops (e.g. softmax, mse_loss), use the `@compound_op` decorator instead of subclassing `Function`.
- Place new functional ops in `simplegrad/functions/`. New nn layers go in `simplegrad/nn/`.
