# Simplegrad — Project Guide

## Project Overview

Simplegrad is the simplest complete deep learning framework you can actually use. It provides tensors with automatic differentiation backed by NumPy (CPU) or CuPy (GPU/NVIDIA CUDA), a full set of neural network primitives, and a clean Pythonic API. The project also includes **SimpleBoard** — a web application for experiment tracking and visualization.

The goal is to be the simplest possible complete DL framework: minimal dependencies, readable source, and a straightforward API that mirrors PyTorch conventions without the complexity.

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
benchmarks/               # Performance benchmarks vs PyTorch (CPU and GPU)
    logs/                 # Saved benchmark output logs
scripts/                  # Developer utility scripts
    check_cupy.py         # Prints CuPy/CUDA device info
tools/                    # Developer utility scripts
    build_simpleboard.py  # Builds the SimpleBoard frontend (local development)
pyproject.toml            # Project metadata, dependencies, tool config
mkdocs.yml                # Documentation site configuration
```

### Dependency rules

`.core` is the foundation. Every other module (`functions`, `nn`, `optimizers`, `schedulers`, `track`, `visual`) imports from `.core`. `simpleboard` is fully standalone and does not import from any other package module. Never create circular imports between modules.

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

Docstrings should be clear and precise. Explain what the function does, the math behind it where relevant, and any non-obvious constraints. Do not include `Example:` sections in docstrings — examples belong in the markdown documentation pages.

Example of the expected style:

```python
def relu(x: Tensor) -> Tensor:
    """Applies the Rectified Linear Unit activation function element-wise.

    Computes max(0, x) for each element. ReLU avoids the vanishing gradient
    problem present in sigmoid and tanh for large inputs while remaining
    computationally cheap.

    Args:
        x: Input tensor of any shape.

    Returns:
        A tensor of the same shape as x, with all negative values set to zero.
        Gradients flow through only the positive elements during backpropagation.
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

The SimpleBoard web app has a pre-built frontend. The compiled output lives in `simplegrad/simpleboard/app/dist/` and is committed to the repository.

**CI rebuilds automatically.** Whenever a push to `main` changes files under `simplegrad/simpleboard/app/src/`, `public/`, or any top-level frontend config file (`package.json`, `vite.config.ts`, `tsconfig*.json`, `index.html`), the `simpleboard.yml` workflow runs `npm ci && npm run build` and commits the updated dist back to `main`.

**Local development.** To preview frontend changes before pushing:

```bash
python tools/build_simpleboard.py
```

`pip install simplegrad` no longer requires Node.js.

### Benchmarks

Performance benchmarks live in `benchmarks/`. They compare simplegrad against PyTorch on CPU and optionally GPU (CUDA or Apple Metal). Logs are saved to `benchmarks/logs/`.

Install benchmark dependencies:

```bash
pip install -e ".[bench]"
```

Run a benchmark:

```bash
python benchmarks/conv_benchmark.py --sg-cpu --torch-cpu
python benchmarks/conv_benchmark.py --sg-cpu --torch-cpu --torch-metal
python benchmarks/conv_benchmark.py --sg-cpu --sg-gpu
python benchmarks/conv_benchmark.py --sg-cpu --torch-cpu --log-file benchmarks/logs/conv.log
```

Backends are always opt-in — at least one flag must be passed or the script exits with an error. Each run writes both a text log (`conv_<datetime>.log`) and a structured JSON file (`conv_<datetime>.json`) to `benchmarks/logs/`.

#### Benchmark dashboard

```bash
python benchmarks/dashboard.py
python benchmarks/dashboard.py --port 8080
```

A standalone web dashboard (`benchmarks/dashboard.py` + `benchmarks/dashboard.html`) served by a stdlib `ThreadingHTTPServer`. Opens the browser automatically. Shows a grouped bar chart of fwd/bwd/total times per backend, with ±std error bars. Loads Chart.js from CDN. Select a run from the sidebar, choose a metric and input config from the dropdowns. System info for each run is shown in a collapsible panel.

Add new benchmark files to `benchmarks/` — one file per operation or module being benchmarked. New benchmarks should import from `benchmarks.utils` to get the shared framework.

#### benchmarks/utils

Shared utilities for all benchmarks:

- `sysinfo.log_system_info(log)` — logs OS, Python, NumPy, PyTorch, CuPy versions and full CUDA/MPS device info.
- `TimingResult` — dataclass holding fwd/bwd mean±std; `.fmt()` returns a single readable line.
- `Backend(name, fn, device_label)` — wraps a benchmark function with its display name and device description. `fn` is called as `fn(**config.params, n_runs=N, warmup=N) -> TimingResult`.
- `Config(label, params, group)` — one benchmark case. Configs with the same `group` string are printed under a shared group header.
- `run_suite(name, backends, configs, n_runs, warmup, log)` — runs all backends against all configs and logs results.
- `add_backend_args(parser)` — adds `--sg-cpu`, `--sg-gpu`, `--torch-cpu`, `--torch-gpu`, `--torch-metal` flags to an argparse parser.
- `setup_logging(logger_name, log_file)` — creates a plain-format logger writing to stdout and a file.
- `default_log_path(benchmark_name)` — returns `benchmarks/logs/<name>_<datetime>.log`.

#### Benchmark logging style

All benchmarks use Python's `logging` module with a plain `%(message)s` formatter (no timestamps, no level prefixes in output). Logging rules:

- Never log separator lines (`===`, `---`, or any repeated character sequences).
- Use indentation to show hierarchy: top-level config at 2 spaces, kernel config at 4 spaces, per-backend result at 6 spaces.
- Separate logical groups with a single `log.info("")` blank line — not a separator string.
- Log to a named logger (`logging.getLogger("benchmark_name")`), not the root logger.
- Optionally write to a file via `--log-file`; use `logging.FileHandler` alongside `StreamHandler`.

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
