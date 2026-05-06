"""MLP benchmark: compare any combination of backends.

Architecture: Linear(in→512) → ReLU → Linear(512→512) → ReLU →
              Linear(512→256) → ReLU → Linear(256→10)

Backends are opt-in — specify at least one:
    --sg-cpu        simplegrad on CPU
    --sg-gpu        simplegrad on CUDA GPU
    --torch-cpu     PyTorch on CPU
    --torch-gpu     PyTorch on CUDA GPU
    --torch-metal   PyTorch on Apple Metal (MPS)

Examples:
    python benchmarks/mlp.py --sg-cpu --torch-cpu
    python benchmarks/mlp.py --sg-cpu --sg-gpu
    python benchmarks/mlp.py --sg-cpu --torch-cpu --n-runs 50

Log file is auto-named with datetime and saved to benchmarks/logs/.
Override with --log-file to use a custom path.
"""

import argparse
import platform
import time

import numpy as np
import torch
import torch.nn as tnn

import simplegrad as sg
from simplegrad.core.devices import default_device

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).parent.parent))

from benchmarks.utils import (
    Backend,
    Config,
    TimingResult,
    add_backend_args,
    default_log_path,
    json_log_path,
    run_suite,
    setup_logging,
    sysinfo,
    time_ms,
    torch_sync,
)


def _build_sg_model(in_features: int) -> sg.nn.Sequential:
    return sg.nn.Sequential(
        sg.nn.Linear(in_features, 512),
        sg.nn.ReLU(),
        sg.nn.Linear(512, 512),
        sg.nn.ReLU(),
        sg.nn.Linear(512, 256),
        sg.nn.ReLU(),
        sg.nn.Linear(256, 10),
    )


def _build_torch_model(in_features: int, dev: torch.device) -> tnn.Sequential:
    return tnn.Sequential(
        tnn.Linear(in_features, 512),
        tnn.ReLU(),
        tnn.Linear(512, 512),
        tnn.ReLU(),
        tnn.Linear(512, 256),
        tnn.ReLU(),
        tnn.Linear(256, 10),
    ).to(dev)


def bench_sg_cpu(*, batch, in_features, n_runs, warmup):
    default_device("cpu")
    x = sg.Tensor(np.random.randn(batch, in_features).astype(np.float32))
    model = _build_sg_model(in_features)

    for _ in range(warmup):
        y = model(x)
        y.zero_grad()
        y.backward()

    fwd_times, bwd_times = [], []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        y = model(x)
        fwd_times.append(time.perf_counter() - t0)

        y.zero_grad()
        t0 = time.perf_counter()
        y.backward()
        bwd_times.append(time.perf_counter() - t0)

    return TimingResult(*time_ms(fwd_times), *time_ms(bwd_times))


def bench_sg_gpu(*, batch, in_features, n_runs, warmup):
    import cupy as cp

    default_device("cuda:0")
    x = sg.Tensor(cp.random.randn(batch, in_features).astype(cp.float32))
    model = _build_sg_model(in_features)

    for _ in range(warmup):
        y = model(x)
        y.zero_grad()
        y.backward()
    cp.cuda.Stream.null.synchronize()

    fwd_times, bwd_times = [], []
    for _ in range(n_runs):
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        y = model(x)
        cp.cuda.Stream.null.synchronize()
        fwd_times.append(time.perf_counter() - t0)

        y.zero_grad()
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        y.backward()
        cp.cuda.Stream.null.synchronize()
        bwd_times.append(time.perf_counter() - t0)

    default_device("cpu")
    return TimingResult(*time_ms(fwd_times), *time_ms(bwd_times))


def bench_torch(*, batch, in_features, n_runs, warmup, device):
    dev = torch.device(device)
    model = _build_torch_model(in_features, dev)
    x = torch.randn(batch, in_features, device=dev, requires_grad=True)

    for _ in range(warmup):
        y = model(x)
        y.sum().backward()
        model.zero_grad()
        x.grad = None
    torch_sync(dev)

    fwd_times, bwd_times = [], []
    for _ in range(n_runs):
        torch_sync(dev)
        t0 = time.perf_counter()
        y = model(x)
        torch_sync(dev)
        fwd_times.append(time.perf_counter() - t0)

        torch_sync(dev)
        t0 = time.perf_counter()
        y.sum().backward()
        torch_sync(dev)
        bwd_times.append(time.perf_counter() - t0)
        model.zero_grad()
        x.grad = None

    return TimingResult(*time_ms(fwd_times), *time_ms(bwd_times))


def _device_label(name: str) -> str:
    try:
        cpu_name = platform.processor() or platform.machine()
        if name == "sg-cpu":
            return f"numpy  cpu  {cpu_name}"
        if name == "sg-gpu":
            import cupy as cp

            props = cp.cuda.runtime.getDeviceProperties(0)
            dev_name = props["name"]
            if isinstance(dev_name, bytes):
                dev_name = dev_name.decode()
            return f"cupy   cuda:0  {dev_name}"
        if name == "torch-cpu":
            return f"torch  cpu  {cpu_name}"
        if name == "torch-gpu":
            return f"torch  cuda:0  {torch.cuda.get_device_name(0)}"
        if name == "torch-metal":
            return "torch  mps  Apple Metal"
    except Exception as exc:
        return f"unknown ({exc})"
    return name


def _build_backends(args) -> list[Backend]:
    entries = []
    if args.sg_cpu:
        entries.append(("sg-cpu", bench_sg_cpu))
    if args.sg_gpu:
        entries.append(("sg-gpu", bench_sg_gpu))
    if args.torch_cpu:
        entries.append(("torch-cpu", lambda **kw: bench_torch(**kw, device="cpu")))
    if args.torch_gpu:
        entries.append(("torch-gpu", lambda **kw: bench_torch(**kw, device="cuda:0")))
    if args.torch_metal:
        entries.append(("torch-metal", lambda **kw: bench_torch(**kw, device="mps")))
    return [Backend(name=n, fn=fn, device_label=_device_label(n)) for n, fn in entries]


def _build_configs() -> list[Config]:
    configs = []
    for in_features in [128, 512, 1024]:
        group = f"in_features={in_features}  fc(→512→512→256→10)"
        for batch in [32, 128, 256]:
            configs.append(
                Config(
                    label=f"batch {batch}",
                    params={"batch": batch, "in_features": in_features},
                    group=group,
                )
            )
    return configs


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_backend_args(parser)
    parser.add_argument("--n-runs", type=int, default=30, metavar="N")
    parser.add_argument("--warmup", type=int, default=5, metavar="N")
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        metavar="PATH",
        help="custom log file path (default: benchmarks/logs/mlp_<datetime>.log)",
    )
    args = parser.parse_args()

    backends = _build_backends(args)
    if not backends:
        parser.error("specify at least one backend (--sg-cpu, --torch-cpu, --torch-metal, ...)")

    log_path = (
        default_log_path("mlp")
        if args.log_file is None
        else __import__("pathlib").Path(args.log_file)
    )
    log = setup_logging("mlp_benchmark", log_path)

    sysinfo.log_system_info(log)
    log.info("")

    run_suite(
        "mlp",
        backends,
        _build_configs(),
        args.n_runs,
        args.warmup,
        log,
        json_path=json_log_path(log_path),
    )


if __name__ == "__main__":
    main()
