"""Benchmark framework: Backend, Config, run_suite, logging helpers."""

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from itertools import groupby
from pathlib import Path
from typing import Callable

import numpy as np
import torch


@dataclass
class TimingResult:
    fwd_mean: float
    fwd_std: float
    bwd_mean: float
    bwd_std: float

    @property
    def total_mean(self) -> float:
        return self.fwd_mean + self.bwd_mean

    @property
    def total_std(self) -> float:
        return float(np.sqrt(self.fwd_std**2 + self.bwd_std**2))

    def fmt(self) -> str:
        return (
            f"fwd {self.fwd_mean:.2f}±{self.fwd_std:.2f} ms  "
            f"bwd {self.bwd_mean:.2f}±{self.bwd_std:.2f} ms  "
            f"total {self.total_mean:.2f}±{self.total_std:.2f} ms"
        )


@dataclass
class Backend:
    """A named, device-labelled benchmark function.

    fn is called as ``fn(**config.params, n_runs=N, warmup=N) -> TimingResult``.
    """
    name: str
    fn: Callable[..., TimingResult]
    device_label: str


@dataclass
class Config:
    """A single benchmark configuration passed as kwargs to each backend.

    Configs sharing the same ``group`` string are printed under a common header.
    """
    label: str
    params: dict
    group: str | None = None


def time_ms(times: list[float]) -> tuple[float, float]:
    """Convert a list of second-resolution timings to (mean_ms, std_ms)."""
    arr = np.array(times) * 1000
    return float(arr.mean()), float(arr.std())


def torch_sync(dev: torch.device) -> None:
    """Synchronize the given PyTorch device before a timing boundary."""
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    elif dev.type == "mps":
        torch.mps.synchronize()


def run_suite(
    suite_name: str,
    backends: list[Backend],
    configs: list[Config],
    n_runs: int,
    warmup: int,
    log: logging.Logger,
    json_path: Path | None = None,
) -> None:
    """Run all backends against all configs and log the results.

    Configs that share the same ``group`` value are printed under a shared
    group header. Within each group (or ungrouped), each config gets its own
    label line followed by per-backend timing rows.

    When ``json_path`` is provided, the full results are also written as a
    structured JSON file alongside the text log, suitable for the benchmark
    dashboard.

    Args:
        suite_name: Top-level label logged at the start (e.g. ``"conv2d"``).
        backends: List of :class:`Backend` objects to measure.
        configs: List of :class:`Config` objects defining each benchmark case.
        n_runs: Number of timed iterations per config per backend.
        warmup: Number of warmup iterations run before timing starts.
        log: Logger to write all output to.
        json_path: Optional path to write a structured JSON results file.
    """
    from . import sysinfo as _sysinfo

    log.info(suite_name)
    log.info("  n_runs  %d    warmup  %d", n_runs, warmup)
    log.info("")
    log.info("  backends")
    for b in backends:
        log.info("    %-14s  %s", b.name, b.device_label)

    groups_data: list[dict] = []

    for group_key, group_configs_iter in groupby(configs, key=lambda c: c.group):
        group_configs = list(group_configs_iter)
        log.info("")
        if group_key is not None:
            log.info("  %s", group_key)

        group_dict: dict = {"label": group_key or "", "configs": []}
        groups_data.append(group_dict)

        for cfg in group_configs:
            log.info("")
            indent = "    " if group_key is not None else "  "
            log.info("%s%s", indent, cfg.label)

            result_indent = indent + "  "
            cfg_results: dict = {}
            group_dict["configs"].append({"label": cfg.label, "results": cfg_results})

            for b in backends:
                try:
                    r = b.fn(**cfg.params, n_runs=n_runs, warmup=warmup)
                    log.info("%s%-14s  %s", result_indent, b.name, r.fmt())
                    cfg_results[b.name] = {
                        "fwd_mean": round(r.fwd_mean, 4),
                        "fwd_std":  round(r.fwd_std,  4),
                        "bwd_mean": round(r.bwd_mean, 4),
                        "bwd_std":  round(r.bwd_std,  4),
                    }
                except Exception as exc:
                    log.warning("%s%-14s  skipped: %s", result_indent, b.name, exc)
                    cfg_results[b.name] = None

    if json_path is not None:
        payload = {
            "suite": suite_name,
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "n_runs": n_runs,
            "warmup": warmup,
            "sysinfo": _sysinfo.collect(),
            "backends": [{"name": b.name, "device_label": b.device_label} for b in backends],
            "groups": groups_data,
        }
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2))

    log.info("")
    log.info("done")


def json_log_path(log_path: Path) -> Path:
    """Return the .json sibling path of a .log path.

    Args:
        log_path: Path to the text log file produced by setup_logging.

    Returns:
        Same path with the extension replaced by ``.json``.
    """
    return log_path.with_suffix(".json")


def default_log_path(benchmark_name: str) -> Path:
    """Return an auto-named log path under benchmarks/logs/."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return Path("benchmarks/logs") / f"{benchmark_name}_{timestamp}.log"


def setup_logging(logger_name: str, log_file: Path) -> logging.Logger:
    """Create a logger that writes plain messages to stdout and a file.

    Args:
        logger_name: Name passed to ``logging.getLogger``.
        log_file: Path to the log file. Parent directories are created if needed.

    Returns:
        Configured :class:`logging.Logger`.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(message)s")
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    for handler in [logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)]:
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def add_backend_args(parser) -> None:
    """Add the standard backend selection flags to an argparse parser."""
    parser.add_argument("--sg-cpu", action="store_true", help="simplegrad on CPU")
    parser.add_argument("--sg-gpu", action="store_true", help="simplegrad on CUDA GPU")
    parser.add_argument("--torch-cpu", action="store_true", help="PyTorch on CPU")
    parser.add_argument("--torch-gpu", action="store_true", help="PyTorch on CUDA GPU")
    parser.add_argument("--torch-metal", action="store_true", help="PyTorch on Apple Metal (MPS)")
