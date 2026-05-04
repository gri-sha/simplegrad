"""System and device information for benchmark headers."""

import logging
import os
import platform
import subprocess
import sys


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _fmt_cuda_version(v: int) -> str:
    major, rest = divmod(v, 1000)
    minor = rest // 10
    return f"{major}.{minor}"


def _sysctl(key: str) -> str:
    return subprocess.check_output(["sysctl", "-n", key], stderr=subprocess.DEVNULL).decode().strip()


def _cpu_info() -> dict:
    """Return CPU model name and core counts (physical + logical)."""
    system = platform.system()
    model = "unknown"
    physical: int | None = None
    logical = os.cpu_count() or 0

    try:
        if system == "Darwin":
            model = _sysctl("machdep.cpu.brand_string")
            physical = int(_sysctl("hw.physicalcpu"))
        elif system == "Linux":
            with open("/proc/cpuinfo") as f:
                content = f.read()
            for line in content.splitlines():
                if line.startswith("model name") and model == "unknown":
                    model = line.split(":", 1)[1].strip()
                    break
            pkg_core: set[tuple[str, str]] = set()
            curr_pkg = curr_core = None
            for line in content.splitlines():
                if line.startswith("physical id"):
                    curr_pkg = line.split(":", 1)[1].strip()
                elif line.startswith("core id"):
                    curr_core = line.split(":", 1)[1].strip()
                elif not line.strip() and curr_pkg is not None and curr_core is not None:
                    pkg_core.add((curr_pkg, curr_core))
                    curr_pkg = curr_core = None
            if pkg_core:
                physical = len(pkg_core)
        elif system == "Windows":
            out = subprocess.check_output(
                ["wmic", "cpu", "get", "name"], stderr=subprocess.DEVNULL
            ).decode()
            lines = [l.strip() for l in out.splitlines() if l.strip() and l.strip() != "Name"]
            if lines:
                model = lines[0]
    except Exception:
        pass

    return {"model": model, "physical_cores": physical, "logical_cores": logical}


def _ram_gb() -> str:
    """Return total system RAM as a human-readable string (e.g. '32 GB')."""
    system = platform.system()
    try:
        if system == "Darwin":
            return f"{int(_sysctl('hw.memsize')) // (1024 ** 3)} GB"
        elif system == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        return f"{int(line.split()[1]) // (1024 ** 2)} GB"
        elif system == "Windows":
            out = subprocess.check_output(
                ["wmic", "os", "get", "TotalVisibleMemorySize"], stderr=subprocess.DEVNULL
            ).decode()
            lines = [l.strip() for l in out.splitlines() if l.strip() and not l.strip().startswith("Total")]
            if lines:
                return f"{int(lines[0]) // (1024 ** 2)} GB"
    except Exception:
        pass
    return "unknown"


def _metal_gpu_name() -> str | None:
    """Return the Apple GPU model name via ioreg, or None if unavailable."""
    try:
        out = subprocess.check_output(
            ["ioreg", "-r", "-d", "1", "-c", "AGXAccelerator"], stderr=subprocess.DEVNULL
        ).decode()
        for line in out.splitlines():
            if '"model"' in line and "=" in line:
                return line.split("=", 1)[1].strip().strip('"')
    except Exception:
        pass
    return None


def _numpy_version() -> str:
    try:
        import numpy as np
        return np.__version__
    except ImportError:
        return "not installed"


def _torch_info() -> dict:
    try:
        import torch
        cuda_build = torch.version.cuda or "cpu-only"
        return {"version": torch.__version__, "cuda_build": cuda_build}
    except ImportError:
        return {}


def _cupy_info() -> dict:
    try:
        import cupy as cp
        runtime_v = _fmt_cuda_version(cp.cuda.runtime.runtimeGetVersion())
        return {"version": cp.__version__, "cuda_runtime": runtime_v}
    except Exception:
        return {}


def _cuda_devices() -> list[dict]:
    devices = []
    try:
        import cupy as cp
        n = cp.cuda.runtime.getDeviceCount()
        driver_v = _fmt_cuda_version(cp.cuda.runtime.driverGetVersion())
        runtime_v = _fmt_cuda_version(cp.cuda.runtime.runtimeGetVersion())
        for i in range(n):
            props = cp.cuda.runtime.getDeviceProperties(i)
            name = props["name"]
            if isinstance(name, bytes):
                name = name.decode()
            devices.append({
                "index": i,
                "name": name,
                "compute": f"{props['major']}.{props['minor']}",
                "memory": _fmt_bytes(props["totalGlobalMem"]),
                "multiprocessors": props["multiProcessorCount"],
                "clock_mhz": props["clockRate"] // 1000,
                "driver": driver_v,
                "runtime": runtime_v,
            })
    except Exception:
        pass
    return devices


def _mps_available() -> bool:
    try:
        import torch
        return torch.backends.mps.is_available()
    except Exception:
        return False


def collect() -> dict:
    """Return a structured dict of system and device info for JSON benchmark output.

    Assembles the same information as log_system_info into a machine-readable
    dict suitable for embedding in JSON benchmark result files.

    Returns:
        A dict with keys: os, cpu, ram, python, numpy, torch, torch_cuda_build,
        cupy, cuda_devices, mps, metal_gpu. cpu is a dict with model,
        physical_cores, logical_cores. ram is a formatted string (e.g. "32 GB").
        metal_gpu is the Apple GPU name string or None.
    """
    uname = platform.uname()
    torch_info = _torch_info()
    cupy_info = _cupy_info()
    return {
        "os": f"{uname.system} {uname.release} {uname.machine}",
        "cpu": _cpu_info(),
        "ram": _ram_gb(),
        "python": sys.version.split()[0],
        "numpy": _numpy_version(),
        "torch": torch_info.get("version", "not installed"),
        "torch_cuda_build": torch_info.get("cuda_build", ""),
        "cupy": cupy_info.get("version", "not installed"),
        "cuda_devices": _cuda_devices(),
        "mps": _mps_available(),
        "metal_gpu": _metal_gpu_name(),
    }


def log_system_info(log: logging.Logger) -> None:
    """Log system info in the canonical order: software, hardware, GPU."""
    uname = platform.uname()
    cpu = _cpu_info()
    cores = (
        f"{cpu['physical_cores']}p / {cpu['logical_cores']}t"
        if cpu["physical_cores"] is not None
        else f"{cpu['logical_cores']} logical"
    )

    log.info("system")
    log.info("  os      %s %s  %s", uname.system, uname.release, uname.machine)
    log.info("  python  %s", sys.version.split()[0])
    log.info("  numpy   %s", _numpy_version())

    cupy_info = _cupy_info()
    log.info("  cupy    %s", cupy_info.get("version", "not installed"))

    torch_info = _torch_info()
    if torch_info:
        log.info("  torch   %s  (cuda build: %s)", torch_info["version"], torch_info["cuda_build"])
    else:
        log.info("  torch   not installed")

    log.info("")
    log.info("  cpu     %s  (%s)", cpu["model"], cores)
    log.info("  ram     %s", _ram_gb())
    log.info("  mps     %s", "available" if _mps_available() else "not available")

    cuda_devs = _cuda_devices()
    if cuda_devs:
        for d in cuda_devs:
            log.info("")
            log.info("  gpu     %s", d["name"])
            log.info("  vram    %s", d["memory"])
            log.info("  cuda    %s", d["runtime"])
            log.info("  driver  %s", d["driver"])
    else:
        log.info("")
        mps = _mps_available()
        if mps:
            gpu_name = _metal_gpu_name()
            log.info("  gpu     %s", gpu_name or "Apple Metal")
        else:
            log.info("  gpu     none")
