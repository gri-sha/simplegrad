"""Device management and backend dispatch for simplegrad."""

import os
import platform
import re
import numpy as np

try:
    import cupy as cp

    _CUPY_AVAILABLE = True
except ImportError:
    cp = None
    _CUPY_AVAILABLE = False

_DEFAULT_DEVICE: str = "cpu"
_CUDA_DEVICE_RE = re.compile(
    r"^cuda:\d+$"
)  # reusable regex object that matches CUDA device strings


def cuda_is_available(verbose: bool = False) -> bool:
    """Return True if CuPy is installed and at least one CUDA device is visible.

    Checks both that CuPy is importable and that the CUDA runtime reports at
    least one device.

    Args:
        verbose: If True, print a message explaining why CUDA is unavailable
            when it is not.

    Returns:
        True if CUDA is usable, False otherwise.
    """
    if not _CUPY_AVAILABLE:
        if verbose:
            print("CuPy is not installed. Install it to use CUDA devices.")
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except cp.cuda.runtime.CUDARuntimeError:
        if verbose:
            print("No CUDA devices are available.")
        return False


def _cpu_description() -> str:
    """Build a human-readable description for the CPU.

    Reads the processor brand string from OS-specific sources (sysctl on
    macOS, /proc/cpuinfo on Linux, registry on Windows) and appends the
    logical core count. Falls back to ``platform.processor()`` or the
    machine architecture when the OS-specific source is unavailable.
    """
    name: str | None = None
    system = platform.system()

    if system == "Darwin":
        try:
            import subprocess

            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=1,
            )
            name = result.stdout.strip() or None
        except Exception:
            pass
    elif system == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        name = line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass
    elif system == "Windows":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
        except Exception:
            pass

    if not name:
        name = platform.processor() or platform.machine() or "CPU"

    cores = os.cpu_count()
    return f"{name} | {cores} logical cores" if cores else name


def _cuda_device_description(index: int) -> str:
    """Build a human-readable description string for a CUDA device.

    Reads device properties via the CuPy runtime API and formats them as
    ``"<name> | <total VRAM> MB"``. Falls back to a generic label if the
    properties cannot be retrieved.
    """
    try:
        props = cp.cuda.runtime.getDeviceProperties(index)
        name = props["name"]
        if isinstance(name, bytes):
            name = name.decode()
        vram_mb = props["totalGlobalMem"] // (1024 * 1024)
        return f"{name} | {vram_mb} MB"
    except Exception:
        return f"CUDA device {index}"


def available_devices() -> dict[str, str]:
    """Return all devices available on this machine as a ``{id: description}`` dict.

    The key is the device string you pass to ``Tensor(..., device=key)`` or
    :func:`set_default_device`. The value is a human-readable description so
    you can identify which physical device each key refers to — useful when
    multiple GPUs are present.

    Always includes ``"cpu"``. Adds ``"cuda:N"`` entries for each visible
    CUDA device when CuPy is installed and the CUDA runtime is functional.

    Returns:
        Dict mapping device strings to descriptions, e.g.::

            {
                "cpu":    "CPU",
                "cuda:0": "NVIDIA GeForce RTX 4090 | 24576 MB",
                "cuda:1": "NVIDIA A100-SXM4-40GB | 40960 MB",
            }
    """
    devices: dict[str, str] = {"cpu": _cpu_description()}
    if _CUPY_AVAILABLE:
        try:
            n = cp.cuda.runtime.getDeviceCount()
            for i in range(n):
                devices[f"cuda:{i}"] = _cuda_device_description(i)
        except cp.cuda.runtime.CUDARuntimeError:
            pass
    return devices


def get_default_device() -> str:
    """Return the current global default device string."""
    return _DEFAULT_DEVICE


def default_device(device: str) -> None:
    """Set the global default device for new tensor creation.

    After calling this, tensors created without an explicit ``device``
    argument will be placed on the given device.

    Args:
        device: Target device string, e.g. ``"cpu"`` or ``"cuda:0"``.
            Must be a valid device identifier (see :func:`validate_device`).

    Raises:
        ValueError: If the device string is not a valid identifier.
    """
    global _DEFAULT_DEVICE
    validate_device(device)
    _DEFAULT_DEVICE = device


def validate_device(device: str) -> str:
    """Validate and return a device string.

    Accepts ``"cpu"`` or ``"cuda:N"`` where N is a non-negative integer.
    This checks format only — it does not verify that the device is present
    on the current machine.

    Args:
        device: Device string to validate.

    Returns:
        The same device string (unchanged).

    Raises:
        ValueError: If the string is not a valid device identifier.
    """
    if device == "cpu":
        return device
    if _CUDA_DEVICE_RE.match(device):
        return device
    raise ValueError(f"Invalid device '{device}'. Expected 'cpu' or 'cuda:N' (e.g. 'cuda:0').")


def get_backend(device: str):
    """Return the compute backend module for the given device.

    Maps device strings to their corresponding array library: ``"cpu"``
    returns :mod:`numpy`, ``"cuda:N"`` returns :mod:`cupy`. This is the
    single dispatch point used throughout the framework — no other module
    imports cupy directly. Using the returned module as a drop-in
    replacement for numpy is safe because cupy mirrors the numpy API.

    Args:
        device: Device string, e.g. ``"cpu"`` or ``"cuda:0"``.

    Returns:
        The :mod:`numpy` or :mod:`cupy` module.

    Raises:
        RuntimeError: If the device is ``"cuda:N"`` but CuPy is not installed.
        ValueError: If the device string is not recognised.
    """
    if device == "cpu":
        return np
    if _CUDA_DEVICE_RE.match(device):
        if not _CUPY_AVAILABLE:
            raise RuntimeError(
                "CuPy is not installed. Install it to use CUDA devices. "
                "See https://docs.cupy.dev/en/stable/install.html"
            )
        return cp
    raise ValueError(f"Unknown device '{device}'. Expected 'cpu' or 'cuda:N' (e.g. 'cuda:0').")


def validate_same_device(*tensors) -> str:
    """Ensure all tensors share the same device and return that device.

    Called by :meth:`~simplegrad.core.Function.apply` before every
    operation to catch accidental mixed-device computations early, before
    any backend call is made.

    Args:
        *tensors: :class:`~simplegrad.core.Tensor` objects to check.

    Returns:
        The shared device string. Returns the default device when no
        tensors are provided.

    Raises:
        RuntimeError: If tensors are on more than one distinct device.
    """
    if not tensors:
        return get_default_device()
    devices = {t.device for t in tensors}
    if len(devices) > 1:
        raise RuntimeError(
            f"All tensors in an operation must be on the same device, "
            f"but found tensors on: {sorted(devices)}"
        )
    return devices.pop()
