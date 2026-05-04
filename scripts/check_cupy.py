import cupy
import cupy.cuda


def fmt_cuda_version(v: int) -> str:
    major, rest = divmod(v, 1000)
    minor, patch = divmod(rest, 10)
    return f"{major}.{minor}.{patch}"


def fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


print("=" * 50)
print("CuPy")
print("=" * 50)
print(f"  version          : {cupy.__version__}")

try:
    runtime_v = cupy.cuda.runtime.runtimeGetVersion()
    print(f"  CUDA runtime     : {fmt_cuda_version(runtime_v)}")
except Exception as e:
    print(f"  CUDA runtime     : unavailable ({e})")

try:
    driver_v = cupy.cuda.runtime.driverGetVersion()
    print(f"  CUDA driver      : {fmt_cuda_version(driver_v)}")
except Exception as e:
    print(f"  CUDA driver      : unavailable ({e})")

try:
    cudnn_v = cupy.cuda.cudnn.getVersion()
    major, rest = divmod(cudnn_v, 1000)
    minor, patch = divmod(rest, 100)
    print(f"  cuDNN            : {major}.{minor}.{patch}")
except Exception:
    print(f"  cuDNN            : not available")

try:
    nccl_v = cupy.cuda.nccl.get_version()
    major, rest = divmod(nccl_v, 1000)
    minor, patch = divmod(rest, 100)
    print(f"  NCCL             : {major}.{minor}.{patch}")
except Exception:
    print(f"  NCCL             : not available")

print()
print("=" * 50)
print("Devices")
print("=" * 50)

try:
    n_devices = cupy.cuda.runtime.getDeviceCount()
    print(f"  visible devices  : {n_devices}")

    for i in range(n_devices):
        props = cupy.cuda.runtime.getDeviceProperties(i)
        name = props["name"]
        if isinstance(name, bytes):
            name = name.decode()

        total_mem = props["totalGlobalMem"]
        compute = f"{props['major']}.{props['minor']}"
        mp_count = props["multiProcessorCount"]
        clock_mhz = props["clockRate"] // 1000
        mem_clock_mhz = props["memoryClockRate"] // 1000
        bus_width = props["memoryBusWidth"]
        l2_cache = props["l2CacheSize"]

        with cupy.cuda.Device(i):
            mem = cupy.cuda.runtime.memGetInfo()
            free_mem, total_mem_rt = mem

        print(f"\n  [cuda:{i}] {name}")
        print(f"    compute capability : {compute}")
        print(f"    multiprocessors    : {mp_count}")
        print(f"    core clock         : {clock_mhz} MHz")
        print(f"    memory clock       : {mem_clock_mhz} MHz")
        print(f"    memory bus width   : {bus_width}-bit")
        print(f"    L2 cache           : {fmt_bytes(l2_cache)}")
        print(f"    total memory       : {fmt_bytes(total_mem)}")
        print(f"    free  memory       : {fmt_bytes(free_mem)}")
        print(f"    used  memory       : {fmt_bytes(total_mem - free_mem)}")

except cupy.cuda.runtime.CUDARuntimeError as e:
    print(f"  could not query devices: {e}")

print()
print("=" * 50)
print("Memory Pool (default device)")
print("=" * 50)
try:
    pool = cupy.get_default_memory_pool()
    print(f"  used bytes   : {fmt_bytes(pool.used_bytes())}")
    print(f"  total bytes  : {fmt_bytes(pool.total_bytes())}")
    pinned = cupy.get_default_pinned_memory_pool()
    print(f"  pinned free blocks : {pinned.n_free_blocks()}")
except Exception as e:
    print(f"  unavailable ({e})")

print()
print("=" * 50)
print("Build config")
print("=" * 50)
cupy.show_config()
