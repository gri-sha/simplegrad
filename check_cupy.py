import cupy
print("CuPy version:", cupy.__version__)
print("CUDA runtime version:", cupy.cuda.runtime.runtimeGetVersion())
print("CUDA driver version:", cupy.cuda.runtime.driverGetVersion())