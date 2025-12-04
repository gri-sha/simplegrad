import numpy as np
from typing import Optional

# default dtype
_DTYPE = "float32"

# supported dtypes
DTYPES = {
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
}


def set_dtype(dtype: str):
    """Set global default dtype (string key)."""
    global _DTYPE
    if dtype not in DTYPES:
        raise ValueError(f"Unsupported dtype '{dtype}'. Must be one of {list(DTYPES)}.")
    _DTYPE = dtype


def get_global_dtype():
    """Return current dtype string key."""
    return _DTYPE


def get_global_dtype_class():
    """Return current NumPy dtype class."""
    return DTYPES[_DTYPE]


def get_dtype_class(dtype: str):
    """Return NumPy dtype class for given dtype string key."""
    if dtype not in DTYPES:
        raise ValueError(f"Unsupported dtype '{dtype}'. Must be one of {list(DTYPES)}.")
    return DTYPES[dtype]


def as_array(values, dtype=None, **kwargs):
    """Convert values into numpy array with current dtype."""
    return np.array(values, dtype=dtype if dtype is not None else get_global_dtype_class(), **kwargs)


def convert_to_dtype(array: np.ndarray, dtype: Optional[str] = None):
    """Convert a numpy array to the global default dtype."""
    return array.astype(get_global_dtype_class() if dtype is None else DTYPES[dtype])
