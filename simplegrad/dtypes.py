import numpy as np

# default dtype
_DTYPE = "float32"

# supported dtypes
DTYPES = {
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

def get_dtype():
    """Return current dtype string key."""
    return _DTYPE

def get_dtype_class():
    """Return current NumPy dtype class."""
    return DTYPES[_DTYPE]

def as_array(values, dtype=None, **kwargs):
    """Convert values into numpy array with current dtype."""
    return np.array(values, dtype=dtype if dtype is not None else get_dtype_class(), **kwargs)
