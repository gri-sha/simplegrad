# Dtypes

Simplegrad uses `float32` as its global default dtype, matching PyTorch conventions. The dtype utilities let you inspect and change the global default, convert raw Python or NumPy data into properly typed arrays, and look up NumPy dtype classes by string name. All tensor factory functions and operations respect the global default.

```python
import simplegrad as sg
from simplegrad.core.dtypes import default_dtype, get_default_dtype

default_dtype("float64")
print(get_default_dtype())  # "float64"

x = sg.ones((3, 3))       # now created as float64
```

::: simplegrad.core.dtypes.default_dtype

::: simplegrad.core.dtypes.get_default_dtype

::: simplegrad.core.dtypes.get_default_dtype_class

::: simplegrad.core.dtypes.get_dtype_class

::: simplegrad.core.dtypes.as_array

::: simplegrad.core.dtypes.convert_to_dtype
