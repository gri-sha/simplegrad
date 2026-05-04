# Flatten

`Flatten` is a `Module` wrapper that collapses spatial dimensions into a single feature vector, bridging convolutional layers and fully connected layers. It calls the functional `flatten` op under the hood and holds no learnable parameters. Use it inside a `Sequential` model to avoid writing a custom `forward` just for reshaping.

```python
import simplegrad.nn as nn

model = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Flatten(),                # (N, 16, 28, 28) -> (N, 12544)
    nn.Linear(12544, 10),
)
```

::: simplegrad.nn.transform.Flatten
