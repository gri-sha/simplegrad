# Activation Layers

Activation layers are `Module` wrappers around the functional activation ops, making it convenient to compose them inside `Sequential` or custom `Module` subclasses. Each layer holds no trainable parameters — it simply calls the corresponding function from `simplegrad.functions.activations` in its `forward` method.

```python
import simplegrad as sg
import simplegrad.nn as nn

model = nn.Sequential(
    nn.Linear(16, 32),
    nn.ReLU(),
    nn.Linear(32, 10),
)
out = model(sg.ones((4, 16)))
```

::: simplegrad.nn.activation_layers.ReLU

::: simplegrad.nn.activation_layers.ELU

::: simplegrad.nn.activation_layers.Tanh

::: simplegrad.nn.activation_layers.Sigmoid

::: simplegrad.nn.activation_layers.GELU

::: simplegrad.nn.activation_layers.Softmax
