# Sequential

`Sequential` chains a list of `Module` layers and calls them in order during the forward pass, passing the output of each layer as the input to the next. It is the fastest way to build simple feed-forward architectures without writing a custom `Module` subclass. All nested parameters are discovered automatically and can be passed to an optimizer.

```python
import simplegrad as sg
import simplegrad.nn as nn

model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)
out = model(sg.normal((32, 784)))
```

::: simplegrad.nn.sequential.Sequential
