# Linear

`Linear` applies an affine transformation `y = x @ W + b` to its input. It is the fundamental building block of fully connected networks. The weight matrix is initialised with Kaiming uniform initialisation and the bias is set to zero by default. Both are registered as learnable parameters and updated by any `Optimizer`.

```python
import simplegrad as sg
import simplegrad.nn as nn

fc = nn.Linear(in_features=784, out_features=256)
x = sg.normal((32, 784))
out = fc(x)  # shape: (32, 256)
out.sum().backward()
```

::: simplegrad.nn.linear.Linear
    options:
      members: false
      show_root_heading: true
      heading_level: 2
      docstring_section_style: list

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `.weight` | `Tensor` | Weight matrix of shape `(in_features, out_features)`. Learnable. |
| `.bias` | `Tensor \| None` | Bias vector of shape `(out_features,)`. `None` if `use_bias=False`. |
| `.in_features` | `int` | Number of input features. |
| `.out_features` | `int` | Number of output features. |
| `.use_bias` | `bool` | Whether a bias term is included. |
| `.dtype` | `str` | Data type of the weight and bias tensors. |

## Methods

| Method | Description |
|--------|-------------|
| [`.forward()`](linear/forward.md) | Compute `x @ W + b`. |

Inherits all methods from [Module](../core/module.md): `.parameters()`, `.submodules()`, `.to_device()`, `.summary()`, `.set_train_mode()`, `.set_eval_mode()`.
