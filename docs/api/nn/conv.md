# Conv2d

`Conv2d` is a 2-D convolutional layer that learns a bank of spatial filters applied to each input channel. It stores a `weight` tensor of shape `(out_channels, in_channels, kH, kW)` and an optional `bias` vector, both initialised with Kaiming uniform initialisation. The layer wraps the functional `conv2d` op and is fully compatible with `Sequential` and any `Optimizer`.

```python
import simplegrad as sg
import simplegrad.nn as nn

conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
x = sg.normal((1, 3, 32, 32))
out = conv(x)  # shape: (1, 16, 32, 32)
```

::: simplegrad.nn.conv.Conv2d
    options:
      members: false
      show_root_heading: true
      heading_level: 2
      docstring_section_style: list

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `.weight` | `Tensor` | Filter bank of shape `(out_channels, in_channels, kH, kW)`. Learnable. |
| `.bias` | `Tensor \| None` | Bias vector of shape `(out_channels,)`. `None` if `use_bias=False`. |
| `.in_channels` | `int` | Number of input channels. |
| `.out_channels` | `int` | Number of output channels (filters). |
| `.kernel_size` | `int \| tuple` | Height and width of the convolution kernel. |
| `.stride` | `int \| tuple` | Step size of the sliding window. |
| `.padding` | `int \| tuple` | Zero-padding added to both spatial dimensions. |

## Methods

| Method | Description |
|--------|-------------|
| [`.forward()`](conv/forward.md) | Apply the convolution to input tensor `x`. |

Inherits all methods from [Module](../core/module.md): `.parameters()`, `.submodules()`, `.to_device()`, `.summary()`, `.set_train_mode()`, `.set_eval_mode()`.
