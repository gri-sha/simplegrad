# Conv2d

`Conv2d` is a 2-D convolutional layer that learns a bank of spatial filters applied to each input channel. It stores a `weight` tensor of shape `(out_channels, in_channels, kH, kW)` and an optional `bias` vector, both initialised with Kaiming uniform initialisation. The layer wraps the functional `conv2d` op and is fully compatible with `Sequential` and any `Optimizer`.

```python
import simplegrad as sg

conv = sg.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
x = sg.normal((1, 3, 32, 32))
out = conv(x)  # shape: (1, 16, 32, 32)
```

::: simplegrad.nn.conv.Conv2d
