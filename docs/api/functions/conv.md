# Convolution

`conv2d` applies a 2-D cross-correlation between an input feature map and a set of learned filters, which is the core operation in convolutional neural networks. `pad` adds zero-padding around the spatial dimensions of a tensor before the convolution, allowing control over output size. Both operations are fully differentiable and support backpropagation through both input and kernel.

```python
import simplegrad as sg

x = sg.normal((1, 3, 28, 28))                          # (N, C, H, W)
kernel = sg.normal((16, 3, 3, 3), requires_grad=True)  # (out_ch, in_ch, kH, kW)
out = sg.conv2d(x, kernel, stride=1, padding=1)
```

::: simplegrad.functions.conv.pad

::: simplegrad.functions.conv.conv2d
