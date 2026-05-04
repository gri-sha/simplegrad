# Pooling

`max_pool2d` down-samples a spatial feature map by taking the maximum value within each non-overlapping window. This reduces spatial resolution while preserving the strongest activations, making the representation more compact and translation-invariant. During backpropagation, the gradient is routed only to the position that held the maximum in the forward pass.

```python
import simplegrad as sg

x = sg.normal((1, 16, 28, 28), requires_grad=True)
out = sg.max_pool2d(x, kernel_size=2, stride=2)
# out.shape == (1, 16, 14, 14)
```

::: simplegrad.functions.pooling.max_pool2d
