# Activations

Activation functions introduce non-linearity into a neural network, enabling it to learn complex mappings. Simplegrad provides the most common activations as differentiable functional ops. All of them operate element-wise (except `softmax`, which acts along a chosen axis) and return a new `Tensor` with gradients wired into the computation graph.

```python
import simplegrad as sg
from simplegrad.functions import relu, softmax

x = sg.Tensor([-1.0, 0.5, 2.0], requires_grad=True)
out = relu(x)            # [0.  0.5 2.0]
probs = softmax(x, axis=0)
```

::: simplegrad.functions.activations.relu

::: simplegrad.functions.activations.tanh

::: simplegrad.functions.activations.sigmoid

::: simplegrad.functions.activations.elu

::: simplegrad.functions.activations.gelu

::: simplegrad.functions.activations.softmax
