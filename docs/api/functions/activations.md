# Activations

Activation functions introduce non-linearity into a neural network, enabling it to learn complex mappings. Simplegrad provides the most common activations as differentiable functional ops. All of them operate element-wise (except `softmax`, which acts along a chosen axis) and return a new `Tensor` with gradients wired into the computation graph.

```python
import simplegrad as sg

x = sg.Tensor([-1.0, 0.5, 2.0], requires_grad=True)
out = sg.relu(x)            # [0.  0.5 2.0]
probs = sg.softmax(x, axis=0)
```

## relu

$$
f(x) = \max(0, x)
$$

::: simplegrad.functions.activations.relu

## tanh

$$
f(x) = \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

::: simplegrad.functions.activations.tanh

## sigmoid

$$
f(x) = \frac{1}{1 + e^{-x}}
$$

::: simplegrad.functions.activations.sigmoid

## elu

$$
f(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha(e^x - 1) & \text{if } x \leq 0 \end{cases}
$$

::: simplegrad.functions.activations.elu

## gelu

$$
f(x) = x \cdot \Phi(x) = \frac{x}{2}\left(1 + \tanh\!\left(\sqrt{\frac{2}{\pi}}\left(x + 0.044715\,x^3\right)\right)\right)
$$

::: simplegrad.functions.activations.gelu

## softmax

$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$

::: simplegrad.functions.activations.softmax
