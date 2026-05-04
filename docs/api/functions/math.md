# Math Functions

These are element-wise mathematical functions with full gradient support. Each one wraps a NumPy primitive inside a `Function` subclass so that gradients flow correctly through the operation during backpropagation. They are useful when building custom loss functions or architectures that require explicit mathematical transformations.

```python
import simplegrad as sg

x = sg.Tensor([1.0, 2.0, 3.0], requires_grad=True)
y = sg.log(x) + sg.exp(x)
y.sum().backward()
print(x.grad)
```

## log

$$
f(x) = \ln(x)
$$

::: simplegrad.functions.math.log

## exp

$$
f(x) = e^x
$$

::: simplegrad.functions.math.exp

## sin

$$
f(x) = \sin(x)
$$

::: simplegrad.functions.math.sin

## cos

$$
f(x) = \cos(x)
$$

::: simplegrad.functions.math.cos

## tan

$$
f(x) = \tan(x)
$$

::: simplegrad.functions.math.tan
