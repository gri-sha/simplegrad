# .to_device()

::: simplegrad.core.autograd.Tensor.to_device
    options:
      show_root_heading: false
      docstring_section_style: list

## Example

```python
import simplegrad as sg

x = sg.Tensor([1.0, 2.0], device="cpu")
x_gpu = x.to_device("cuda:0")
print(x_gpu.device)  # cuda:0
```
