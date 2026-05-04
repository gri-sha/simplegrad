# Loss Layers

Loss layers are `Module` wrappers around the functional loss ops, making it easy to slot a loss function into a model definition or training loop as a named module. `CELoss` wraps cross-entropy loss for classification and `MSELoss` wraps mean squared error for regression. Neither layer has trainable parameters.

```python
import simplegrad as sg
import simplegrad.nn as nn

criterion = nn.CELoss()
logits = sg.normal((8, 10), requires_grad=True)
targets = sg.Tensor([0, 1, 2, 3, 4, 5, 6, 7])
loss = criterion(logits, targets)
loss.backward()
```

::: simplegrad.nn.loss_layers.CELoss

::: simplegrad.nn.loss_layers.MSELoss
