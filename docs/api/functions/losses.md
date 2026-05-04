# Loss Functions

Loss functions measure the discrepancy between model predictions and target labels. `ce_loss` (cross-entropy) is the standard choice for classification — it expects raw logits and integer class indices. `mse_loss` (mean squared error) is the go-to for regression tasks. Both return a scalar `Tensor` whose `.backward()` triggers the gradient computation for the whole network.

```python
import simplegrad as sg
from simplegrad.functions.losses import ce_loss, mse_loss

logits = sg.normal((4, 10), requires_grad=True)
targets = sg.Tensor([2, 7, 0, 5])
loss = ce_loss(logits, targets)
loss.backward()
```

::: simplegrad.functions.losses.ce_loss

::: simplegrad.functions.losses.mse_loss
