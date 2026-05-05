# Loss Functions

Loss functions measure the discrepancy between model predictions and target labels. `ce_loss` (cross-entropy) is the standard choice for classification — it expects raw logits and integer class indices. `mse_loss` (mean squared error) is the go-to for regression tasks. Both return a scalar `Tensor` whose `.backward()` triggers the gradient computation for the whole network.

```python
import simplegrad as sg

logits = sg.normal((4, 10), requires_grad=True)
targets = sg.Tensor([2, 7, 0, 5])
loss = sg.ce_loss(logits, targets)
loss.backward()
```

## ce_loss

Cross-entropy loss over raw logits. A softmax is applied internally, so do **not** pass pre-softmaxed probabilities.

$$
\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} \log\frac{e^{x_{i,y_i}}}{\sum_j e^{x_{i,j}}}
$$

::: simplegrad.functions.losses.ce_loss

## mse_loss

$$
\mathcal{L} = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2
$$

::: simplegrad.functions.losses.mse_loss
