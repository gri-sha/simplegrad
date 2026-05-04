# Dropout

`Dropout` randomly zeroes elements of its input tensor with probability `p` during training, then scales the remaining values by `1/(1-p)` to keep the expected magnitude constant. This regularisation technique prevents co-adaptation of neurons and reduces overfitting. During evaluation (after calling `model.set_eval_mode()`), the layer becomes a pass-through with no masking applied.

```python
import simplegrad as sg

model = sg.nn.Sequential(
    sg.nn.Linear(128, 64),
    sg.nn.ReLU(),
    sg.nn.Dropout(p=0.3),
    sg.nn.Linear(64, 10),
)
out = model(sg.ones((8, 128)))
```

::: simplegrad.nn.dropout.Dropout
