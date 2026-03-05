# simplegrad

A simple DL framework.

## Installation

```bash
pip install simplegrad
```

## Quick start

```python
import simplegrad as sg

# Create leaf tensors
x = sg.Tensor([[1.0, 2.0], [3.0, 4.0]], label="x")
w = sg.Tensor([[0.5], [-0.5]], label="w")

# Forward pass — graph is built automatically
y = sg.mean(x @ w)

# Backward pass — gradients flow to all leaves
y.backward()

print(x.grad)  # d(mean(x @ w)) / dx
print(w.grad)  # d(mean(x @ w)) / dw
```

## Training a model

```python
import simplegrad as sg

model = sg.nn.Sequential(Linear(4, 16), ReLU(), Linear(16, 3))
loss_fn = sg.nn.CELoss()
optimizer = sg.opt.Adam(model, lr=1e-3)

for step in range(100):
    optimizer.zero_grad()
    logits = model(x_train)
    loss = loss_fn(logits, y_train)
    loss.backward()
    optimizer.step()
```

## Features

- **Experiment tracking** — SQLite-backed `Tracker` with metric logging
- **Visualization** — computation graphs and training plots inline and in the web app `simpleboard`
