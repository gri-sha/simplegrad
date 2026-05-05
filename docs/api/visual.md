# Inline Visualization

Simplegrad includes inline visualisation utilities for Jupyter notebooks. `graph` renders the full computation graph of a tensor using Graphviz, showing each operation as a node and the data-flow edges between them. `plot` and `scatter` produce quick training curves directly in the notebook without needing a separate plotting library.

```python
import simplegrad as sg

x = sg.Tensor([1.0, 2.0], requires_grad=True)
y = (x * x).sum()
sg.graph(y)           # renders the computation graph inline

losses = [0.9, 0.7, 0.5, 0.3]
sg.plot(losses, title="Training loss")
```

## Computation Graph

::: simplegrad.visual.inline_comp_graph.graph

---

## Training Plots

::: simplegrad.visual.inline_training_graphs.plot

::: simplegrad.visual.inline_training_graphs.scatter
