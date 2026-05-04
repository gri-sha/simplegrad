# Embedding

`Embedding` maps integer token indices to dense floating-point vectors, acting as a learnable lookup table. It stores a weight matrix of shape `(num_embeddings, embedding_dim)` and indexes into it during the forward pass. This is the standard first layer for NLP models that consume tokenised text.

```python
import simplegrad as sg
import simplegrad.nn as nn

embed = nn.Embedding(num_embeddings=1000, embedding_dim=64)
token_ids = sg.Tensor([4, 17, 3, 99])   # integer indices
out = embed(token_ids)                   # shape: (4, 64)
```

::: simplegrad.nn.embedding.Embedding
