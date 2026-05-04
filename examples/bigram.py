#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !mkdir -p datasets
# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -O ../datasets/shakespeare.txt

# In[2]:


import simplegrad as sg
import numpy as np

# In[3]:


BATCH_SIZE = 32
BLOCK_SIZE = 8
LEARNING_RATE = 1e-2
MAX_ITERS = 3000
VAL_INTERVAL = 300
VAL_ITERS = 200

# In[4]:


with open("../datasets/shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

print("Vocab. size:", vocab_size)

# In[5]:


s2i = {ch: i for i, ch in enumerate(chars)}
i2s = {i: ch for i, ch in enumerate(chars)}

encode = lambda string: [s2i[ch] for ch in string]
decode = lambda tokens: [i2s[tok] for tok in tokens]

# In[6]:


data = encode(text)
split_idx = int(len(data) * 0.9)
train_data = data[:split_idx]
val_data = data[split_idx:]
print("Train:", len(train_data))
print("Val.:", len(val_data))

# In[7]:


class BigramModel(sg.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_table = sg.nn.Embedding(self.vocab_size, self.vocab_size)

    def forward(self, context):
        return self.embedding_table(context)

    def generate(self, context, max_new_tokens=300):
        res = [context.values.item()]
        current = context
        for _ in range(max_new_tokens):
            next = sg.Tensor(np.random.choice(range(self.vocab_size), size=1, p=sg.softmax(self.forward(current), dim=-1).values)[0], dtype="int8")
            res.append(int(next.values.item()))
            current = next
        return res


model = BigramModel(vocab_size)

# In[8]:


def get_batches(split="train"):
    data = train_data if split == "train" else val_data
    idxs = np.random.randint(low=0, high=len(data) - BLOCK_SIZE, size=(BATCH_SIZE,))
    x = sg.Tensor([data[i : i + BLOCK_SIZE] for i in idxs], dtype="int8")
    y = [data[i + 1 : i + BLOCK_SIZE + 1] for i in idxs]
    y_one_hot = sg.zeros((BATCH_SIZE, BLOCK_SIZE, vocab_size), comp_grad=False)
    for b in range(BATCH_SIZE):
        for t in range(BLOCK_SIZE):
            y_one_hot.values[b, t, y[b][t]] = 1
    return x, y_one_hot


x_example, y_example = get_batches("train")
print("x_example:", x_example.shape)
print("y_example:", y_example.shape)

# In[9]:


def estimate_loss():
    out = {}
    for split in ["train", "val"]:
        losses = np.zeros(VAL_ITERS)
        for i in range(VAL_ITERS):
            x_batch, y_batch = get_batches(split)
            losses[i] = sg.ce_loss(model(x_batch), y_batch).values.item()
        out[split] = losses.mean()
    return out

# In[10]:


print("".join(decode(model.generate(context=sg.Tensor(s2i["T"], dtype="int8"), max_new_tokens=500))))

# In[11]:


optimizer = sg.opt.Adam(model, lr=LEARNING_RATE)

tracker = sg.Tracker()
tracker.set_experiment("bigram_language_model")
tracker.start_run(name="training_run_1")

for i in range(MAX_ITERS):
    if i % VAL_INTERVAL == 0:
        eval_loss = estimate_loss()
        tracker.record("val_loss", eval_loss["val"], i)
        print(eval_loss)
    x_batch, y_batch = get_batches()
    loss = sg.ce_loss(model(x_batch), y_batch)
    tracker.record("train_loss", loss.values.item(), i)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

run_id = tracker.end_run()

# Save the computation graph for a forward pass
x_batch, y_batch = get_batches()
final_loss = sg.ce_loss(model(x_batch), y_batch)
tracker.save_comp_graph(tensor=final_loss, run_id=run_id)

# In[12]:


print(''.join(decode(model.generate(context=sg.Tensor(s2i["T"], dtype="int8"), max_new_tokens=500))))
