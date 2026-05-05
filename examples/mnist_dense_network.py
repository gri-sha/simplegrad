#!/usr/bin/env python
# coding: utf-8

# In[1]:


import simplegrad as sg
from tqdm import tqdm

# In[2]:


VAL_SPLIT = 0.05
EPOCHS = 24
BATCH_SIZE = 32
LEARNING_RATE = 1e-4

# In[3]:


# !curl -L -o ~/Downloads/mnist-dataset.zip https://www.kaggle.com/api/v1/datasets/download/hojjatk/mnist-dataset
# !mkdir -p ~/Developer/simplegrad/datasets/mnist/
# !unzip ~/Downloads/mnist-dataset.zip -d ~/Developer/simplegrad/datasets/mnist/

# In[4]:


import numpy as np

# In[5]:


np.random.seed(42)

# In[6]:


def parse_mnist(folder_path):
    parse = lambda file_path, offset, dtype: np.frombuffer(
        open(file_path, "rb").read(), dtype=dtype, offset=offset
    )
    x = (
        parse(f"{folder_path}/train-images.idx3-ubyte", 16, np.uint8)
        .reshape(-1, 28, 28)
        .astype("float32")
        / 255.0
    )
    labels = parse(f"{folder_path}/train-labels.idx1-ubyte", 8, np.uint8)
    x_test = (
        parse(f"{folder_path}/t10k-images.idx3-ubyte", 16, np.uint8)
        .reshape(-1, 28, 28)
        .astype("float32")
        / 255.0
    )
    labels_test = parse(f"{folder_path}/t10k-labels.idx1-ubyte", 8, np.uint8)
    return (x, labels), (x_test, labels_test)


def to_one_hot(labels, num_classes=10):
    one_hot = np.zeros((len(labels), num_classes), dtype="float32")
    one_hot[np.arange(len(labels)), labels] = 1.0
    return one_hot


# After parsing MNIST data
(x, labels), (x_test, labels_test) = parse_mnist("datasets/mnist")
x_train = x[: int(len(x) * (1 - VAL_SPLIT))]
labels_train = labels[: int(len(labels) * (1 - VAL_SPLIT))]
x_val = x[int(len(x) * (1 - VAL_SPLIT)) :]
labels_val = labels[int(len(labels) * (1 - VAL_SPLIT)) :]

# Convert labels to one-hot
y_train = to_one_hot(labels_train)
y_val = to_one_hot(labels_val)
y_test = labels_test

print("x train:", x_train.shape, x_train.dtype)
print("y train:", y_train.shape, y_train.dtype)
print("x val:", x_val.shape, x_val.dtype)
print("y val:", y_val.shape, y_val.dtype)
print("x test:", x_test.shape, x_test.dtype)
print("y test:", y_test.shape, y_test.dtype)

# In[7]:


import matplotlib.pyplot as plt

idx = 2211
print("One-hot:", y_train[idx])
print("Label:", np.argmax(y_train[idx]))
plt.imshow(x_train[idx], cmap="gray")

# In[8]:


model = sg.nn.Sequential(
    sg.nn.Linear(28 * 28, 64),
    sg.nn.ReLU(),
    sg.nn.Dropout(0.2),
    sg.nn.Linear(64, 64),
    sg.nn.ReLU(),
    sg.nn.Dropout(0.2),
    sg.nn.Linear(64, 10),
)

model.summary()

# In[9]:


optim = sg.opt.Adam(model, lr=LEARNING_RATE)
loss_fn = sg.nn.CELoss(dim=1, reduction="mean")  # dim=1 for class dimension

# In[10]:


tracker = sg.Tracker()
tracker.set_experiment("mnist_dense_net_v3")

# In[11]:


tracker.start_run(name="my_run_3")
model.set_train_mode()
step = 0
for epoch in tqdm(range(EPOCHS)):
    for i in range(0, len(x_train), BATCH_SIZE):
        x_batch = sg.Tensor(x_train[i : i + BATCH_SIZE].reshape(-1, 28 * 28), dtype="float32")
        y_batch = sg.Tensor(y_train[i : i + BATCH_SIZE], dtype="float32")
        logits = model(x_batch)
        loss = loss_fn(logits, y_batch)
        tracker.record("train_loss", loss.values.item(), optim.step_count)
        optim.zero_grad()
        loss.backward()
        optim.step()

    # Validation
    x_val_tensor = sg.Tensor(x_val.reshape(-1, 28 * 28), dtype="float32", comp_grad=False)
    y_val_tensor = sg.Tensor(y_val, dtype="float32", comp_grad=False)
    val_logits = model(x_val_tensor)
    val_loss = loss_fn(val_logits, y_val_tensor)
    tracker.record("val_loss", val_loss.values.item(), optim.step_count)
id = tracker.end_run()

# In[12]:


sg.vis.plot(tracker.get_results(id), num_cols=1, cell_w=20, cell_h=6)

# In[13]:


with sg.no_grad():
    model.set_eval_mode()
    x = sg.reshape(sg.Tensor(x_test), (-1, 28 * 28)).convert_to("float32", inplace=False)
    predictions = sg.flatten(sg.argmax(sg.softmax(model(x), dim=1), dim=1, dtype="int8"))
    predictions.label = "Predictions"

correct = 0
for i in range(len(y_test)):
    if predictions[i][0] == y_test[i]:  # returns a tuple of value and a grad
        correct += 1

accuracy = correct / len(y_test)
print(f"Predictions: {correct}/{len(y_test)}")
print(f"Accuracy: {accuracy:.2%}")

# In[14]:


sg.vis.graph(predictions)

# In[15]:


tracker.save_comp_graph(tensor=predictions, run_id=id)
