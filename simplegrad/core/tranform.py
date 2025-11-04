from simplegrad.core.tensor import Tensor

def flatten(x):
    out = Tensor(x.values.flatten())
    out.prev = {x}
    out.oper = "Flatten"
    out.comp_grad = x.comp_grad
    out.is_leaf = False

    def backward_step():
        if x.comp_grad:
            x.grad = out.grad.reshape(x.values.shape)

    out.backward_step = backward_step
    return out
    
def reshape(x, shape):
    out = Tensor(x.values.reshape(shape))
    out.prev = {x}
    out.oper = f"Reshape({shape})"
    out.comp_grad = x.comp_grad
    out.is_leaf = False

    def backward_step():
        if x.comp_grad:
            x.grad = out.grad.reshape(x.values.shape)

    out.backward_step = backward_step
    return out