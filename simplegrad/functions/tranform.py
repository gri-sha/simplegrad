from simplegrad.core.tensor import Tensor

# both start abd end are flattened
def flatten(x, start_dim=0, end_dim=-1):
    if end_dim < 0:
        end_dim = len(x.values.shape) + end_dim
    if start_dim < 0:
        start_dim = len(x.values.shape) + start_dim

    out = Tensor(
        x.values.reshape(
            *x.values.shape[:start_dim], -1, *x.values.shape[end_dim + 1 :]
        )
    )
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
