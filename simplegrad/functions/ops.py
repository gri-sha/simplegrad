import numpy as np
import simplegrad as sg
from simplegrad.core.tensor import Tensor
from typing import Optional, Union
from simplegrad.dtypes import get_dtype_class


def log(x: Tensor) -> Tensor:
    if np.any(x.values <= 0):
        raise ValueError("Log of negative value is undefined")

    out = Tensor(np.log(x.values))
    out.prev = {x}
    out.oper = "log"
    out.comp_grad = x.comp_grad
    out.is_leaf = False

    def backward_step():
        if x.comp_grad:
            x._init_grad_if_needed()
            x.grad += out.grad / x.values

    out.backward_step = backward_step
    return out


def exp(x: Tensor) -> Tensor:
    out = Tensor(np.exp(x.values))
    out.prev = {x}
    out.oper = "exp"
    out.comp_grad = x.comp_grad
    out.is_leaf = False

    def backward_step():
        if x.comp_grad:
            x._init_grad_if_needed()
            x.grad += out.grad * np.exp(x.values)

    out.backward_step = backward_step
    return out


def sin(x: Tensor) -> Tensor:
    out = Tensor(np.sin(x.values))
    out.prev = {x}
    out.oper = "sin"
    out.comp_grad = x.comp_grad
    out.is_leaf = False

    def backward_step():
        if x.comp_grad:
            x._init_grad_if_needed()
            x.grad += out.grad * np.cos(x.values)

    out.backward_step = backward_step
    return out


def cos(x: Tensor) -> Tensor:
    out = Tensor(np.cos(x.values))
    out.prev = {x}
    out.oper = "cos"
    out.comp_grad = x.comp_grad
    out.is_leaf = False

    def backward_step():
        if x.comp_grad:
            x._init_grad_if_needed()
            x.grad += -out.grad * np.sin(x.values)

    out.backward_step = backward_step
    return out


def tan(x: Tensor) -> Tensor:
    out = Tensor(np.tan(x.values))
    out.prev = {x}
    out.oper = "tan"
    out.comp_grad = x.comp_grad
    out.is_leaf = False

    def backward_step():
        if x.comp_grad:
            x._init_grad_if_needed()
            x.grad += out.grad / (np.cos(x.values) ** 2)

    out.backward_step = backward_step
    return out


def sum(x: Tensor, dim: Optional[int] = None) -> Tensor:
    # dim 0: sum columns, resulting in a single row
    # dim 1: sum rows, resulting in a single column
    # etc.
    out = Tensor(np.sum(x.values, axis=dim, keepdims=True))
    out.prev = {x}
    out.oper = f"sum(d={dim})"
    out.comp_grad = x.comp_grad
    out.is_leaf = False

    def backward_step():
        if x.comp_grad:
            x._init_grad_if_needed()
            x.grad += np.ones_like(x.values) * out.grad

    out.backward_step = backward_step
    return out


def trace(x: Tensor) -> Tensor:
    if x.values.ndim != 2 or x.values.shape[0] != x.values.shape[1]:
        raise ValueError("Trace is only defined for square matrices")

    out = Tensor(np.array([[np.trace(x.values)]]))
    out.prev = {x}
    out.oper = "trace"
    out.comp_grad = x.comp_grad
    out.is_leaf = False

    def backward_step():
        if x.comp_grad:
            x._init_grad_if_needed()
            grad_matrix = np.zeros_like(x.values)
            np.fill_diagonal(grad_matrix, out.grad.flatten())
            x.grad += grad_matrix

    out.backward_step = backward_step
    return out


def mean(x: Tensor, dim: Optional[int] = None) -> Tensor:
    if dim is None:
        return sum(x) / x.values.size
    return sum(x, dim=dim) / x.values.shape[dim]


def argmax(x: Tensor, dim: Optional[int] = None, dtype: str = "int32") -> Tensor:
    out = Tensor(np.argmax(x.values, axis=dim), dtype=get_dtype_class(dtype))
    out.prev = {x}
    out.oper = f"argmax(d={dim})"
    out.comp_grad = False
    out.is_leaf = False

    def backward_step():
        raise RuntimeError(
            "argmax is not differentiable and does not support backpropagation"
        )

    out.backward_step = backward_step
    return out


def argmin(x: Tensor, dim: Optional[int] = None, dtype: str = "int32") -> Tensor:
    out = Tensor(np.argmin(x.values, axis=dim), dtype=get_dtype_class(dtype))
    out.prev = {x}
    out.oper = f"argmin(d={dim})"
    out.comp_grad = False
    out.is_leaf = False

    def backward_step():
        raise RuntimeError(
            "argmin is not differentiable and does not support backpropagation"
        )

    out.backward_step = backward_step
    return out


# check https://numpy.org/doc/stable/reference/generated/numpy.pad.html for mode options
def pad(
    x: Tensor, width: Union[int, tuple], mode: str = "constant", value: int = 0
) -> Tensor:
    out = Tensor(
        np.pad(
            array=x.values,
            pad_width=width,
            mode=mode,
            constant_values=value,
        )
    )
    out.prev = {x}
    out.oper = f"pad(width={width}, mode={mode})"
    out.comp_grad = x.comp_grad
    out.is_leaf = False

    if mode == "constant":

        def backward_step():
            if x.comp_grad:
                x._init_grad_if_needed()
                grad = np.zeros_like(x.values)
                for pad in width:
                    print(pad)
                slices = tuple(
                    # remember: the stop index of slice is not included when using the slice
                    slice(pad[0], out.grad.shape[i] - pad[1]) for i, pad in enumerate(width)
                )
                print(grad.shape, out.grad.shape, slices)
                print(out.grad[slices].shape)
                grad += out.grad[slices]
                x.grad += grad

        out.backward_step = backward_step
    else:

        def backward_step():
            raise NotImplementedError(
                f"Backward pass for padding mode '{mode}' is not implemented."
            )

        out.backward_step = backward_step

    return out


# convolution without padding
def _convd2d(
    padded_input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[int, tuple] = 1,
):
    in_h, in_w = padded_input.values.shape[-2:]
    sh, sw = (stride, stride) if isinstance(stride, int) else stride
    out_h = (in_h - weight.values.shape[-2]) // sh + 1
    out_w = (in_w - weight.values.shape[-1]) // sw + 1

    # we can predict output shape here
    out_array = np.zeros(
        (
            padded_input.values.shape[0] if padded_input.values.ndim == 4 else 1,
            weight.values.shape[0],
            out_h,
            out_w,
        )
    )

    for i in range(out_h):
        for j in range(out_w):
            h_start = i * sh
            h_end = h_start + weight.values.shape[-2]
            w_start = j * sw
            w_end = w_start + weight.values.shape[-1]
            input_slice = padded_input.values[:, :, h_start:h_end, w_start:w_end]
            for k in range(weight.values.shape[0]):
                # the convolution operation is done for k-th channel on the whole input batch
                out_array[:, k, i, j] = np.sum(
                    input_slice * weight.values[k, :, :, :], axis=(1, 2, 3)
                )  # sum over in_channels, kernel_height, kernel_width

    if bias is not None:
        assert bias.shape == (
            1,
            weight.values.shape[0],
        ), f"Bias shape mismatch: expected (1, out_channels), got {bias.shape}"
        out_array += bias.values[:, :, None, None]  # Reshape from (1, C) to (1, C, 1, 1) for broadcasting

    out_tensor = Tensor(out_array)
    out_tensor.prev = {padded_input, weight}
    if bias is not None:
        out_tensor.prev.add(bias)
    out_tensor.oper = "conv2d"
    out_tensor.comp_grad = bool(
        padded_input.comp_grad
        or weight.comp_grad
        or (bias.comp_grad if bias is not None else False)
    )
    out_tensor.is_leaf = False

    # backprop only for convolution operation (for padding we have separate function)
    def backward_step():
        if padded_input.comp_grad:
            padded_input._init_grad_if_needed()
            # Gradient computation for input
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * sh
                    h_end = h_start + weight.values.shape[-2]
                    w_start = j * sw
                    w_end = w_start + weight.values.shape[-1]
                    for k in range(weight.values.shape[0]):
                        padded_input.grad[
                            :,
                            :,
                            h_start:h_end,
                            w_start:w_end,
                        ] += (
                            # Add 3 dimensions
                            out_tensor.grad[:, k, i, j][:, None, None, None]
                            # Add 1 dimension
                            * weight.values[k, :, :, :][None, :, :, :]
                        )

        if weight.comp_grad:
            weight._init_grad_if_needed()
            # Gradient computation for weights
            for k in range(weight.values.shape[0]):
                for c in range(weight.values.shape[1]):
                    for i in range(weight.values.shape[2]):
                        for j in range(weight.values.shape[3]):
                            weight.grad[k, c, i, j] += np.sum(
                                padded_input.values[
                                    :,
                                    c,
                                    i : i + out_h * sh : sh,
                                    j : j + out_w * sw : sw,
                                ]
                                * out_tensor.grad[:, k, :, :],
                                axis=(0, 1, 2)
                            )

        if bias is not None and bias.comp_grad:
            bias._init_grad_if_needed()
            bias.grad += np.sum(out_tensor.grad, axis=(0, -1, -2))
    out_tensor.backward_step = backward_step
    return out_tensor


# proper convolution with padding
def conv2d(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[int, tuple] = 1,
    # 2 options for padding:
    # int - same padding for all directions
    # tuple of 4 ints - (top, bottom, left, right)
    pad_width: Union[int, tuple] = 0,
    pad_mode: str = "constant",
    pad_value: int = 0,
):
    assert (
        x.values.ndim == 4 or x.values.ndim == 3
    ), "Input tensor must be 4-dimensional (batch_size, in_channels, height, width) or 3-dimensional (in_channels, height, width)"
    assert weight.values.ndim == 4, "Weight tensor must be 4-dimensional"

    # deal with pad_width transformation
    if pad_width == 0 or pad_width == (0, 0, 0, 0):
        padded_input = x
    else:
        if isinstance(pad_width, int):
            pad_width_np = (
                (0, 0),
                (0, 0),
                (pad_width, pad_width),
                (pad_width, pad_width),
            )
        elif isinstance(pad_width, tuple) and len(pad_width) == 4:
            # (top, bottom, left, right)
            pad_width_np = (
                (0, 0),
                (0, 0),
                (pad_width[0], pad_width[1]),
                (pad_width[2], pad_width[3]),
            )
        else:
            raise ValueError(
                "pad_width must be an int or a tuple of 4 ints (top, bottom, left, right)"
            )

        padded_input = pad(x=x, width=pad_width_np, mode=pad_mode, value=pad_value)

    out = _convd2d(padded_input=padded_input, weight=weight, bias=bias, stride=stride)
    return out
