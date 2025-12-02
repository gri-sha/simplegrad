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
                    slice(pad[0], out.grad.shape[i] - pad[1])
                    for i, pad in enumerate(width)
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


# the idea of the following functions is to organize the convolution operation as a single matrix multiplication
# (read more here: https://cs231n.github.io/convolutional-networks/#conv)
# this allows us to avoid nested loops and speed up the computation significantly


# this function will conver the image input to a matrix rows of wich are the flattened receptive fields
def _get_rec_fields_from_img(img, kh, kw, sh, sw):
    # get dimensions of input image
    batch_size, channels, in_h, in_w = img.shape

    # get output dimensions
    out_h = (in_h - kh) // sh + 1
    out_w = (in_w - kw) // sw + 1

    rec_fields = np.zeros((batch_size, channels, kh, kw, out_h, out_w))

    # here the idea is to mininze the number of loops
    # so we loop over the kernel height and width
    # and get all the values for that kernel neurons positioned at the specific height and width at once
    for h in range(kh):
        h_max = h + sh * out_h
        for w in range(kw):
            w_max = w + sw * out_w
            # reminder: las part of the slice is not included
            rec_fields[:, :, h, w, :, :] = img[:, :, h:h_max:sh, w:w_max:sw]

    # Reshape to (batch_size, channels * kh * kw, out_h * out_w)
    rec_fields = rec_fields.transpose(0, 4, 5, 1, 2, 3).reshape(
        batch_size * out_h * out_w, -1
    )
    return rec_fields


# this function does exactly opposite of get_rec_fields_from_img
# it reconstructs the image from the matrix of receptive fields
def _get_img_from_rec_fields(rec_fields, img_shape, kh, kw, sh, sw):
    batch_size, channels, in_h, in_w = img_shape
    out_h = (in_h - kh) // sh + 1
    out_w = (in_w - kw) // sw + 1

    rec_fields = rec_fields.reshape(
        batch_size, out_h, out_w, channels, kh, kw
    ).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros(img_shape)
    for h in range(kh):
        h_max = h + sh * out_h
        for w in range(kw):
            w_max = w + sw * out_w
            img[:, :, h:h_max:sh, w:w_max:sw] += rec_fields[:, :, h, w, :, :]

    return img


def _convd2d(
    padded_input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[int, tuple] = 1,
):
    batch_size = padded_input.values.shape[0] if padded_input.values.ndim == 4 else 1
    in_h, in_w = padded_input.values.shape[-2:]
    out_channels, in_channels, kh, kw = weight.values.shape
    sh, sw = (stride, stride) if isinstance(stride, int) else stride
    out_h = (in_h - kh) // sh + 1
    out_w = (in_w - kw) // sw + 1

    rec_fields = _get_rec_fields_from_img(
        padded_input.values, kh, kw, sh, sw
    )  # (batch*out_h*out_w, in_channels*kh*kw)
    weight_col = weight.values.reshape(
        out_channels, -1
    ).T  # (in_channels*kh*kw, out_channels)

    # Single matrix multiplication instead of loops!
    out_array = rec_fields @ weight_col  # (batch*out_h*out_w, out_channels)
    out_array = out_array.reshape(batch_size, out_h, out_w, out_channels).transpose(
        0, 3, 1, 2
    )

    if bias is not None:
        out_array += bias.values[:, :, None, None]

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

    # backprop only for _convd2d, for padding the gradient is computed separately (in pad())
    def backward_step():
        # Reshape output gradient
        dout = out_tensor.grad.transpose(0, 2, 3, 1).reshape(-1, out_channels)

        # compute gradiesnt as for usual matrix multiplication
        if weight.comp_grad:
            weight._init_grad_if_needed()
            d_w = rec_fields.T @ dout  # (in_channels*kh*kw, out_channels)
            weight.grad += d_w.T.reshape(weight.values.shape)

        # compute gradiesnt as for usual matrix multiplication
        if padded_input.comp_grad:
            padded_input._init_grad_if_needed()
            d_rec_fields = dout @ weight_col.T
            padded_input.grad += _get_img_from_rec_fields(
                d_rec_fields, padded_input.values.shape, kh, kw, sh, sw
            )

        if bias is not None and bias.comp_grad:
            bias._init_grad_if_needed()
            bias.grad += np.sum(out_tensor.grad, axis=(0, 2, 3))

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


def max_pool2d(
    x: Tensor,
    kernel_size: Union[int, tuple],
    stride: Union[int, tuple] = None,
    # 2 options for padding:
    # int - same dimention for all directions
    # tuple - (top, bottom, left, right)
    pad_width: Union[int, tuple] = 0,
    pad_mode: str = "constant",
    pad_value: int = 0,
) -> Tensor:
    assert (
        x.values.ndim == 4 or x.values.ndim == 3
    ), "Input tensor must be 4-dimensional (batch_size, in_channels, height, width) or 3-dimensional (in_channels, height, width)"
    sh, sw = (stride, stride) if isinstance(stride, int) else stride

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

    in_h, in_w = padded_input.values.shape[-2:]
    kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    out_h = (in_h - kh) // sh + 1
    out_w = (in_w - kw) // sw + 1
    out_array = np.zeros(
        (
            padded_input.values.shape[0] if padded_input.values.ndim == 4 else 1,
            padded_input.values.shape[1],
            out_h,
            out_w,
        )
    )

    mask = np.zeros_like(padded_input.values, dtype=bool)

    for i in range(out_h):
        for j in range(out_w):
            h_start = i * sh
            h_end = h_start + kh
            w_start = j * sw
            w_end = w_start + kw
            input_slice = padded_input.values[:, :, h_start:h_end, w_start:w_end]

            # Get max value
            out_array[:, :, i, j] = np.max(input_slice, axis=(-1, -2))

            # Create mask for max positions
            max_val = out_array[:, :, i, j][
                :, :, None, None
            ]  # make 2d array of shape (B, C, 1, 1) (for broadcasting)
            mask[:, :, h_start:h_end, w_start:w_end] |= np.isclose(input_slice, max_val)

    out_tensor = Tensor(out_array)
    out_tensor.prev = {padded_input}
    out_tensor.oper = "max_pool2d"
    out_tensor.comp_grad = padded_input.comp_grad
    out_tensor.is_leaf = False

    def backward_step():
        if padded_input.comp_grad:
            padded_input._init_grad_if_needed()
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * sh
                    h_end = h_start + kh
                    w_start = j * sw
                    w_end = w_start + kw
                    # Gradient only goes to max positions
                    padded_input.grad[:, :, h_start:h_end, w_start:w_end] += (
                        mask[:, :, h_start:h_end, w_start:w_end]
                        * out_tensor.grad[:, :, i, j][:, :, None, None]
                    )

    out_tensor.backward_step = backward_step
    return out_tensor
