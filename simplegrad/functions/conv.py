import numpy as np
from simplegrad.core.tensor import Tensor
from typing import Optional, Union


# check https://numpy.org/doc/stable/reference/generated/numpy.pad.html for mode options
def pad(x: Tensor, width: Union[int, tuple], mode: str = "constant", value: int = 0) -> Tensor:
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
                slices = tuple(
                    # remember: the stop index of slice is not included when using the slice
                    slice(p[0], out.grad.shape[i] - p[1])
                    for i, p in enumerate(width)
                )
                x.grad += out.grad[slices]

        out.backward_step = backward_step
    else:

        def backward_step():
            raise NotImplementedError(f"Backward pass for padding mode '{mode}' is not implemented.")

        out.backward_step = backward_step

    return out


# the idea of the following functions is to organize the convolution operation as a single matrix multiplication
# (read more here: https://cs231n.github.io/convolutional-networks/#conv)
# this allows us to avoid nested loops and speed up the computation significantly
# also these functions are used in max pooling implementation


def _get_rec_fields_from_img(img, kh, kw, sh, sw):
    """
    Extract receptive fields from image using strided slicing.

    Args:
        img: numpy array of shape (batch_size, channels, in_h, in_w)
        kh, kw: kernel height and width
        sh, sw: stride height and width

    Returns:
        rec_fields: numpy array of shape (batch_size, channels, kh, kw, out_h, out_w)
    """
    batch_size, channels, in_h, in_w = img.shape
    out_h = (in_h - kh) // sh + 1
    out_w = (in_w - kw) // sw + 1

    rec_fields = np.zeros((batch_size, channels, kh, kw, out_h, out_w))

    for h in range(kh):
        h_max = h + sh * out_h
        for w in range(kw):
            w_max = w + sw * out_w
            rec_fields[:, :, h, w, :, :] = img[:, :, h:h_max:sh, w:w_max:sw]

    return rec_fields


def _get_img_from_rec_fields(rec_fields, img_shape, kh, kw, sh, sw):
    """
    Reconstruct image from receptive fields (inverse of _get_rec_fields_from_img).
    Used in backward pass to compute gradient w.r.t. input image.

    Args:
        rec_fields: numpy array of shape (batch_size, channels, kh, kw, out_h, out_w)
        img_shape: tuple (batch_size, channels, in_h, in_w)
        kh, kw: kernel height and width
        sh, sw: stride height and width

    Returns:
        img: numpy array of shape (batch_size, channels, in_h, in_w)
    """
    batch_size, channels, in_h, in_w = img_shape
    out_h = (in_h - kh) // sh + 1
    out_w = (in_w - kw) // sw + 1

    # Ensure correct shape
    assert rec_fields.shape == (
        batch_size,
        channels,
        kh,
        kw,
        out_h,
        out_w,
    ), f"rec_fields shape mismatch, expected {(batch_size, channels, kh, kw, out_h, out_w)}, got {rec_fields.shape}"

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

    # Get receptive fields: (batch, in_channels, kh, kw, out_h, out_w)
    rec_fields = _get_rec_fields_from_img(padded_input.values, kh, kw, sh, sw)

    # Reshape for matmul: (batch * out_h * out_w, in_channels * kh * kw)
    rec_fields_flat = rec_fields.transpose(0, 4, 5, 1, 2, 3).reshape(batch_size * out_h * out_w, -1)

    # Weight as column matrix: (in_channels * kh * kw, out_channels)
    weight_flat = weight.values.reshape(out_channels, -1).T

    # Single matrix multiplication, result shape: (batch * out_h * out_w, out_channels)
    out_array = rec_fields_flat @ weight_flat

    # Reshape output: (batch_size, out_channels, out_h, out_w)
    out_array = out_array.reshape(batch_size, out_h, out_w, out_channels).transpose(0, 3, 1, 2)

    if bias is not None:
        out_array += bias.values[:, :, None, None]

    out_tensor = Tensor(out_array)
    out_tensor.prev = {padded_input, weight}
    if bias is not None:
        out_tensor.prev.add(bias)
    out_tensor.oper = "conv2d"
    out_tensor.comp_grad = bool(padded_input.comp_grad or weight.comp_grad or (bias.comp_grad if bias is not None else False))
    out_tensor.is_leaf = False

    def backward_step():
        # Reshape output gradient: (batch * out_h * out_w, out_channels)
        out_grad = out_tensor.grad.transpose(0, 2, 3, 1).reshape(-1, out_channels)

        if weight.comp_grad:
            weight._init_grad_if_needed()
            weight_flat_grad = rec_fields_flat.T @ out_grad  # weight_flat_grad.shape: (in_channels * kh * kw, out_channels)
            weight.grad += weight_flat_grad.T.reshape(weight.values.shape)

        if padded_input.comp_grad:
            padded_input._init_grad_if_needed()
            # (batch * out_h * out_w, in_channels * kh * kw)
            rec_fields_grad = out_grad @ weight_flat.T
            # Reshape back: (batch, in_channels, kh, kw, out_h, out_w)
            rec_fields_grad = rec_fields_grad.reshape(batch_size, out_h, out_w, in_channels, kh, kw).transpose(0, 3, 4, 5, 1, 2)
            padded_input.grad += _get_img_from_rec_fields(rec_fields_grad, padded_input.values.shape, kh, kw, sh, sw)

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
            raise ValueError("pad_width must be an int or a tuple of 4 ints (top, bottom, left, right)")

        padded_input = pad(x=x, width=pad_width_np, mode=pad_mode, value=pad_value)

    out = _convd2d(padded_input=padded_input, weight=weight, bias=bias, stride=stride)
    return out
