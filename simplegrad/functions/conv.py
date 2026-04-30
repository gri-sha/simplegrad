"""2D convolution and padding operations with autograd support."""

import numpy as np
from ..core import Tensor, Function, Context


def _pad_output_shape(x_shape: tuple, width) -> tuple:
    """Compute output shape of pad() without executing numpy."""
    if isinstance(width, int):
        return tuple(s + 2 * width for s in x_shape)
    return tuple(s + w[0] + w[1] for s, w in zip(x_shape, width))


def _conv2d_output_shape(
    padded_shape: tuple,
    weight_shape: tuple,
    stride: int | tuple,
) -> tuple:
    """Compute conv2d output shape without executing the convolution."""
    batch_size = padded_shape[0] if len(padded_shape) == 4 else 1
    in_h, in_w = padded_shape[-2], padded_shape[-1]
    out_channels, in_channels, kh, kw = weight_shape
    sh, sw = (stride, stride) if isinstance(stride, int) else stride
    out_h = (in_h - kh) // sh + 1
    out_w = (in_w - kw) // sw + 1
    return (batch_size, out_channels, out_h, out_w)


# check https://numpy.org/doc/stable/reference/generated/numpy.pad.html for mode options
class _Pad(Function):
    @staticmethod
    def output_shape(x: Tensor, width, mode: str, value: int) -> tuple:
        return _pad_output_shape(x.shape, width)

    @staticmethod
    def forward(ctx: Context, x: Tensor, width, mode: str, value: int) -> np.ndarray:
        ctx.width = width
        ctx.mode = mode
        return np.pad(array=x.values, pad_width=width, mode=mode, constant_values=value)

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> np.ndarray:
        if ctx.mode == "constant":
            slices = tuple(slice(p[0], grad_output.shape[i] - p[1]) for i, p in enumerate(ctx.width))
            return grad_output[slices]
        raise NotImplementedError(f"Backward pass for padding mode '{ctx.mode}' is not implemented.")


def pad(x: Tensor, width: int | tuple[int, int, int, int], mode: str = "constant", value: int = 0) -> Tensor:
    """Pad a tensor along its spatial dimensions.

    Args:
        x: Input tensor.
        width: Padding widths. An int or nested tuples as accepted by ``numpy.pad``.
        mode: Padding mode (e.g. ``"constant"``, ``"reflect"``). See numpy.pad docs.
        value: Fill value for ``"constant"`` mode.

    Returns:
        Padded tensor.
    """
    return _Pad.apply(x, width, mode, value, oper=f"pad(width={width}, mode={mode})")


# the idea of the following functions is to organize the convolution operation as a single matrix multiplication
# (read more here: https://cs231n.github.io/convolutional-networks/#conv, https://inria.hal.science/inria-00112631v1/document)
# this allows us to avoid nested loops and speed up the computation significantly
# also these functions are used in max pooling implementation


def _get_rec_fields_from_img(img: np.ndarray, kh: int, kw: int, sh: int, sw: int) -> np.ndarray:
    """
    Extract receptive fields from image using numpy strides (no Python loops).

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

    # get the current strides for the input tensor
    # note: stride - how many bytes to step in memory to move to the next element along each dimension 
    # for example: s_batch - how many bytes to step to move to the next batch in memory
    s_batch, s_channel, s_h, s_w = img.strides
    # this is the output shape we need for output
    strided_shape = (batch_size, channels, kh, kw, out_h, out_w)
    # define the strides for this output shape, we use the original strides to consider the datatypes (since strides are in bytes)
    strided_strides = (s_batch, s_channel, s_h, s_w, s_h * sh, s_w * sw)
    # here we create a view of the input image with the shape of the receptive fields and the appropriate strides (memory layout)
    # notice that:
    # num_elements(rec_fields) >= num_elements(img) if we analyse the shapes of the arrays (one element )
    # but we are not creating copies of some elements yet, for now with this approach we just change the way we access the elements in memory
    rec_fields = np.lib.stride_tricks.as_strided(img, shape=strided_shape, strides=strided_strides)
    # here we actually create the copies of the elements in memory, WITHOUT USING LOOPS
    # required because BLAS (Basic Linear Algebra Subroutines) (matmul) cannot work with non-contiguous strided arrays
    # note: contigous array - an array stored in a single, contiguous block of memory (C order by default in numpy)
    return np.ascontiguousarray(rec_fields)


def _get_img_from_rec_fields(rec_fields: np.ndarray, img_shape: tuple[int, int, int, int], kh: int, kw: int, sh: int, sw: int) -> np.ndarray:
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


class _Conv2dNoPad(Function):
    oper = "conv2d"

    @staticmethod
    def output_shape(padded_input: Tensor, weight: Tensor, bias, stride) -> tuple:
        return _conv2d_output_shape(padded_input.shape, weight.shape, stride)

    @staticmethod
    def forward(ctx: Context, padded_input: Tensor, weight: Tensor, bias, stride) -> np.ndarray:
        batch_size = padded_input.values.shape[0] if padded_input.values.ndim == 4 else 1
        out_channels, in_channels, kh, kw = weight.values.shape
        sh, sw = (stride, stride) if isinstance(stride, int) else stride
        out_shape = _conv2d_output_shape(padded_input.shape, weight.shape, stride)
        out_h, out_w = out_shape[-2], out_shape[-1]

        rec_fields = _get_rec_fields_from_img(padded_input.values, kh, kw, sh, sw)
        ctx.rec_fields_flat = rec_fields.transpose(0, 4, 5, 1, 2, 3).reshape(batch_size * out_h * out_w, -1)
        ctx.weight_flat = weight.values.reshape(out_channels, -1).T
        ctx.kh, ctx.kw, ctx.sh, ctx.sw = kh, kw, sh, sw
        ctx.batch_size = batch_size
        ctx.out_h, ctx.out_w = out_h, out_w
        ctx.in_channels = in_channels
        ctx.out_channels = out_channels
        ctx.padded_input_shape = padded_input.values.shape
        ctx.weight_shape = weight.values.shape
        ctx.has_bias = bias is not None

        out_array = ctx.rec_fields_flat @ ctx.weight_flat
        out_array = out_array.reshape(batch_size, out_h, out_w, out_channels).transpose(0, 3, 1, 2)

        if bias is not None:
            out_array = out_array + bias.values[None, :, None, None]
        return out_array

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> tuple:
        out_grad = grad_output.transpose(0, 2, 3, 1).reshape(-1, ctx.out_channels)

        weight_flat_grad = ctx.rec_fields_flat.T @ out_grad
        grad_weight = weight_flat_grad.T.reshape(ctx.weight_shape)

        rec_fields_grad = out_grad @ ctx.weight_flat.T
        rec_fields_grad = rec_fields_grad.reshape(ctx.batch_size, ctx.out_h, ctx.out_w, ctx.in_channels, ctx.kh, ctx.kw).transpose(
            0, 3, 4, 5, 1, 2
        )
        grad_padded_input = _get_img_from_rec_fields(rec_fields_grad, ctx.padded_input_shape, ctx.kh, ctx.kw, ctx.sh, ctx.sw)

        if ctx.has_bias:
            grad_bias = np.sum(grad_output, axis=(0, 2, 3))
            return grad_padded_input, grad_weight, grad_bias
        return grad_padded_input, grad_weight


def _conv2d_no_pad(
    padded_input: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int | tuple = 1,
):
    """Perform conv2d on an already-padded input (no additional padding applied)."""
    return _Conv2dNoPad.apply(padded_input, weight, bias, stride)


# proper convolution with padding
def conv2d(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int | tuple[int, int] = 1,
    # 2 options for padding:
    # int - same padding for all directions
    # tuple of 4 ints - (top, bottom, left, right)
    pad_width: int | tuple[int, int, int, int] = 0,
    pad_mode: str = "constant",
    pad_value: int = 0,
) -> Tensor:
    """Apply a 2D convolution over the input tensor.

    Implements convolution as a single matrix multiplication (im2col). Supports
    batched inputs (4D) and single-sample inputs (3D).

    Args:
        x: Input tensor of shape ``(batch, in_channels, H, W)`` or
            ``(in_channels, H, W)``.
        weight: Kernel tensor of shape ``(out_channels, in_channels, kH, kW)``.
        bias: Optional bias of shape ``(out_channels,)``.
        stride: Convolution stride. Int or ``(height_stride, width_stride)``.
        pad_width: Padding to apply before convolution. Int (same on all sides)
            or ``(top, bottom, left, right)``.
        pad_mode: Padding mode passed to ``numpy.pad``. Defaults to ``"constant"``.
        pad_value: Fill value for constant padding. Defaults to 0.

    Returns:
        Output tensor of shape ``(batch, out_channels, out_H, out_W)``.
    """
    assert (
        len(x.shape) == 4 or len(x.shape) == 3
    ), "Input tensor must be 4-dimensional (batch_size, in_channels, height, width) or 3-dimensional (in_channels, height, width)"
    assert len(weight.shape) == 4, "Weight tensor must be 4-dimensional"

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
            pad_width_np = (
                (0, 0),
                (0, 0),
                (pad_width[0], pad_width[1]),
                (pad_width[2], pad_width[3]),
            )
        else:
            raise ValueError("pad_width must be an int or a tuple of 4 ints (top, bottom, left, right)")

        padded_input = pad(x=x, width=pad_width_np, mode=pad_mode, value=pad_value)

    return _conv2d_no_pad(padded_input=padded_input, weight=weight, bias=bias, stride=stride)
