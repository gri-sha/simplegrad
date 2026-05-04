"""Pooling operations with autograd support."""

from ..core import Tensor, Function, Context
from .conv import pad, _get_rec_fields_from_img, _get_img_from_rec_fields


class _MaxPool2d(Function):
    oper = "max_pool2d"

    @staticmethod
    def output_shape(padded_input: Tensor, kh: int, kw: int, sh: int, sw: int) -> tuple:
        """Compute pooling output shape without executing numpy."""
        batch = padded_input.shape[0] if len(padded_input.shape) == 4 else 1
        channels = padded_input.shape[-3]
        in_h, in_w = padded_input.shape[-2], padded_input.shape[-1]
        out_h = (in_h - kh) // sh + 1
        out_w = (in_w - kw) // sw + 1
        return (batch, channels, out_h, out_w)

    @staticmethod
    def forward(ctx: Context, padded_input: Tensor, kh: int, kw: int, sh: int, sw: int):
        xp = ctx.backend
        out_shape = _MaxPool2d.output_shape(padded_input, kh, kw, sh, sw)
        batch_size = padded_input.values.shape[0] if padded_input.values.ndim == 4 else 1
        channels = padded_input.values.shape[-3]
        out_h, out_w = out_shape[-2], out_shape[-1]

        rec_fields = _get_rec_fields_from_img(padded_input.values, xp, kh, kw, sh, sw)
        rec_fields = xp.ascontiguousarray(rec_fields) # in fact reshape silently make contiguous copy, but we make it explicitly
        ctx.rec_fields_flat = rec_fields.reshape(batch_size, channels, kh * kw, out_h, out_w)
        ctx.max_idx = xp.argmax(ctx.rec_fields_flat, axis=2)
        ctx.padded_input_shape = padded_input.values.shape
        ctx.batch_size = batch_size
        ctx.channels = channels
        ctx.out_h = out_h
        ctx.out_w = out_w
        ctx.kh = kh
        ctx.kw = kw
        ctx.sh = sh
        ctx.sw = sw
        return xp.max(ctx.rec_fields_flat, axis=2)

    @staticmethod
    def backward(ctx: Context, grad_output):
        xp = ctx.backend
        mask = xp.zeros((ctx.batch_size, ctx.channels, ctx.kh * ctx.kw, ctx.out_h, ctx.out_w))
        b_idx = xp.arange(ctx.batch_size)[:, None, None, None]
        c_idx = xp.arange(ctx.channels)[None, :, None, None]
        h_idx = xp.arange(ctx.out_h)[None, None, :, None]
        w_idx = xp.arange(ctx.out_w)[None, None, None, :]
        mask[b_idx, c_idx, ctx.max_idx, h_idx, w_idx] = 1.0

        rec_fields_grad = mask * grad_output[:, :, None, :, :]
        rec_fields_grad = rec_fields_grad.reshape(ctx.batch_size, ctx.channels, ctx.kh, ctx.kw, ctx.out_h, ctx.out_w)
        return _get_img_from_rec_fields(rec_fields_grad, xp, ctx.padded_input_shape, ctx.kh, ctx.kw, ctx.sh, ctx.sw)


def max_pool2d(
    x: Tensor,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] = None,
    pad_width: int | tuple[int, int, int] = 0,
    pad_mode: str = "constant",
    pad_value: int = 0,
) -> Tensor:
    """Apply 2D max pooling over the input tensor.

    Args:
        x: Input tensor of shape ``(batch, channels, H, W)`` or ``(channels, H, W)``.
        kernel_size: Pooling window size. Int or ``(kH, kW)``.
        stride: Step between pooling windows. Int or ``(sH, sW)``. Defaults to
            ``kernel_size`` if not specified.
        pad_width: Padding before pooling. Int (all sides) or ``(top, bottom, left, right)``.
        pad_mode: Padding mode. Defaults to ``"constant"``.
        pad_value: Fill value for constant padding. Defaults to 0.

    Returns:
        Output tensor of shape ``(batch, channels, out_H, out_W)``.
    """
    assert len(x.shape) == 4 or len(x.shape) == 3, "Input tensor must be 4D (batch, channels, H, W) or 3D (channels, H, W)"

    kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    if stride is None:
        sh, sw = kh, kw
    elif isinstance(stride, int):
        sh, sw = stride, stride
    else:
        sh, sw = stride

    if pad_width == 0 or pad_width == (0, 0, 0, 0):
        padded_input = x
    else:
        if isinstance(pad_width, int):
            pad_width_np = ((0, 0), (0, 0), (pad_width, pad_width), (pad_width, pad_width))
        elif isinstance(pad_width, tuple) and len(pad_width) == 4:
            pad_width_np = (
                (0, 0),
                (0, 0),
                (pad_width[0], pad_width[1]),
                (pad_width[2], pad_width[3]),
            )
        else:
            raise ValueError("pad_width must be an int or tuple of 4 ints")
        padded_input = pad(x=x, width=pad_width_np, mode=pad_mode, value=pad_value)

    return _MaxPool2d.apply(padded_input, kh, kw, sh, sw)
