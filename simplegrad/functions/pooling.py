import numpy as np
from simplegrad.core.tensor import Tensor, _should_compute_grad
from typing import Union
from .conv import pad, _get_rec_fields_from_img, _get_img_from_rec_fields


def max_pool2d(
    x: Tensor,
    kernel_size: Union[int, tuple],
    stride: Union[int, tuple] = None,
    pad_width: Union[int, tuple] = 0,
    pad_mode: str = "constant",
    pad_value: int = 0,
) -> Tensor:
    assert x.values.ndim == 4 or x.values.ndim == 3, "Input tensor must be 4D (batch, channels, H, W) or 3D (channels, H, W)"

    kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    sh, sw = (stride, stride) if isinstance(stride, int) else stride

    # Handle padding
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
            raise ValueError("pad_width must be an int or tuple of 4 ints")
        padded_input = pad(x=x, width=pad_width_np, mode=pad_mode, value=pad_value)

    batch_size = padded_input.values.shape[0] if padded_input.values.ndim == 4 else 1
    channels = padded_input.values.shape[-3]
    in_h, in_w = padded_input.values.shape[-2:]
    out_h = (in_h - kh) // sh + 1
    out_w = (in_w - kw) // sw + 1

    # Get receptive fields: (batch, channels, kh, kw, out_h, out_w)
    rec_fields = _get_rec_fields_from_img(padded_input.values, kh, kw, sh, sw)

    # Reshape for pooling: (batch, channels, kh * kw, out_h, out_w)
    rec_fields_flat = rec_fields.reshape(batch_size, channels, kh * kw, out_h, out_w)

    # Vectorized max over kernel dimension
    out_array = np.max(rec_fields_flat, axis=2)  # (batch, channels, out_h, out_w)

    # Store argmax indices for backward pass
    max_idx = np.argmax(rec_fields_flat, axis=2)  # (batch, channels, out_h, out_w)

    out_tensor = Tensor(out_array)
    out_tensor.prev = {padded_input}
    out_tensor.oper = "max_pool2d"
    out_tensor.comp_grad = _should_compute_grad(padded_input)
    out_tensor.is_leaf = False

    if out_tensor.comp_grad:
        def backward_step():
            if padded_input.comp_grad:
                padded_input._init_grad_if_needed()

                # Create one-hot mask for max positions: (batch, channels, kh * kw, out_h, out_w)
                mask = np.zeros((batch_size, channels, kh * kw, out_h, out_w))

                # Use advanced indexing to set 1s at max positions
                b_idx = np.arange(batch_size)[:, None, None, None]
                c_idx = np.arange(channels)[None, :, None, None]
                h_idx = np.arange(out_h)[None, None, :, None]
                w_idx = np.arange(out_w)[None, None, None, :]

                mask[b_idx, c_idx, max_idx, h_idx, w_idx] = 1.0

                # Multiply mask by output gradient: (batch, channels, kh * kw, out_h, out_w)
                rec_fields_grad = mask * out_tensor.grad[:, :, None, :, :]

                # Reshape back to (batch, channels, kh, kw, out_h, out_w) for _get_img_from_rec_fields
                rec_fields_grad = rec_fields_grad.reshape(batch_size, channels, kh, kw, out_h, out_w)

                # Convert back to image space
                padded_input.grad += _get_img_from_rec_fields(rec_fields_grad, padded_input.values.shape, kh, kw, sh, sw)

        out_tensor.backward_step = backward_step
    return out_tensor
