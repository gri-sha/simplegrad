import numpy as np
from simplegrad.core import Module, Tensor, uniform
from simplegrad.functions.conv import conv2d
from typing import Union, Optional


class Conv2d(Module):
    def __init__(
        self,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        kernel_size: Optional[Union[int, tuple]] = None,
        weight: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        use_bias: bool = True,
        dtype: Optional[str] = None,  # use global dtype if None
        stride: int = 1,
        # 2 options for padding:
        # int - same dimention for all directions
        # tuple - (top, bottom, left, right)
        pad_width: Union[int, tuple[int]] = 0,
        pad_mode: str = "constant",
        pad_value: int = 0,
        weight_label: str = "W",
        bias_label: str = "b",
    ):
        super().__init__()
        self.dtype = dtype if dtype is not None else "float32"
        if weight is not None:
            assert weight.values.ndim == 4, "Weight tensor must be 4-dimensional"
            assert isinstance(weight, Tensor), "Weight must be a sg.Tensor"
            self.weight = weight.convert_to_dtype(self.dtype, inplace=False)
            self.in_channels = weight.shape[1]
            self.out_channels = weight.shape[0]
            self.kernel_size = weight.shape[2:]
            self.weight.label = weight_label
        else:
            assert in_channels > 0, "in_channels must be a positive integer"
            self.in_channels = in_channels

            assert out_channels > 0, "out_channels must be a positive integer"
            self.out_channels = out_channels

            assert ((isinstance(kernel_size, int)) and kernel_size > 0) or (
                (isinstance(kernel_size, tuple) and all(k > 0 for k in kernel_size))
            ), "kernel_size must be a positive integer or a tuple of positive integers"
            self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
            weight_shape = (out_channels, in_channels, *self.kernel_size)
            self.weight = uniform(
                shape=weight_shape,
                dtype=self.dtype,
                label=weight_label,
                high=np.sqrt(1 / (in_channels * np.prod(self.kernel_size))),
                low=-np.sqrt(1 / (in_channels * np.prod(self.kernel_size))),
            )

        if use_bias:
            if bias is not None:
                assert bias.shape == (out_channels,), "Bias shape must be (out_channels,), " f"but got {bias.shape} instead."
                self.bias = bias
                self.bias.label = bias_label
            else:
                self.bias = uniform(
                    shape=(out_channels,),
                    dtype=self.dtype,
                    label=weight_label,
                    high=np.sqrt(1 / (in_channels * np.prod(self.kernel_size))),
                    low=-np.sqrt(1 / (in_channels * np.prod(self.kernel_size))),
                )
        else:
            self.bias = None

        assert ((isinstance(stride, int)) and stride > 0) or (
            (isinstance(stride, tuple) and all(s > 0 for s in stride))
        ), "stride must be a positive integer or a tuple of positive integers"
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)

        assert ((isinstance(pad_width, int)) and pad_width >= 0) or (
            isinstance(pad_width, tuple) and len(pad_width) == 4 and all(isinstance(p, int) and p >= 0 for p in pad_width)
        ), "padding must be a non-negative integer or a tuple of 4 non-negative integers"
        self.pad_width = (pad_width, pad_width, pad_width, pad_width) if isinstance(pad_width, int) else pad_width
        self.pad_mode = pad_mode
        self.pad_value = pad_value

    def forward(self, x: Tensor) -> Tensor:
        assert (
            x.values.ndim == 4 or x.values.ndim == 3
        ), "Input tensor must be 4-dimensional (batch_size, in_channels, height, width) or 3-dimensional (in_channels, height, width)"
        assert x.values.shape[-3] == self.in_channels, f"Expected input with {self.in_channels} channels, got {x.values.shape[-3]}"

        return conv2d(
            x=x,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            pad_width=self.pad_width,
            pad_mode=self.pad_mode,
            pad_value=self.pad_value,
        )

    def __str__(self):
        return (
            f"Conv2D(in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.pad_width}, "
            f"bias={'True' if self.bias is not None else 'False'})"
        )
