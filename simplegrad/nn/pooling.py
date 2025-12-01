from ..core import Module, Tensor
from ..functions.ops import max_pool2d
from typing import Union, Optional


class MaxPool2d(Module):
    def __init__(
        self,
        kernel_size: Union[int, tuple],
        stride: Union[int, tuple],
        # 2 options for padding:
        # int - same dimention for all directions
        # tuple - (top, bottom, left, right)
        pad_width: Union[int, tuple[int]] = 0,
        pad_mode: str = "constant",
        pad_value: int = 0,
    ):
        super().__init__()
        assert ((isinstance(kernel_size, int)) and kernel_size > 0) or (
            (isinstance(kernel_size, tuple) and all(k > 0 for k in kernel_size))
            and len(kernel_size) == 2
        ), "kernel_size must be a positive integer or a tuple of positive integers"
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        assert ((isinstance(stride, int)) and stride > 0) or (
            (isinstance(stride, tuple) and all(s > 0 for s in stride))
            and len(stride) == 2
        ), "stride must be a positive integer or a tuple of positive integers"
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)

        self.pad_width = pad_width
        self.pad_mode = pad_mode
        self.pad_value = pad_value

    def forward(self, x: Tensor) -> Tensor:
        return max_pool2d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            pad_width=self.pad_width,
            pad_mode=self.pad_mode,
            pad_value=self.pad_value,
        )
