"""Dropout regularization layer."""

import numpy as np
from ..core import Tensor, Function, Context, Module


class _DropoutEval(Function):
    oper = "Dropout(0)"

    @staticmethod
    def forward(ctx: Context, x: Tensor) -> np.ndarray:
        return x.values.copy()

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> np.ndarray:
        return grad_output


class _DropoutTrain(Function):
    @staticmethod
    def forward(ctx: Context, x: Tensor, p: float) -> np.ndarray:
        ctx.mask = np.random.rand(*x.values.shape) >= p
        return x.values * ctx.mask

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * ctx.mask


class Dropout(Module):
    """Apply dropout regularization during training.

    During training, randomly zeroes elements of the input tensor with
    probability ``p``. Disabled automatically in evaluation mode.

    Args:
        p: Probability of zeroing each element. Must be in ``[0, 1)``.
            Defaults to 0.5.

    Raises:
        ValueError: If ``p`` is not in ``[0, 1)``.
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        if not 0 <= p < 1:
            raise ValueError("Dropout probability must be in the range [0, 1).")
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        """Apply dropout to the input.

        In eval mode (or when ``p=0``), the input passes through unchanged.
        The random mask is generated at realize time, so this method is
        compatible with lazy execution mode.

        Args:
            x: Input tensor.

        Returns:
            Tensor with random elements zeroed (training) or unchanged (eval).
        """
        if not isinstance(x, Tensor):
            raise TypeError("Input must be a Tensor.")

        if self.eval_mode or self.p == 0:
            return _DropoutEval.apply(x)

        return _DropoutTrain.apply(x, self.p, oper=f"Dropout(p={self.p})")

    def __str__(self):
        return f"Dropout(p={self.p})"
