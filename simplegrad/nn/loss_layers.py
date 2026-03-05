"""Loss function layers (Module wrappers)."""

from .module import Module
from simplegrad.functions.losses import ce_loss, mse_loss


class CELoss(Module):
    """Cross-entropy loss layer with built-in softmax.

    Args:
        dim: Class dimension. Defaults to -1 (last dim).
        reduction: ``"mean"``, ``"sum"``, or ``None``. Defaults to ``"mean"``.
    """

    def __init__(self, dim=-1, reduction="mean"):
        super().__init__()
        self.dim = dim
        self.reduction = reduction

    def forward(self, z, y):
        """Compute cross-entropy loss.

        Args:
            z: Logits tensor.
            y: Target probability distribution, same shape as ``z``.

        Returns:
            Scalar loss tensor.
        """
        return ce_loss(z, y, dim=self.dim, reduction=self.reduction)

    def __str__(self):
        return f"CELoss(dim={self.dim}, reduction={self.reduction})"


class MSELoss(Module):
    """Mean squared error loss layer.

    Args:
        reduction: ``"mean"``, ``"sum"``, or ``None``. Defaults to ``"mean"``.
    """

    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, p, y):
        """Compute MSE loss.

        Args:
            p: Predictions tensor.
            y: Targets tensor, same shape as ``p``.

        Returns:
            Scalar loss tensor.
        """
        return mse_loss(p, y, reduction=self.reduction)

    def __str__(self):
        return f"MSELoss(reduction={self.reduction})"
