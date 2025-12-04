from simplegrad.nn import Module
from simplegrad.functions.losses import ce_loss, mse_loss


class CELoss(Module):
    def __init__(self, dim=-1, reduction="mean"):
        super().__init__()
        self.dim = dim
        self.reduction = reduction

    def forward(self, z, y):
        return ce_loss(z, y, dim=self.dim, reduction=self.reduction)

    def __str__(self):
        return f"CELoss(dim={self.dim}, reduction={self.reduction})"


class MSELoss(Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, p, y):
        return mse_loss(p, y, reduction=self.reduction)

    def __str__(self):
        return f"MSELoss(reduction={self.reduction})"
