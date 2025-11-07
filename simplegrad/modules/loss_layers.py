from simplegrad.modules import Module
from simplegrad.core.losses import ce_loss, mse_loss


class CrossEntropyLoss(Module):
    def __init__(self, dim=-1, reduction='mean'):
        super().__init__()
        self.dim = dim
        self.reduction = reduction

    def forward(self, z, y):
        return ce_loss(z, y, dim=self.dim, reduction=self.reduction)
    
class MSELoss(Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, p, y):
        return mse_loss(p, y, reduction=self.reduction)