import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceCELoss(nn.Module):
    def __init__(self, smooth=1e-5, weight_ce=0.5):
        super().__init__()
        self.smooth = smooth
        self.ce = nn.CrossEntropyLoss()
        self.weight_ce = weight_ce  # ⬅️ 新增：可调节两种loss权重

    def forward(self, preds, targets):
        ce_loss = self.ce(preds, targets)

        preds_soft = F.softmax(preds, dim=1)
        targets_onehot = F.one_hot(targets, num_classes=preds.shape[1]).permute(0, 3, 1, 2).float()

        intersection = (preds_soft * targets_onehot).sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (
            preds_soft.sum(dim=(2, 3)) + targets_onehot.sum(dim=(2, 3)) + self.smooth
        )
        dice_loss = 1 - dice.mean()

        return self.weight_ce * ce_loss + (1 - self.weight_ce) * dice_loss
