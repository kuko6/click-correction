import torch
from torch import nn
import torch.nn.functional as F


def dice_coefficient(y_pred: torch.Tensor, y_true: torch.Tensor, eps=1e-6) -> torch.Tensor:
    """Computes the dice coeff. by summing over the height and width."""

    # sum for each image in batch
    intersection = torch.sum(y_pred * y_true, dim=[2, 3, 4])
    union = torch.sum(y_pred + y_true, dim=[2, 3, 4])
    dice = (2.0 * intersection + eps) / (union + eps)

    # mean of the whole batch
    return dice.mean()


def dice_coefficient2d(y_pred: torch.Tensor, y_true: torch.Tensor, eps=1e-6) -> torch.Tensor:
    """Computes the dice coeff. by summing over the depth, height and width."""

    # sum for each volume in batch
    intersection = torch.sum(y_pred * y_true, dim=[2, 3])
    union = torch.sum(y_pred + y_true, dim=[2, 3])
    dice = (2.0 * intersection + eps) / (union + eps)

    # mean of the whole batch
    return dice.mean()


class DiceLoss(nn.Module):
    """
    Simple Dice loss function defined as:
    ```
    dice_loss = 1 - dice_coeff
    ```
    """

    def __init__(self, volumetric=True):
        self.use_3d = volumetric
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        if self.use_3d:
            return 1 - dice_coefficient(y_pred, y_true)
        return 1 - dice_coefficient2d(y_pred, y_true)


class DiceBCELoss(nn.Module):
    """
    BCE Dice loss function defined as combination of Dice and BCE:
    ```
    dice_bce_loss = bce_loss + dice_loss
    ```
    """

    def __init__(self, weight=None):
        super(DiceBCELoss, self).__init__()
        self.weight = weight

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        dice_loss = 1 - dice_coefficient(y_pred, y_true)
        bce_loss = F.binary_cross_entropy(y_pred, y_true, reduction='mean', weight=self.weight)
        combined_loss = bce_loss + dice_loss

        return combined_loss
