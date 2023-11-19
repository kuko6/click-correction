import torch
from torch import nn
import torch.nn.functional as F


def dice_coefficient(y_pred, y_true, eps=1e-6):
  """ computes the dice coeff. for each class by summing over the depth, height, and width """

  intersection = torch.sum(y_pred * y_true, dim=[2, 3, 4])
  union = torch.sum(y_pred, dim=[2, 3, 4]) + torch.sum(y_true, dim=[2, 3, 4])
  dice = (2. * intersection + eps) / (union + eps)
  # print(dice.shape)

  return dice.mean()


class DiceLoss(nn.Module):
  def __init__(self):
      super().__init__()

  def forward(self, y_pred, y_true):
    return 1 - dice_coefficient(y_pred, y_true)


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None):
        super(DiceBCELoss, self).__init__()
        self.weight = weight

    def forward(self, y_pred, y_true):  
        dice_loss = 1 - dice_coefficient(y_pred, y_true)
        bce_loss = F.binary_cross_entropy(y_pred, y_true, reduction='mean', weight=self.weight)
        combined_loss = bce_loss + dice_loss
        
        return combined_loss