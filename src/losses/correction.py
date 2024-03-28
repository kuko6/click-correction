import torch
from torch import nn
from losses.dice import dice_coefficient2d
from utils import get_glioma_indices, get_weight_map

class CorrectionLoss(nn.Module):
    '''
    Loss for training the correction network, consisting of two parts:
        - one should be pulling it towards the middle
        - the other should be something like a dice loss, correcting the parts further from the middle
    '''
    def __init__(self, cutshape: tuple, device: str, batch_size: int):
        super().__init__()
        # self.alpha = alpha
        # self.probs = probs
        self.weight_map = get_weight_map(cutshape)
        self.weight_map = self.weight_map.to(device)
        # self.weight_map = torch.stack((self.weight_map, self.weight_map))

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # TODO: figure out the weighting 

        
        dice_loss = 1 - dice_coefficient2d(y_pred, y_true)
        # loss = dice_loss * self.weight_map
        # loss = dice_loss + weight_map
        loss = dice_loss

        return loss