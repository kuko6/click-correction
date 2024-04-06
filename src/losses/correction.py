import torch
from torch import nn
import scipy
import numpy as np

from losses.dice import dice_coefficient2d


def _get_weight_map(dims: tuple[int], min_thresh=9, max_thresh=20, inverted=False) -> torch.Tensor:
    """
    Generate weight map around the middle pixel, where the highest value is in the middle .

    Args:
        dims (tuple of ints): dimensions of the map in shape (depth, width, height)
        min_thresh (int): minimal thresh value, lower pixel values are set to zero
        max_thresh (int): maximum thresh value, higher pixel values are set to this value
        inverted (bool): whether to invert the weight map
    Returns:
        Tensor: final weight map
    """
    
    tmp = torch.zeros(dims)
    tmp[:, dims[2] // 2, dims[2] // 2] = 1
    dst = scipy.ndimage.distance_transform_edt(1 - tmp[0])

    if inverted:
        weight_map = (1 - dst) + np.abs(np.min(1 - dst))
    else:
        weight_map = dst
    weight_map[weight_map > max_thresh] = max_thresh
    weight_map[weight_map < min_thresh] = 0

    return torch.as_tensor(weight_map, dtype=torch.float32).unsqueeze(0)


class CorrectionLoss(nn.Module):
    """
    Loss for training the correction network, consisting of two parts:
        - one should be pulling it towards the middle
        - the other should be something like a dice loss, correcting the parts further from the middle
    """

    def __init__(self, dims: tuple, device: str, batch_size: int):
        super().__init__()
        # self.alpha = alpha
        # self.probs = probs
        self.weight_map = _get_weight_map(dims)
        self.weight_map = self.weight_map.to(device)
        # self.weight_map = torch.stack((self.weight_map, self.weight_map))

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # TODO: figure out the weighting

        dice_loss = 1 - dice_coefficient2d(y_pred, y_true)
        # loss = dice_loss * self.weight_map
        # loss = dice_loss + weight_map
        loss = dice_loss

        return loss
