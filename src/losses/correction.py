import numpy as np
import scipy
import torch
from torch import nn

from losses.dice import dice_coefficient


def _get_weight_map(dims: tuple[int], min_thresh=9, max_thresh=20, inverted=False) -> torch.Tensor:
    """
    Generate weight map around the middle pixel (the highest value being in the middle).

    Args:
        dims (tuple of ints): dimensions of the map in shape (depth, width, height)
        min_thresh (int): minimal thresh value, lower pixel values are set to zero
        max_thresh (int): maximum thresh value, higher pixel values are set to this value
        inverted (bool): whether to invert the weight map
    Returns:
        Tensor: final weight map
    """
    
    tmp = torch.zeros(dims)
    if len(dims) == 4:
        tmp[:, :, dims[2] // 2, dims[2] // 2] = 1
    else:
        tmp[:, dims[2] // 2, dims[2] // 2] = 1
    dst = scipy.ndimage.distance_transform_edt(1 - tmp[0])

    if inverted:
        weight_map = (1 - dst) + np.abs(np.min(1 - dst))
    else:
        weight_map = dst
    weight_map[weight_map > max_thresh] = max_thresh
    weight_map[weight_map < min_thresh] = 0

    return torch.as_tensor(weight_map, dtype=torch.float32).unsqueeze(0)


def weighted_coefficient(y_pred: torch.Tensor, y_true: torch.Tensor, weight_map: torch.Tensor, eps=1e-6) -> torch.Tensor:
    """
    Calculates the weighted coefficient for a batch of predicted and true tensors.

    Args:
        y_pred (Tensor): The predicted tensor
        y_true (Tensor): The true tensor
        weight_map (Tensor): The weight map tensor
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-6
    Returns:
        Tensor: The weighted coefficient
    """
    # get which dimensions to sum over (2, 3) for 2d, (2, 3, 4) for 3d
    dims = [i for i in range(2, len(y_true.shape), 1)]
    
    # sum along the spatial dimensions (width and height) 
    # computing coefficient for each volume/image in the batch
    intersection = torch.sum(y_pred * y_true * weight_map, dim=dims)
    union = torch.sum(weight_map * (y_pred + y_true), dim=dims)
    coeff = (2.0 * intersection + eps) / (union + eps)
    
    # mean of the whole batch
    return coeff.mean()


class CorrectionLoss(nn.Module):
    """
    Loss for training the correction network.
        - consists of a weighted dice coefficient which has higher weights in the middle
    """

    def __init__(self, dims: tuple, device: str, inverted=False):
        super().__init__()
        # self.alpha = alpha
        self.weight_map = _get_weight_map(dims, inverted)
        self.weight_map = self.weight_map.to(device)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # loss = 1 - dice_coefficient(y_pred, y_true)
        loss = 1 - weighted_coefficient(y_pred, y_true, self.weight_map)

        return loss


class VolumetricCorrectionLoss(nn.Module):
    """
    Loss for training the correction network with volumetric cuts.
        - consists of a weighted dice coefficient which has higher weights in the middle
    """

    def __init__(self, dims: tuple, device: str, alpha=0.75,  inverted=False):
        super().__init__()
        self.alpha = alpha
        self.weight_map = _get_weight_map(dims, inverted)
        self.weight_map = self.weight_map.to(device)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        depth = y_true.shape[2]
        
        dice_loss = 1 - dice_coefficient(y_pred, y_true)
        middle_loss = 1 - weighted_coefficient(y_pred[:,:,depth//2].unsqueeze(0), y_true[:,:,depth//2].unsqueeze(0), self.weight_map)

        loss = middle_loss + self.alpha * dice_loss

        return loss
