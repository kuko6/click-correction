import torch
from torch import nn
import scipy
from losses.dice import dice_coefficient

class DistanceLoss(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device

    def forward(self, y_pred, y_true):
        dst = torch.as_tensor(
            scipy.ndimage.distance_transform_edt(y_pred.detach().cpu()), 
            dtype=torch.float32, device=y_pred.device
        )

        inverted_dst = torch.as_tensor(
            scipy.ndimage.distance_transform_edt(1 - y_pred.detach().cpu()), 
            dtype=torch.float32, device=y_pred.device
        )

        combined = dst + inverted_dst

        # thresholding
        # thresh_val = torch.max(dst)
        thresh_val = 15
        combined[combined > thresh_val] = thresh_val
        print(thresh_val)

        # get the object indices from the annotation
        a = torch.where(y_true > 0, True, False)
        
        # loss calculation
        distance_loss = 1 - torch.mean(combined[a])
        dice_loss = 1 - dice_coefficient(y_pred, y_true)
        print(torch.mean(combined[a]))
        print(distance_loss, dice_loss)
        # loss = combined[a] / len(a)
        # print(dice_loss.grad_fn)
        loss = distance_loss + dice_loss
        
        return loss
    

if __name__ == '__main__':
    a = torch.zeros((2, 1, 40, 128, 128)).to('mps')
    b = torch.ones((2, 1, 40, 128, 128)).to('mps')
    
    loss_fn = DistanceLoss(device='mps')
    loss = loss_fn(a, b)
    print(loss)

    # print(loss.grad_fn)