import torch
from torch import nn
import scipy
from losses.dice import dice_coefficient
from utils import get_glioma_indices

class DistanceLoss(nn.Module):
    def __init__(self, thresh_val: None | int, thresh_mode='max', probs=True, probs_threshold=0.7):
        super().__init__()
        
        self.thresh_val = thresh_val
        self.thresh_mode = thresh_mode
        self.probs = probs
        self.probs_threshold = probs_threshold
        self.alpha = 3

    def forward(self, y_pred, y_true):
        combined = torch.zeros_like(y_pred)
        
        if self.probs:
            # threshold the probabilities
            # ! might overide the original predictions !
            y_threshed = (y_pred > self.probs_threshold).type(torch.float32).detach().cpu()

        for seg_idx in range(len(y_pred)):
            first, last = get_glioma_indices(y_threshed[seg_idx])
            
            for slice_idx in range(first, last+1):
                dst = torch.as_tensor(
                    scipy.ndimage.distance_transform_edt(y_threshed[seg_idx,0,slice_idx,:,:]), 
                    dtype=torch.float32, device=y_pred.device
                )

                inverted_dst = torch.as_tensor(
                    scipy.ndimage.distance_transform_edt(1 - y_threshed[seg_idx,0,slice_idx,:,:]), 
                    dtype=torch.float32, device=y_pred.device
                )

                combined[seg_idx,:,slice_idx,:,:] = dst + inverted_dst

            # thresholding
            # thresh_val = 5.
            if self.thresh_mode == 'max' or self.thresh_val == None:
                self.thresh_val = torch.max(dst).item()
            combined[seg_idx][combined[seg_idx] > self.thresh_val] = self.thresh_val

        # get the object indices from the annotation
        a = torch.where(y_true > 0, True, False)
        
        # loss calculation
        distance_loss = torch.abs(1 - torch.mean(combined[a]))
        dice_loss = 1 - dice_coefficient(y_pred, y_true)
        loss = dice_loss + (self.alpha * distance_loss)

        # print(torch.mean(combined[a]))
        # print(distance_loss, dice_loss)
        # loss = combined[a] / len(a)
        # print(dice_loss.grad_fn)
        
        return loss
    

if __name__ == '__main__':
    a = torch.zeros((2, 1, 40, 128, 128)).to('mps')
    b = torch.ones((2, 1, 40, 128, 128)).to('mps')
    
    loss_fn = DistanceLoss(device='mps')
    loss = loss_fn(a, b)
    print(loss)

    # print(loss.grad_fn)