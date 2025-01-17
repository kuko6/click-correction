import scipy
import torch
from kornia.contrib import distance_transform as kornia_dst
from torch import nn

from utils import get_glioma_indices


class DistanceLoss(nn.Module):
    def __init__(self, thresh_val: None | int, thresh_mode='max', probs=True, preds_threshold=0.7):
        super().__init__()
        
        self.thresh_val = thresh_val
        self.thresh_mode = thresh_mode
        self.probs = probs
        self.preds_threshold = preds_threshold
        self.alpha = 3

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        combined = torch.zeros_like(y_pred)
        
        if self.probs:
            # threshold the probabilities
            y_threshed = (y_pred > self.preds_threshold).type(torch.float32).detach().cpu()
        
        # iterate over batch
        for seg_idx in range(len(y_pred)):
            first, last = get_glioma_indices(y_threshed[seg_idx])

            # iterate over slices in segmentation
            # this might be wrong
            for slice_idx in range(first, last+1):
                dst = torch.as_tensor(
                    scipy.ndimage.distance_transform_edt(y_threshed[seg_idx,0,slice_idx,:,:]), 
                    dtype=torch.float32, 
                    device=y_pred.device
                )

                inverted_dst = torch.as_tensor(
                    scipy.ndimage.distance_transform_edt(1 - y_threshed[seg_idx,0,slice_idx,:,:]), 
                    dtype=torch.float32, 
                    device=y_pred.device
                )

                combined[seg_idx,:,slice_idx,:,:] = dst + inverted_dst

            # thresholding
            # thresh_val = 5.
            if self.thresh_mode == 'max' or self.thresh_val is None:
                self.thresh_val = torch.max(dst).item()
            combined[seg_idx][combined[seg_idx] > self.thresh_val] = self.thresh_val
        
        combined.requires_grad_()
        
        # get the object indices from the annotation
        # a = torch.where(y_true > 0, True, False)
        
        # loss calculation
        # distance_loss = torch.abs(1 - torch.mean(combined[a]))
        # distance_loss = torch.abs(torch.mean(1 - combined[a]))

        # dice_loss = 1 - dice_coefficient(y_pred, y_true)
        # loss = dice_loss + (self.alpha * distance_loss)
        
        # take 2
        distance_loss = torch.sum(torch.mul(combined, y_true)) / torch.count_nonzero(y_true)
        # overlap = torch.sum(torch.mul(y_pred, y_true)) / torch.count_nonzero(y_true)
        # loss = overlap + (self.alpha * distance_loss)
        loss = distance_loss
    
        return loss
    

# Convolutional dst
class DistanceLoss2(nn.Module):
    def __init__(self, thresh_val: None | int, thresh_mode='max', probs=True, preds_threshold=0.7):
        super().__init__()
        
        self.thresh_val = thresh_val
        self.thresh_mode = thresh_mode
        self.probs = probs
        self.preds_threshold = preds_threshold
        self.alpha = 3

    def forward(self, y_pred, y_true):
        combined = torch.zeros_like(y_pred)
        y_threshed = torch.clone(y_pred) 
        if self.probs:
            # threshold the probabilities
            # y_threshed = (y_pred > self.preds_threshold).type(torch.float32)
            y_threshed[y_threshed <= self.preds_threshold] = 0
            y_threshed[y_threshed > self.preds_threshold] = 1

        dst = kornia_dst(y_threshed[:,0,:,:,:])
        inverted_dst = kornia_dst(1 - y_threshed[:,0,:,:,:])

        combined[:,0,:,:,:] = dst + inverted_dst

        if self.thresh_mode == 'max' or self.thresh_val is None:
            self.thresh_val = torch.max(dst).item()
        combined[combined > self.thresh_val] = self.thresh_val

        distance_loss = torch.sum(torch.mul(combined, y_true)) / torch.count_nonzero(y_true)
        # overlap = torch.sum(torch.mul(y_pred, y_true)) / torch.count_nonzero(y_true)
        # loss = overlap + (self.alpha * distance_loss)
        loss = distance_loss

        # loss.requires_grad_()

        return loss


if __name__ == '__main__':
    a = torch.zeros((2, 1, 40, 128, 128)).to('mps')
    b = torch.ones((2, 1, 40, 128, 128)).to('mps')
    
    loss_fn = DistanceLoss(device='mps')
    loss = loss_fn(a, b)
    print(loss)

    # print(loss.grad_fn)
