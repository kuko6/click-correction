import torch
import torch.nn.functional as F
from torch import nn


def tversky_index(y_pred: torch.Tensor, y_true: torch.Tensor, epsilon=1e-6, alpha=0.5, beta=0.5):
    """ 
    Computes the Tversky index. With alpha, beta = 0.5 corresponds to Dice coefficient.
    """
    
    TP = torch.sum(y_pred * y_true, dim=[2, 3, 4])    
    FP = torch.sum((1-y_true) * y_pred, dim=[2, 3, 4])
    FN = torch.sum(y_true * (1-y_pred), dim=[2, 3, 4])
    tversky = (TP + epsilon) / (TP + alpha*FP + beta*FN + epsilon)  
    
    return tversky.mean()


class TverskyLoss(nn.Module):
    """ 
    Tversky loss function defined as:
    ```
    tversky_loss = 1 - tversky_index
    ```
    With alpha, beta = 0.5 the loss corresponds to Dice loss. 

    https://link.springer.com/chapter/10.1007/978-3-319-67389-9_44
    """

    def __init__(self, alpha=0.5, beta=0.5):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, epsilon=1e-6):             
        tversky = tversky_index(y_pred, y_true, epsilon=epsilon, alpha=self.alpha, beta=self.beta)
        
        return 1 - tversky


class FocalLoss(nn.Module):
    """ 
    Focal loss function. The gamma should be from [0.5, 5].

    https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf
    """
    
    def __init__(self, alpha, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        ce_loss = F.binary_cross_entropy(y_pred, y_true, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()
        
        return loss


class FocalTverskyLoss(nn.Module):
    """ 
    Focal Tversky loss function defined as:
    ```
    focal_tversky_loss = (1 - tversky_index) ** gamma
    ```
    With alpha, beta = 0.5 the loss corresponds to Dice loss.
    
    https://ieeexplore.ieee.org/abstract/document/8759329
    """

    def __init__(self, alpha=0.5, beta=0.5, gamma=.75):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, epsilon=1e-6):      
        tversky = tversky_index(y_pred, y_true, epsilon=epsilon, alpha=self.alpha, beta=self.beta)
        focaltversky = (1 - tversky) ** self.gamma
        
        return focaltversky
