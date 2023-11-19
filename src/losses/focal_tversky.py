import torch
from torch import nn
import torch.nn.functional as F


def tversky_index(y_pred, y_true, epsilon=1e-6, alpha=0.5, beta=0.5):
    TP = torch.sum(y_pred * y_true, dim=[2, 3, 4])    
    FP = torch.sum((1-y_true) * y_pred, dim=[2, 3, 4])
    FN = torch.sum(y_true * (1-y_pred), dim=[2, 3, 4])
    tversky = (TP + epsilon) / (TP + alpha*FP + beta*FN + epsilon)  
    
    return tversky.mean()


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=.75):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, y_pred, y_true, epsilon=1e-6):      
        tversky = tversky_index(y_pred, y_true, epsilon=epsilon, alpha=self.alpha, beta=self.beta)
        focaltversky = (1 - tversky) ** self.gamma
        
        return focaltversky
    
if __name__ == '__main__':
    FocalTverskyLoss()