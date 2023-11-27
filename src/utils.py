import cv2
import torch
import numpy as np
import nibabel as nib


def get_glioma_indices(mask: torch.Tensor) -> tuple[int, int]:
    """ Returns the first and last slice indices of the tumour in given mask """

    first = torch.nonzero((mask == 1))[:,1][0].item()
    last = torch.nonzero((mask == 1))[:,1][-1].item()

    return first, last


# https://stackoverflow.com/a/73704579
class EarlyStopper:
    """ Early stopper for training. Supports `'min'` and `'max'` mode.  """
    
    def __init__(self, patience=1, delta=0.0, mode='min'):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.min_value = np.inf
        self.max_value = 0
        self.mode = mode

    def __call__(self, value): 
        if self.mode == 'min':
            if value < self.min_value:
                self.min_value = value
                self.counter = 0     
            elif value > (self.min_value + self.delta):
                self.counter += 1
                print('-------------------------------')
                print(f'early stopping: {self.counter}/{self.patience}')
                
        elif self.mode == 'max':
            if value > self.max_value:
                self.max_value = value
                self.counter = 0     
            elif value < (self.max_value - self.delta):
                self.counter += 1
                print('-------------------------------')
                print(f'early stopping: {self.counter}/{self.patience}')
                
        if self.counter >= self.patience:
            return True
        
        return False
    