import cv2
import torch
import numpy as np


def get_glioma_indices(mask: torch.Tensor) -> tuple[int, int]:
    """ Returns the first and last slice indices of the tumour in given mask """

    first = torch.nonzero((mask == 1))[:,1][0].item()
    last = torch.nonzero((mask == 1))[:,1][-1].item()

    return first, last


def generate_clicks(mask: torch.Tensor, fg=False, bg=False, clicks_num=2, click_size=2) -> tuple[torch.Tensor, torch.Tensor]:
    """ Generate click masks """

    first, last = get_glioma_indices(mask)

    for slice in range(first, last):
        erosion_kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(5, 5))
        dilatation_kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(17, 17))
        eroded_seg = cv2.erode(mask[0,slice,:,:], kernel=erosion_kernel)
        dilated_seg = cv2.dilate(mask[0,slice,:,:,], kernel=dilatation_kernel, iterations=4)

        diff = mask[0,slice,:,:] - eroded_seg
        diff2 = dilated_seg - mask[0,slice,:,:]

        border_idx = torch.where(diff == 1)
        border_coords = list(zip(*border_idx))

        # Get fg coordinates
        inner_idx = torch.where(mask[0,slice,:,:] == 1)
        inner_coords = list(zip(*inner_idx))
        inner_coords = list(set(inner_coords) - set(border_coords))
        # np.random.shuffle(inner_coords)
        
        # Get bg coordinates
        outer_idx = torch.where(diff2 == 1)
        outer_coords = list(zip(*outer_idx))
        outer_coords = list(set(outer_coords) - set(inner_coords))
        # np.random.shuffle(outer_coords)

        # Add bg clicks
        bg_clicks = torch.zeros_like(mask)
        if bg:
            for c in outer_coords[:clicks_num]:
                bg_clicks[0,slice,c[0]:c[0]+click_size, c[1]:c[1]+click_size] = 3

        # Add fg clicks
        fg_clicks = torch.zeros_like(mask)
        if fg:
            for c in inner_coords[:clicks_num]:
                fg_clicks[0,slice,c[0]:c[0]+click_size, c[1]:c[1]+click_size] = 4

    return bg_clicks, fg_clicks


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