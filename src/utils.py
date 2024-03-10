# import cv2
import torch
import numpy as np
# import nibabel as nib
import matplotlib.pyplot as plt

# from data_generator import MRIDataset


def get_glioma_indices(mask: torch.Tensor) -> tuple[int, int]:
    """ Returns the first and last slice indices of the tumour in given mask """
    
    glioma_indices = torch.nonzero((mask > 0))[:,1]
    # first = torch.nonzero((mask > 0))[:,1][0].item()
    # last = torch.nonzero((mask > 0))[:,1][-1].item()
    if len(glioma_indices) == 0:
        return 0, 0
    
    first = glioma_indices[0].item()
    last = glioma_indices[-1].item()

    return first, last


# !!! Outdated !!!
# def preview_clicks(t1_list, t2_list, seg_list, clicks):
#     """ Saves a png containing both sequences and a segmentation mask with clicks """

#     data = MRIDataset([t1_list[10]], [t2_list[10]], [seg_list[10]], (40, 80, 80), clicks=clicks)
#     img, clicks = data[0]
#     data = MRIDataset([t1_list[10]], [t2_list[10]], [seg_list[10]], (40, 80, 80))
#     img, seg = data[0]

#     # Compute number of slices with the tumour
#     first, last = get_glioma_indices(seg)
#     length = (last-first+1)
#     n_graphs = (length*3)//6
#     rows = n_graphs
#     cols = 6
#     res = cols if cols > rows else rows

#     mask = seg[0,:,:,:] + clicks[1,:,:,:] + clicks[0,:,:,:]

#     # Plot them
#     fig, axs = plt.subplots(rows, cols, figsize=(res*2, res*2))
#     axs = axs.flatten()
#     j = 0
#     for i in range(first, last):
#         if j >= len(axs): break
#         axs[j].imshow(img[0,i,:,:], cmap='gray')
#         axs[j].axis('off')
#         axs[j].set_title(f't1 slice {i}', fontsize=9)

#         axs[j+1].imshow(img[1,i,:,:], cmap='gray')
#         axs[j+1].axis('off')
#         axs[j+1].set_title(f't2 slice {i}', fontsize=9)

#         axs[j+2].imshow(mask[i,:,:], cmap='magma')
#         axs[j+2].axis('off')
#         axs[j+2].set_title(f'mask slice {i}', fontsize=9)
    
#         # axs[j+3].imshow(clicks[0,i,:,:], cmap='magma')
#         # axs[j+3].axis('off')
#         # axs[j+3].set_title(f'bg clicks slice {i}', fontsize=9)

#         # axs[j+4].imshow(clicks[1,i,:,:], cmap='magma')
#         # axs[j+4].axis('off')
#         # axs[j+4].set_title(f'fg clicks slice {i}', fontsize=9)
#         j += 3

#     fig.savefig('outputs/clicks_preview.png')
#     plt.close(fig)


def preview(y_pred: torch.Tensor, y: torch.Tensor, dice: torch.Tensor, epoch=0):
    """ Saves a png of sample prediction `y_pred` for scan `y` """

    # Compute number of slices with the tumour
    first, last = get_glioma_indices(y)
    length = (last-first+1)
    n_graphs = (length*2)//6
    rows = n_graphs
    cols = 6
    res = cols if cols > rows else rows

    # Plot them
    fig, axs = plt.subplots(rows, cols, figsize=(res*2, res*2))
    axs = axs.flatten()
    j = 0
    for i in range(first, last):
        if j >= len(axs): 
            break

        axs[j].imshow(y[0,i,:,:].cpu().detach(), cmap='magma')
        axs[j].axis('off')
        axs[j].set_title(f'mask slice {i}', fontsize=9)
        
        axs[j+1].imshow(y_pred[0,i,:,:].cpu().detach(), cmap='magma')
        axs[j+1].axis('off')
        axs[j+1].set_title(f'pred slice {i}', fontsize=9)

        if y.shape[0] == 2:
            axs[j+2].imshow(y[1,i,:,:].cpu().detach(), cmap='magma')
            axs[j+2].axis('off')
            axs[j+2].set_title(f'clicks slice {i}', fontsize=9)
            j += 3
        else:
            j += 2

    fig.suptitle(f'Dice: {dice}', fontsize=10)
    plt.subplots_adjust(top=0.9)

    fig.savefig(f'outputs/images/{epoch}_preview.png')
    plt.close(fig)


def plot_tumour(mask):
    # Compute number of slices with the tumour
    first, last = get_glioma_indices(mask)
    print('Tumour indices: ', first, last)

    length = (last-first+1)
    n_graphs = (length)//4
    rows = n_graphs
    cols = 4
    res = cols if cols > rows else rows

    # Plot them
    fig, axs = plt.subplots(rows, cols, figsize=(res*2, res*2))
    axs = axs.flatten()
    j = 0
    for i in range(first, last):
        if j >= len(axs): 
            break
        axs[j].imshow(mask[0,i,:,:], cmap='magma')
        axs[j].axis('off')
        axs[j].set_title(f'mask slice {i}', fontsize=9)
        j += 1

    plt.show()    


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
    