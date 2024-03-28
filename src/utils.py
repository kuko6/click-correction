import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy

def get_weight_map(cutshape: tuple, minthresh=9, maxthresh=20, inverted=False) -> torch.Tensor:
    tmp = torch.zeros(cutshape)
    tmp[:,cutshape[2]//2, cutshape[2]//2] = 1
    dst = scipy.ndimage.distance_transform_edt(1-tmp[0])

    if inverted:
        weight_map = (1-dst)+np.abs(np.min(1-dst))
    else: 
        weight_map = dst
    weight_map[weight_map > maxthresh] = maxthresh
    weight_map[weight_map < minthresh] = 0
    
    return torch.as_tensor(weight_map, dtype=torch.float32).unsqueeze(0)


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


def cut_volume(label: torch.Tensor, cut_size=20, num=12):
    cut_size = cut_size // 2 # needed only as a distance from the center
    
    click_coords = torch.nonzero(label[1])
    cuts = []
    k = num if len(click_coords) > num else len(click_coords)
    
    for click_idx in range(0, k):
        coords = click_coords[click_idx]

        click = torch.zeros_like(label[0][coords[0]])
        click[coords[1], coords[2]] = 1

        # a = label[0][coords[0]] + click
        a = torch.clone(label[0][coords[0]])
        # a[coords[1], coords[2]] = 2
        cut = a[
            coords[1]-cut_size:coords[1]+cut_size,
            coords[2]-cut_size:coords[2]+cut_size
        ]
        cuts.append(cut)
    
    return cuts, click_coords


def fake_errors(cuts: list[torch.Tensor]):
    # cuts, _ = cut_volume(label, cut_size=cut_size, num=num)
    erosion_kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(3, 3))
    dilatation_kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(3, 3))

    faked_cuts = []
    for cut in cuts:
        pp = np.random.uniform(low=0.0, high=1.0)

        if 0.5 > pp:
            cut = cv2.erode(cut.numpy(), kernel=erosion_kernel, iterations=1)
        else:
            cut = cv2.dilate(cut.numpy(), kernel=dilatation_kernel, iterations=1)
        faked_cuts.append(torch.tensor(cut))

    return faked_cuts


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
    