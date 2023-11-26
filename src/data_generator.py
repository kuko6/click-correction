import nibabel as nib
import math
import glob
import os

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from utils import generate_clicks

def min_max_normalise(image: torch.Tensor) -> torch.Tensor:
    """ 
    Basic min-max scaler. \n
    https://arxiv.org/abs/2011.01045
    https://github.com/lescientifik/open_brats2020/tree/main
    """

    min_ = torch.min(image)
    max_ = torch.max(image)
    scale = max_ - min_
    image = (image - min_) / scale
    
    return image


class MRIDataset(Dataset):
    """ Torch Dataset which returns the stacked sequences and encoded mask. """
    
    def __init__(self, t1_list: tuple[str], t2_list: tuple[str], seg_list: tuple[str], img_dims: tuple[int], gen_clicks=False):
        self.t1_list = t1_list
        self.t2_list = t2_list
        self.seg_list = seg_list
        self.img_dims = img_dims
        self.gen_clicks = gen_clicks

    def __len__(self):
        return len(self.t1_list)
    
    def _get_glioma_indices(self, mask: torch.Tensor) -> tuple[int, int]:
        """ Returns the first and last slice indices of the tumour in given mask """
        
        first = torch.nonzero((mask == 1))[:,0][0].item()
        last = torch.nonzero((mask == 1))[:,0][-1].item()

        return first, last

    def _get_new_depth(self, mask: torch.Tensor):
        """ Calculates new depth based on the position of the tumour """
        
        first, last = self._get_glioma_indices(mask)

        # compute the new start and end indices of the cropped depth dimension
        mid_index = (first + last) // 2
        start_index = max(mid_index - math.floor(self.img_dims[0] / 2), 0)
        end_index = min(start_index + self.img_dims[0], mask.shape[0])

        return start_index, end_index
    
    def _normalise(self, volume: torch.Tensor) -> torch.Tensor:    
        """ Normalise given volume """
        # mean = torch.mean(volume, dim=(0, 1, 2), keepdim=True)
        # sd = torch.std(volume, dim=(0, 1, 2), keepdim=True)
        # return (volume - mean) / sd
        # return irm_min_max_preprocess(volume)
        
        return min_max_normalise(volume)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Load the data and run the whole preprocessing pipeline. \n
        Returns the stacked sequences and encoded mask.
        """
        
        # Load the data
        t1 = torch.as_tensor(nib.load(self.t1_list[idx]).get_fdata(), dtype=torch.float32).permute(2, 0, 1)
        t2 = torch.as_tensor(nib.load(self.t2_list[idx]).get_fdata(), dtype=torch.float32).permute(2, 0, 1)
        seg = torch.as_tensor(nib.load(self.seg_list[idx]).get_fdata(), dtype=torch.float32).permute(2, 0, 1)
        # print('old shapes: ', t1.shape, t2.shape, seg.shape)

        # Crop the image 
        t1 = TF.center_crop(t1, (self.img_dims[1]*2, self.img_dims[2]*2))
        t2 = TF.center_crop(t2, (self.img_dims[1]*2, self.img_dims[2]*2))
        seg = TF.center_crop(seg, (self.img_dims[1]*2, self.img_dims[2]*2))

        # Crop the depth of the volumes when they have bigger depth than required
        if t1.shape[0] > self.img_dims[0]:
            start_index, end_index = self._get_new_depth(seg)
            t1 = t1[start_index:end_index,:,:]
            t2 = t2[start_index:end_index,:,:]
            seg = seg[start_index:end_index,:,:]
            # print(t1.shape[0], t2.shape[0], seg.shape[0])

            # first, last = self._get_glioma_indices(seg)
            # print(f'new indices: {first}, {last} : {first - last}')
        
        # When the depth is smaller than required fill the difference with zeros     
        elif t1.shape[0] < self.img_dims[0]:
            pad = (0, 0, 0, 0, (self.img_dims[0]-t1.shape[0])//2, (self.img_dims[0]-t1.shape[0])//2)
            t1 = F.pad(t1, pad, "constant", 0)
            t2 = F.pad(t2, pad, "constant", 0)
            seg = F.pad(seg, pad, "constant", 0)
            # print(t1.shape[0], t2.shape[0], seg.shape[0])

        # Resizing to required width/height 
        t1 = TF.resize(t1, (self.img_dims[1], self.img_dims[2]), interpolation=TF.InterpolationMode.NEAREST, antialias=False)
        t2 = TF.resize(t2, (self.img_dims[1], self.img_dims[2]), interpolation=TF.InterpolationMode.NEAREST, antialias=False)
        seg = TF.resize(seg, (self.img_dims[1], self.img_dims[2]), interpolation=TF.InterpolationMode.NEAREST, antialias=False)

        if self.gen_clicks:
            bg_clicks, fg_clicks = generate_clicks(seg)

        # Normalisation
        t1 = self._normalise(t1)
        t2 = self._normalise(t2)

        stacked = torch.stack((t1, t2), axis=0)
        seg = seg.unsqueeze(0)

        return stacked, seg


if __name__ == '__main__':
    data_path = 'data/all/'
    t1_list = glob.glob(os.path.join(data_path, 'VS-*-*/vs_*/*_t1_*'))
    t2_list = glob.glob(os.path.join(data_path, 'VS-*-*/vs_*/*_t2_*'))
    seg_list = glob.glob(os.path.join(data_path, 'VS-*-*/vs_*/*_seg_*'))

    # data = MRIDataset([t1_list[0]], [t2_list[0]], [seg_list[0]], (40, 80, 80))
    data = MRIDataset(
        ['data/all/VS-31-61/vs_gk_56/vs_gk_t1_refT2.nii.gz'], 
        ['data/all/VS-31-61/vs_gk_56/vs_gk_t2_refT2.nii.gz'], 
        ['data/all/VS-31-61/vs_gk_56/vs_gk_seg_refT2.nii.gz'], 
        (40, 80, 80)
    )
    img, label = data[0]
    print(img.shape, label.shape)
    print(img.dtype, label.dtype)