import nibabel as nib
import glob
import os
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from data.utils import generate_clicks, get_new_depth


class CorrectionMRIDataset(Dataset):
    """ Torch Dataset which returns ... """
    
    def __init__(self, seg_list: tuple[str], img_dims: tuple[int], clicks, cuts):
        self.seg_list = seg_list
        self.img_dims = img_dims
        self.clicks = clicks
        self.cuts = cuts

    def __len__(self):
        return len(self.seg_list)
    
    def _cut_volume(self, label: torch.Tensor, cut_size=32, num=np.inf, random=False) -> list[torch.Tensor]:
        """ """

        cut_size = cut_size // 2 # needed only as a distance from the center
        
        click_coords = torch.nonzero(label[1])
        # randomize cuts
        if random:
            click_coords = click_coords[torch.randperm(len(click_coords))]
        
        cuts = []
        k = num if len(click_coords) > num else len(click_coords)
        for click_idx in range(0, k):
            coords = click_coords[click_idx]

            click = torch.zeros_like(label[0][coords[0]])
            click[coords[1], coords[2]] = 1

            cut = torch.clone(label[0][coords[0]])
            # cut[coords[1], coords[2]] = 2
            cut = cut[
                coords[1]-cut_size:coords[1]+cut_size,
                coords[2]-cut_size:coords[2]+cut_size
            ]
            cuts.append(cut.unsqueeze(0))
            # cuts.append(cut)
        
        return cuts

    def _simulate_errors(self, cuts: list[torch.Tensor]) -> list[torch.Tensor]:
        """ """

        # cuts, _ = cut_volume(label, cut_size=cut_size, num=num)
        erosion_kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(5, 5))
        dilatation_kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(5, 5))

        faked_cuts = []
        for cut in cuts:
            pp = np.random.uniform(low=0.0, high=1.0)

            if 0.5 > pp:
                cut = cv2.erode(cut.numpy(), kernel=erosion_kernel, iterations=1)
            else:
                cut = cv2.dilate(cut.numpy(), kernel=dilatation_kernel, iterations=1)
            faked_cuts.append(torch.tensor(cut))

        return faked_cuts
        
    def __getitem__(self, idx: int) -> list[torch.Tensor, torch.Tensor]:
        """  """
        
        # Load the data
        seg = torch.as_tensor(nib.load(self.seg_list[idx]).get_fdata(), dtype=torch.float32).permute(2, 0, 1)
        # print('old shapes: ', t1.shape, t2.shape, seg.shape)

        # Crop the image 
        if self.img_dims[1]*2 > seg.shape[1]:
            seg = TF.center_crop(seg, (self.img_dims[1], self.img_dims[2]))
        else:
            seg = TF.center_crop(seg, (self.img_dims[1]*2, self.img_dims[2]*2))

        # Crop the depth of the volumes when they have bigger depth than required
        if seg.shape[0] > self.img_dims[0]:
            start_index, end_index = get_new_depth(seg, self.img_dims)
            seg = seg[start_index:end_index,:,:]
        
        # When the depth is smaller than required, fill the difference with zeros     
        elif seg.shape[0] < self.img_dims[0]:
            pad = (0, 0, 0, 0, (self.img_dims[0]-seg.shape[0])//2, (self.img_dims[0]-seg.shape[0])//2)
            seg = F.pad(seg, pad, "constant", 0)

        # Resizing to required width/height 
        seg = TF.resize(seg, (self.img_dims[1], self.img_dims[2]), interpolation=TF.InterpolationMode.NEAREST, antialias=False)

        clicks = generate_clicks(
            seg, 
            border=True, 
            clicks_num=self.clicks['num'], 
            clicks_dst=self.clicks.get('dst') or 4
        )

        seg = torch.stack((seg, clicks))
        
        cuts = self._cut_volume(
            seg, 
            cut_size=self.cuts['size'], 
            num=self.cuts['num'], 
            random=self.cuts['random']
        )

        faked_cuts = self._simulate_errors(cuts)

        return faked_cuts, cuts 


class CorrectionDataloader:
    """ """

    def __init__(self, dataset: Dataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size
    
    def __len__(self):
        return len(self.dataset)    

    def __iter__(self):
        for data in self.dataset:
            for batch_idx in range(0, len(data), self.batch_size):
                faked_batch = torch.stack(data[0][batch_idx:batch_idx+self.batch_size])
                batch = torch.stack(data[1][batch_idx:batch_idx+self.batch_size])

                yield faked_batch, batch


if __name__ == '__main__':
    data_path = 'data/all/'
    t1_list = glob.glob(os.path.join(data_path, 'VS-*-*/vs_*/*_t1_*'))
    t2_list = glob.glob(os.path.join(data_path, 'VS-*-*/vs_*/*_t2_*'))
    seg_list = glob.glob(os.path.join(data_path, 'VS-*-*/vs_*/*_seg_*'))

    data = CorrectionMRIDataset(
        ['data/all/VS-31-61/vs_gk_56/vs_gk_seg_refT2.nii.gz'], 
        (40, 256, 256),
        clicks = {
            'num': 3,
            'dst': 10
        },
        cuts = {
            'num': np.inf,
            'size': 32,
            'random': False,
        }
    )
    faked_cuts, cuts = data[0]
    print(len(cuts))
    print(cuts[0].shape)

    # batch_size = 4
    # batches = [seg[i:i+batch_size] for i in range(0, len(seg), batch_size)]
    # batch_tensors = [torch.stack(batch) for batch in batches]
    # print(len(batch_tensors))
    # print(batch_tensors[0].shape)

    dataloader = CorrectionDataloader(data, 4)
    for i, (x, y) in enumerate(dataloader):
        print(i, x.shape, y.shape)

    # preview(img, seg, torch.tensor(0.213), 100)