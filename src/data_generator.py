import nibabel as nib
import math
import glob
import os

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.transforms.functional as TF


def get_glioma_indices(mask):
  first = torch.nonzero((mask == 1))[:,1][0].item()
  last = torch.nonzero((mask == 1))[:,1][-1].item()

  return first, last


# https://arxiv.org/abs/2011.01045
# https://github.com/lescientifik/open_brats2020/tree/main
def normalize(image):
    """ Basic min max scaler. """
    min_ = torch.min(image)
    max_ = torch.max(image)
    scale = max_ - min_
    image = (image - min_) / scale
    return image


class MRIDataset(Dataset):
    def __init__(self, t1_list, t2_list, seg_list, img_dims):
        self.t1_list = t1_list
        self.t2_list = t2_list
        self.seg_list = seg_list
        self.img_dims = img_dims

    def __len__(self):
      return len(self.t1_list)

    def _get_glioma_indices(self, mask):
      first = torch.nonzero((mask == 1))[:,0][0].item()
      last = torch.nonzero((mask == 1))[:,0][-1].item()

      return first, last

    def _crop_depth(self, mask):
      first, last = self._get_glioma_indices(mask)
      range_length = last - first + 1

      # print(f'old indices: {first}, {last} : {first - last}')

      # compute the desired padding size on both sides
      padding_size = self.img_dims[0] - range_length
      padding_size_left = math.floor(padding_size / 2)
      padding_size_right = math.ceil(padding_size / 2)

      # compute the new start and end indices of the cropped depth dimension
      mid_index = (first + last) // 2
      start_index = max(mid_index - math.floor(self.img_dims[0] / 2), 0)
      end_index = min(start_index + self.img_dims[0], mask.shape[0])

      # crop the volume along the depth dimension
      # cropped_volume = volume[start_index:end_index,:,:]

      return start_index, end_index

    def _normalise(self, volume):
      # mean = torch.mean(volume, dim=(0, 1, 2), keepdim=True)
      # sd = torch.std(volume, dim=(0, 1, 2), keepdim=True)
      # return (volume - mean) / sd
      # return irm_min_max_preprocess(volume)
      return normalize(volume)

    def __getitem__(self, idx):
      t1 = torch.as_tensor(nib.load(self.t1_list[idx]).get_fdata(), dtype=torch.float32).permute(2, 0, 1)
      t2 = torch.as_tensor(nib.load(self.t2_list[idx]).get_fdata(), dtype=torch.float32).permute(2, 0, 1)
      seg = torch.as_tensor(nib.load(self.seg_list[idx]).get_fdata(), dtype=torch.float32).permute(2, 0, 1)
      # print('old shapes: ', t1.shape, t2.shape, seg.shape)

      t1 = TF.center_crop(t1, (self.img_dims[1]*2, self.img_dims[2]*2))
      t2 = TF.center_crop(t2, (self.img_dims[1]*2, self.img_dims[2]*2))
      seg = TF.center_crop(seg, (self.img_dims[1]*2, self.img_dims[2]*2))

      if t1.shape[0] > self.img_dims[0]:
        start_index, end_index = self._crop_depth(seg)
        t1 = t1[start_index:end_index,:,:]
        t2 = t2[start_index:end_index,:,:]
        seg = seg[start_index:end_index,:,:]
        # print(t1.shape[0], t2.shape[0], seg.shape[0])

        # first, last = self._get_glioma_indices(seg)
        # print(f'new indices: {first}, {last} : {first - last}')

      elif t1.shape[0] < self.img_dims[0]:
        pad = (0, 0, 0, 0, (self.img_dims[0]-t1.shape[0])//2, (self.img_dims[0]-t1.shape[0])//2)
        t1 = F.pad(t1, pad, "constant", 0)
        t2 = F.pad(t2, pad, "constant", 0)
        seg = F.pad(seg, pad, "constant", 0)
        # print(t1.shape[0], t2.shape[0], seg.shape[0])

      t1 = TF.resize(t1, (self.img_dims[1], self.img_dims[2]), interpolation=TF.InterpolationMode.NEAREST, antialias=False)
      t2 = TF.resize(t2, (self.img_dims[1], self.img_dims[2]), interpolation=TF.InterpolationMode.NEAREST, antialias=False)
      seg = TF.resize(seg, (self.img_dims[1], self.img_dims[2]), interpolation=TF.InterpolationMode.NEAREST, antialias=False)

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

   data = MRIDataset([t1_list[0]], [t2_list[0]], [seg_list[0]], (40, 80, 80))
   img, label = data[0]
   print(img.shape, label.shape)
   print(img.dtype, label.dtype)