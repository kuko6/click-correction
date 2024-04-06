import nibabel as nib
import glob
import os

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from .utils import generate_clicks, get_new_depth


# private function
def _min_max_normalise(image: torch.Tensor) -> torch.Tensor:
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
    """Torch Dataset which returns the stacked sequences and encoded mask."""

    def __init__(
        self,
        t1_list: tuple[str],
        t2_list: tuple[str],
        seg_list: tuple[str],
        img_dims: tuple[int],
        clicks=None,
    ):
        self.t1_list = t1_list
        self.t2_list = t2_list
        self.seg_list = seg_list
        self.img_dims = img_dims
        self.clicks = clicks

    def __len__(self):
        return len(self.t1_list)

    def _normalise(self, volume: torch.Tensor) -> torch.Tensor:
        """Normalise given volume."""
        # mean = torch.mean(volume, dim=(0, 1, 2), keepdim=True)
        # sd = torch.std(volume, dim=(0, 1, 2), keepdim=True)
        # return (volume - mean) / sd
        # return irm_min_max_preprocess(volume)

        return _min_max_normalise(volume)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Load the data and run the whole preprocessing pipeline.

        Args:
            idx (int): index of the data file
        Returns:
            tuple of tensors: stacked sequences and encoded mask
        """

        # Load the data
        t1 = torch.as_tensor(nib.load(self.t1_list[idx]).get_fdata(), dtype=torch.float32).permute(2, 0, 1)
        t2 = torch.as_tensor(nib.load(self.t2_list[idx]).get_fdata(), dtype=torch.float32).permute(2, 0, 1)
        seg = torch.as_tensor(nib.load(self.seg_list[idx]).get_fdata(), dtype=torch.float32).permute(2, 0, 1)
        # print('old shapes: ', t1.shape, t2.shape, seg.shape)

        # Crop the image
        if self.img_dims[1] * 2 > t1.shape[1]:
            t1 = TF.center_crop(t1, (self.img_dims[1], self.img_dims[2]))
            t2 = TF.center_crop(t2, (self.img_dims[1], self.img_dims[2]))
            seg = TF.center_crop(seg, (self.img_dims[1], self.img_dims[2]))
        else:
            t1 = TF.center_crop(t1, (self.img_dims[1] * 2, self.img_dims[2] * 2))
            t2 = TF.center_crop(t2, (self.img_dims[1] * 2, self.img_dims[2] * 2))
            seg = TF.center_crop(seg, (self.img_dims[1] * 2, self.img_dims[2] * 2))

        # Crop the depth of the volumes when they have bigger depth than required
        if t1.shape[0] > self.img_dims[0]:
            start_index, end_index = get_new_depth(seg, self.img_dims)
            t1 = t1[start_index:end_index, :, :]
            t2 = t2[start_index:end_index, :, :]
            seg = seg[start_index:end_index, :, :]

        # When the depth is smaller than required, fill the difference with zeros
        elif t1.shape[0] < self.img_dims[0]:
            pad = (0, 0, 0, 0, (self.img_dims[0]-t1.shape[0])//2, (self.img_dims[0]-t1.shape[0])//2)
            t1 = F.pad(t1, pad, "constant", 0)
            t2 = F.pad(t2, pad, "constant", 0)
            seg = F.pad(seg, pad, "constant", 0)

        # Resizing to required width/height
        t1 = TF.resize(t1, (self.img_dims[1], self.img_dims[2]), interpolation=TF.InterpolationMode.NEAREST, antialias=False)
        t2 = TF.resize(t2, (self.img_dims[1], self.img_dims[2]), interpolation=TF.InterpolationMode.NEAREST, antialias=False)
        seg = TF.resize(seg, (self.img_dims[1], self.img_dims[2]), interpolation=TF.InterpolationMode.NEAREST, antialias=False)

        # Normalisation
        t1 = self._normalise(t1)
        t2 = self._normalise(t2)
        stacked = torch.stack((t1, t2), axis=0)

        if self.clicks and self.clicks["use"]:
            clicks = generate_clicks(
                seg,
                fg=self.clicks["gen_fg"],
                bg=self.clicks["gen_bg"],
                border=self.clicks["gen_border"],
                clicks_num=self.clicks["num"],
                clicks_dst=self.clicks.get("dst") or 4,
            )

            # seg = seg.unsqueeze(0)
            # return stacked, clicks, seg
            seg = torch.stack((seg, clicks))
            return stacked, seg

        seg = seg.unsqueeze(0)
        return stacked, seg


if __name__ == "__main__":
    data_path = "data/all/"
    t1_list = glob.glob(os.path.join(data_path, "VS-*-*/vs_*/*_t1_*"))
    t2_list = glob.glob(os.path.join(data_path, "VS-*-*/vs_*/*_t2_*"))
    seg_list = glob.glob(os.path.join(data_path, "VS-*-*/vs_*/*_seg_*"))

    data = MRIDataset(
        ["data/all/VS-31-61/vs_gk_56/vs_gk_t1_refT2.nii.gz"],
        ["data/all/VS-31-61/vs_gk_56/vs_gk_t2_refT2.nii.gz"],
        ["data/all/VS-31-61/vs_gk_56/vs_gk_seg_refT2.nii.gz"],
        (40, 80, 80),
        clicks={
            "use": True,
            "gen_fg": False,
            "gen_bg": False,
            "gen_border": True,
            "num": 20,
        },
    )
    img, seg = data[0]
    print(img.shape, seg.shape)
    print(img.dtype, seg.dtype)
