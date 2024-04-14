import nibabel as nib
import glob
import os
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from .utils import generate_clicks
# from utils import generate_clicks


class CorrectionMRIDataset(Dataset):
    """Torch Dataset which returns 2d cuts with and without errors."""

    def __init__(
        self,
        seg_list: tuple[str],
        img_dims: tuple[int],
        clicks: dict,
        cuts: dict,
        seed: int | None =None,
        random=False,
        include_unchanged=False,
        augment=False
    ):
        self.seg_list = seg_list
        self.img_dims = img_dims
        self.clicks = clicks
        self.cuts_config = cuts
        self.seed = seed
        self.random = random
        self.include_unchanged = include_unchanged
        self.augment = augment

    def __len__(self):
        return len(self.seg_list)

    def _augment(self, cut: torch.Tensor) -> torch.Tensor:
        """ 
        Rotates and flips provided cut.

        Args:
            cut (Tensor): preprocessed cut
        Returns:
            tensor: augmented cut
        """

        rng = np.random.default_rng()
        pp = rng.uniform(low=0.0, high=1.0)

        if pp > 0.8:
            angle = rng.uniform(low=-90, high=90)
            cut = TF.rotate(cut, angle)
        
        if pp > 0.7:
            cut = TF.hflip(cut)
        
        if pp > 0.6:
            cut = TF.vflip(cut)
        
        return cut

    def _cut_volume(
        self,
        seg: torch.Tensor,
        cut_size=32,
        num: int = np.inf,
        random=False,
        augment=False,
    ) -> list[torch.Tensor]:
        """
        Generated cuts from the givine segmentation.

        Args:
            seg (Tensor): stacked tensor of both the segmentation mask and generated clicks
            cut_size (int): size of the generated cuts
            num (int): number of cuts to generate
            random (bool): whether to randomize the cuts or create them in order
        Returns:
            list of tensors: list of generated cuts in shape (cut_size, cut_size)
        """

        cut_size = cut_size // 2  # needed only as a distance from the center

        # Get coordinates of the generated points
        click_coords = torch.nonzero(seg[1])

        # Randomize cuts
        if random:
            click_coords = click_coords[torch.randperm(len(click_coords))]

        cuts = []
        k = num if len(click_coords) > num else len(click_coords)
        for click_idx in range(0, k):
            coords = click_coords[click_idx]

            # Cut the volume based on the specified cut size
            cut = torch.clone(seg[0][coords[0]])
            cut = cut[
                coords[1] - cut_size : coords[1] + cut_size,
                coords[2] - cut_size : coords[2] + cut_size,
            ].unsqueeze(0)

            if augment:
                cut = self._augment(cut)

            cuts.append(cut)

        return cuts

    def _simulate_errors(self, cuts: list[torch.Tensor], hide_unchanged=False, seed: int | None =None) -> list[torch.Tensor]:
        """
        Simulate errors for the generated cuts and return 'erroneous' cuts.
        
        Args:
            cuts (list of tensors): list of generated cuts
        Returns:
            list of tensors: list of simulated erroneous cuts
        """

        # cuts, _ = cut_volume(label, cut_size=cut_size, num=num)
        erosion_kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(5, 5))
        dilatation_kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(7, 7))

        faked_cuts = []
        if seed is not None:
            rng = np.random.default_rng(seed=seed)
        else:
            rng = np.random.default_rng()

        # Iterate over generated cuts and either erode or dilate the segmentation
        for cut in cuts:
            pp = rng.uniform(low=0.0, high=1.0)

            # "hide" unchanged cuts
            if hide_unchanged and pp > 0.9:
                faked_cuts.append(cut)
                continue

            # use only 1 iterations for smaller tumours
            if len(cut[cut > 0]) < 250:
                iterations = 1
            else:
                iterations = 2
            
            if len(cut[cut > 0]) < 50:
                # only do dilatation, when the tumours is too small
                cut = cv2.dilate(cut.numpy(), kernel=dilatation_kernel, iterations=1)
            else:
                if 0.5 > pp:
                    cut = cv2.erode(cut.numpy(), kernel=erosion_kernel, iterations=iterations)
                else:
                    cut = cv2.dilate(cut.numpy(), kernel=dilatation_kernel, iterations=iterations)
            faked_cuts.append(torch.tensor(cut))

        return faked_cuts

    def __getitem__(self, idx: int) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Load the segmentation mask and run the whole preprocessing pipeline.

        Args:
            idx (int): index of the data file
        Returns:
            tuple of list of tensors: both the errorenous and true cuts given as lists of tensors
        """

        # Load the data
        seg = torch.as_tensor(nib.load(self.seg_list[idx]).get_fdata(), dtype=torch.float32).permute(2, 0, 1)

        # Crop the image
        if self.img_dims[0] * 2 > seg.shape[1]:
            seg = TF.center_crop(seg, (self.img_dims[0], self.img_dims[1]))
        else:
            seg = TF.center_crop(seg, (self.img_dims[0] * 2, self.img_dims[1] * 2))

        # Resizing to required width/height
        seg = TF.resize(
            seg,
            (self.img_dims[0], self.img_dims[1]),
            interpolation=TF.InterpolationMode.NEAREST,
            antialias=False,
        )

        clicks = generate_clicks(
            seg,
            border=True,
            clicks_num=self.clicks["num"],
            clicks_dst=self.clicks.get("dst") or 4,
            seed=self.seed
        )
        seg = torch.stack((seg, clicks))

        cuts = self._cut_volume(
            seg,
            cut_size=self.cuts_config["size"],
            num=self.cuts_config["num"],
            random=self.random,
            augment=self.augment,
        )
        faked_cuts = self._simulate_errors(cuts, self.include_unchanged, self.seed)

        return faked_cuts, cuts


class CorrectionDataLoader:
    """Custom dataloader for iterating over the `CorrectionDatates`."""

    def __init__(self, dataset: Dataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size

    def __len__(self):
        data_length = 0
        for data in self.dataset:
            if len(data[0]) % self.batch_size > 0:
                data_length += (len(data[0])//self.batch_size + 1)
            else:
                data_length += len(data[0])//self.batch_size
        return data_length
        # return len(self.dataset)

    def __iter__(self):
        # Iterate over the dataset
        for data in self.dataset:
            # Iterate over the generated cuts and yield them in batches
            for batch_idx in range(0, len(data[0]), self.batch_size):
                faked_batch = torch.stack(data[0][batch_idx : batch_idx + self.batch_size])
                batch = torch.stack(data[1][batch_idx : batch_idx + self.batch_size])
                yield faked_batch, batch


if __name__ == "__main__":
    data_path = "data/all/"
    t1_list = glob.glob(os.path.join(data_path, "VS-*-*/vs_*/*_t1_*"))
    t2_list = glob.glob(os.path.join(data_path, "VS-*-*/vs_*/*_t2_*"))
    seg_list = glob.glob(os.path.join(data_path, "VS-*-*/vs_*/*_seg_*"))

    data = CorrectionMRIDataset(
        # ["data/all/VS-31-61/vs_gk_56/vs_gk_seg_refT2.nii.gz", 
        #  "data/all/VS-31-61/vs_gk_56/vs_gk_seg_refT2.nii.gz"],
        seg_list[:4],
        (256, 256),
        clicks={"num": 3, "dst": 10},
        cuts={
            "num": 32,
            "size": 32,
        },
        seed=690,
        random=False,
        include_unchanged=True,
        augment=True
    )
    faked_cuts, cuts = data[0]
    # print(type(cuts))
    # print(len(cuts))
    print(cuts[0].shape)

    dataloader = CorrectionDataLoader(data, batch_size=4)
    total = len(dataloader)
    for i, (x, y) in enumerate(dataloader):
        print(f"{i+1}/{total}, {x.shape}, {y.shape}")
    print(len(dataloader))
