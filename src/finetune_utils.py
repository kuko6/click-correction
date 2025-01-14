import cv2
import numpy as np
import scipy
import torch

from data.utils import _select_points


def get_glioma_indices(mask: torch.Tensor) -> tuple[int, int]:
    glioma_indices = torch.nonzero((mask > 0))[:, 1]
    if len(glioma_indices) == 0:
        return 0, 0

    first = glioma_indices[0].item()
    last = glioma_indices[-1].item()

    return first, last


def get_border(mask: torch.Tensor):
    mask = mask.numpy()
    erosion_kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(3, 3))
    eroded_seg = cv2.erode(mask, kernel=erosion_kernel)
    border = mask - eroded_seg
    
    return border


def get_potential_coords(y_coords: list, diff_coords: list):
    # get coords (potential clicks), 
    # that are also in the original seg
    potential_clicks = []
    for coord in y_coords:
        if coord in diff_coords:
            potential_clicks.append(coord)
    
    return potential_clicks


def get_clicks(mask: torch.Tensor, pred: torch.Tensor, clicks_num=2, clicks_dst=4, seed=None) -> torch.Tensor:
    border = get_border(mask)
    
    if len(mask[mask == 1]) > len(pred[pred == 1]):
        dst = scipy.ndimage.distance_transform_edt(1 - pred)
    else:
        dst = scipy.ndimage.distance_transform_edt(pred)
    weighted_border = dst * border 
    
    if weighted_border.max() == 0.0:
        return torch.zeros_like(mask)

    # maybe find pixels in certain threshold
    indicies = np.where(weighted_border == weighted_border.max())
    clicks_coords = list(zip(*indicies))
    
    # select_clicks
    selected_points = _select_points(np.array(clicks_coords), clicks_num, clicks_dst)
    clicks = np.zeros_like(mask)
    for c in selected_points:
        clicks[c[0], c[1]] = 1

    return torch.as_tensor(clicks)


def simulate_clicks(mask: torch.Tensor, pred: torch.Tensor, clicks_num=2, clicks_dst=4, seed=None) -> torch.Tensor:
    clicks = torch.zeros_like(mask)
    start, end = get_glioma_indices(pred)
    for slice_idx in range(start, end+1):
        clicks[0,slice_idx] = get_clicks(mask[0,slice_idx], pred[0,slice_idx], clicks_num, clicks_dst, seed)
    return clicks


def cut_volume(seg: torch.Tensor, cut_size=32, num: int = np.inf) -> list[torch.Tensor]:
    cut_size = cut_size // 2  # needed only as a distance from the center

    # Get coordinates of the generated points
    click_coords = torch.nonzero(seg[1])

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

        cuts.append(cut)

    return cuts

def generate_cuts(seg, seq, clicks, cut_size):
    seg_cuts = cut_volume(torch.stack((seg, clicks)), cut_size=cut_size)
    t1_cuts = cut_volume(torch.stack((seq[0], clicks)), cut_size=cut_size)
    t2_cuts = cut_volume(torch.stack((seq[1], clicks)), cut_size=cut_size)

    return seg_cuts, t1_cuts, t2_cuts
