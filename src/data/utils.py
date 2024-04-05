import cv2
import numpy as np
import torch


# private function
def _get_glioma_indices(mask: torch.Tensor) -> tuple[int, int]:
    """ Returns the first and last slice indices of the tumour in given mask """
    
    first = torch.nonzero((mask == 1))[:,0][0].item()
    last = torch.nonzero((mask == 1))[:,0][-1].item()

    return first, last

# private function
def _select_points(coords, n: int, d: int) -> list:
    """ 
    Select points for click annotations from generated coordinates 
    considering their distance and number.
    """
    
    valid_points = []
    valid_points.append(coords[0])
    
    i = 1
    while n != len(valid_points) and i < len(coords):
        new_point = coords[i]

        valid = True
        for p in valid_points:
            dist = np.linalg.norm(p - new_point)
            if dist < d:
                valid = False
                break

        if valid:
            valid_points.append(new_point)
        
        i += 1
    
    return valid_points
    

def generate_clicks(
    mask: torch.Tensor,
    fg=False,
    bg=False,
    border=False,
    clicks_num=2,
    clicks_dst=4
) -> torch.Tensor:
    """ Generate click masks """

    first, last = _get_glioma_indices(mask)
    mask = mask.numpy()

    bg_clicks = np.zeros_like(mask)
    fg_clicks = np.zeros_like(mask)
    border_clicks = np.zeros_like(mask)
    for slice in range(first, last):
        erosion_kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(3, 3))
        dilatation_kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(17, 17))
        eroded_seg = cv2.erode(mask[slice,:,:], kernel=erosion_kernel)
        dilated_seg = cv2.dilate(mask[slice,:,:], kernel=dilatation_kernel, iterations=4)

        diff = mask[slice,:,:] - eroded_seg
        diff2 = dilated_seg - mask[slice,:,:]

        border_idx = np.where(diff == 1)
        border_coords = list(zip(*border_idx))
        np.random.shuffle(border_coords)
        
        # Get fg coordinates
        inner_idx = np.where(mask[slice,:,:] == 1)
        inner_coords = list(zip(*inner_idx))
        inner_coords = list(set(inner_coords) - set(border_coords))
        np.random.shuffle(inner_coords)
        
        # Get bg coordinates
        outer_idx = np.where(diff2 == 1)
        outer_coords = list(zip(*outer_idx))
        outer_coords = list(set(outer_coords) - set(inner_coords))
        np.random.shuffle(outer_coords)

        # Add border clicks
        if border:
            selected_points = _select_points(np.array(border_coords), clicks_num, clicks_dst)
            for c in selected_points:
                border_clicks[slice,c[0], c[1]] = 1

        # Add bg clicks
        if bg:
            selected_points = _select_points(np.array(outer_coords), clicks_num, clicks_dst)
            for c in selected_points:
                bg_clicks[slice,c[0], c[1]] = 1

        # Add fg clicks
        if fg:
            selected_points = _select_points(np.array(inner_coords), clicks_num, clicks_dst)
            for c in selected_points:
                fg_clicks[slice,c[0], c[1]] = 1
    
    if border: 
        return torch.as_tensor(border_clicks)
        # return torch.as_tensor(border_clicks).unsqueeze(0)
    return torch.stack((torch.as_tensor(bg_clicks), torch.as_tensor(fg_clicks)), axis=0)


def get_new_depth(mask: torch.Tensor, img_dims: tuple[int]) -> tuple[int, int]:
    """ Calculates new (start, end indices) based on the position of the tumour """
    
    first, last = _get_glioma_indices(mask)

    # new starting position will be 0, when the tumour starts low enough or 
    # couple slices bellow the first index so the tumour isnt in the first few
    # slices of the new volume
    start_index = max(first - (img_dims[0] // 2), 0)
    
    # new end position is calculated so the final dimensions match with 
    # the requested ones in `self.img_dims`
    end_index = start_index + img_dims[0]
    
    return start_index, end_index
