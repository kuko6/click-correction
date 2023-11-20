import torch


def get_glioma_indices(mask: torch.Tensor) -> tuple[int, int]:
    """ Returns the first and last slice indices of the tumour in given mask """

    first = torch.nonzero((mask == 1))[:,1][0].item()
    last = torch.nonzero((mask == 1))[:,1][-1].item()

    return first, last