# import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt


def get_glioma_indices(mask: torch.Tensor) -> tuple[int, int]:
    """
    Returns the first and last slice indices of the tumour in given mask.

    Args:
        mask (Tensor): segmentation mask in shape (depth, width, height)
    Returns:
        tuple of ints: the first and last indices of the glioma
    """

    glioma_indices = torch.nonzero((mask > 0))[:, 1]
    # first = torch.nonzero((mask > 0))[:,1][0].item()
    # last = torch.nonzero((mask > 0))[:,1][-1].item()
    if len(glioma_indices) == 0:
        return 0, 0

    first = glioma_indices[0].item()
    last = glioma_indices[-1].item()

    return first, last


def preview(y_pred: torch.Tensor, y: torch.Tensor, dice: torch.Tensor, epoch=0):
    """Saves a png of sample prediction `y_pred` for scan `y`."""

    # Compute number of slices with the tumour
    first, last = get_glioma_indices(y)
    length = last - first + 1
    n_graphs = (length * 2) // 6
    rows = n_graphs
    cols = 6
    res = cols if cols > rows else rows

    # Plot them
    fig, axs = plt.subplots(rows, cols, figsize=(res * 2, res * 2))
    axs = axs.flatten()
    j = 0
    for i in range(first, last):
        if j >= len(axs):
            break

        axs[j].imshow(y[0, i, :, :].cpu().detach(), cmap="magma")
        axs[j].axis("off")
        axs[j].set_title(f"mask slice {i}", fontsize=9)

        axs[j + 1].imshow(y_pred[0, i, :, :].cpu().detach(), cmap="magma")
        axs[j + 1].axis("off")
        axs[j + 1].set_title(f"pred slice {i}", fontsize=9)

        if y.shape[0] == 2:
            axs[j + 2].imshow(y[1, i, :, :].cpu().detach(), cmap="magma")
            axs[j + 2].axis("off")
            axs[j + 2].set_title(f"clicks slice {i}", fontsize=9)
            j += 3
        else:
            j += 2

    fig.suptitle(f"Dice: {dice}", fontsize=10)
    plt.subplots_adjust(top=0.9)

    fig.savefig(f"outputs/images/{epoch}_preview.png")
    plt.close(fig)


def plot_tumour(mask: torch.Tensor):
    """Plot individual slices of the tumour in one figure."""
    
    # Compute number of slices with the tumour
    first, last = get_glioma_indices(mask)
    print("Tumour indices: ", first, last)

    length = last - first + 1
    n_graphs = (length) // 4
    rows = n_graphs
    cols = 4
    res = cols if cols > rows else rows

    # Plot them
    fig, axs = plt.subplots(rows, cols, figsize=(res * 2, res * 2))
    axs = axs.flatten()
    j = 0
    for i in range(first, last):
        if j >= len(axs):
            break
        axs[j].imshow(mask[0, i, :, :], cmap="magma")
        axs[j].axis("off")
        axs[j].set_title(f"mask slice {i}", fontsize=9)
        j += 1

    plt.show()


# https://stackoverflow.com/a/73704579
class EarlyStopper:
    """Early stopper for training. Supports `'min'` and `'max'` mode."""

    def __init__(self, patience=1, delta=0.0, mode="min"):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.min_value = np.inf
        self.max_value = 0
        self.mode = mode

    def __call__(self, value):
        if self.mode == "min":
            if value < self.min_value:
                self.min_value = value
                self.counter = 0
            elif value > (self.min_value + self.delta):
                self.counter += 1
                print("-------------------------------")
                print(f"early stopping: {self.counter}/{self.patience}")

        elif self.mode == "max":
            if value > self.max_value:
                self.max_value = value
                self.counter = 0
            elif value < (self.max_value - self.delta):
                self.counter += 1
                print("-------------------------------")
                print(f"early stopping: {self.counter}/{self.patience}")

        if self.counter >= self.patience:
            return True

        return False
