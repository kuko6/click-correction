import numpy as np

class TrainOptions:
    """Configuration used for training basic segmentation models."""

    def __init__(self, use_wand=False) -> None:
        self.use_wandb = use_wand
        self.config = {
            "lr": 1e-3,
            "img_channels": 2,
            "num_classes": 1,
            "conv_blocks": 3,  # 3 if device == 'cpu' else 4
            "dataset": "Schwannoma",
            "name": None,
            "epochs": 40,
            "batch_size": 2,
            "loss": "dice",
            "optimizer": "Adam",
            "augment": False,
            "scheduler": True,
            "early_stopper": True,
            "img_dims": (40, 128, 128),  # (64, 80, 80) if device == 'cpu' else (64, 128, 128)
            "training": "base",  # base, clicks-pretraining, clicks
            "train_size": 30,
            "val_size": 10,
            "clicks": {
                "use": False,
                "gen_fg": False,
                "gen_bg": False,
                "gen_border": True,
                "num": 10,
                "dst": 4,
            },
            "seed": 420
        }


class TrainCorrectionOptions:
    """Configuration used for training correction models."""
    
    def __init__(self, use_wand) -> None:
        self.use_wandb = use_wand
        self.config = {
            "lr": 1e-3,
            "img_channels": 1,
            "num_classes": 1,
            "conv_blocks": 3,  # 3 if device == 'cpu' else 4
            "dataset": "Schwannoma",
            "name": "correctionloss_32imgs",
            "epochs": 100,
            "batch_size": 4,
            "loss": "correction",
            "optimizer": "Adam",
            "augment": False,
            "scheduler": True,
            "early_stopper": True,
            "img_dims": (256, 256),
            "train_size": 32,
            "val_size": 8,
            "clicks": {"num": 3, "dst": 10},
            "cuts": {
                "num": np.inf,  # np.inf
                "size": 32,
                "random": False,
            },
            "seed": 420
        }
