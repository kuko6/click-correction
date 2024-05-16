import numpy as np


class TrainOptions:
    """Configuration used for training basic segmentation models."""

    def __init__(self, use_wand=False, name="") -> None:
        self.use_wandb = use_wand
        self.name = name
        self.tags = ["active-lr"]
        self.config = {
            "lr": 1e-3,
            "img_channels": 2,
            "num_classes": 1,
            "conv_blocks": 3,  # 3 if device == 'cpu' else 4
            "epochs": 50,
            "batch_size": 2,
            "loss": "focaltversky",
            "optimizer": "Adam",
            "augment": False,
            "scheduler": True,
            "early_stopper": False,
            "img_dims": (40, 256, 256), 
            "training": "clicks-pretraining",  # base, clicks-pretraining, clicks
            "train_size": 32,
            "val_size": 8,
            "clicks": {
                "use": False,
                "gen_fg": False,
                "gen_bg": False,
                "gen_border": False,
                "num": 10,
                "dst": 4,
            },
            "seed": 420,
        }


class TrainCorrectionOptions:
    """Configuration used for training correction models."""

    def __init__(self, use_wand=False, name="") -> None:
        self.use_wandb = use_wand
        self.name = name
        self.tags = ["correction"]
        self.config = {
            "lr": 1e-3,
            "model": "standard", # standard, multimodal
            "in_channels": [1, 2], # 3
            "img_channels": 3,
            "num_classes": 1,
            "conv_blocks": 3,
            "block_channels": [32, 64, 128, 256],
            "use_seq": True,
            "use_attention": True,
            "use_dropout": True,
            "epochs": 50,
            "batch_size": 4,
            "loss": "dice",
            "optimizer": "Adam",
            "scheduler": True,
            "early_stopper": False,
            "img_dims": (256, 256),
            "train_size": None, # 128, None
            "val_size": None, # 32, None
            "clicks": {"num": 5, "dst": 10},
            "cuts": {
                "num": np.inf,  # np.inf, 30
                "size": 48,
                "volumetric": False,
                "cut_depth": None
            },
            "random": True,
            "include_unchanged": True,
            "augment": False,
            "seed": 690,
        }

class FineTunningCorrectionOptions:
    """Configuration used for training correction models."""

    def __init__(self, use_wand=False, name="") -> None:
        self.use_wandb = use_wand
        self.name = name
        self.tags = ["correction"]
        self.config = {
            "lr": 1e-3,
            "model": "multimodal", # standard, multimodal
            "in_channels": [1, 2], # 3
            "img_channels": 3,
            "num_classes": 1,
            "conv_blocks": 3,
            "block_channels": [32, 64, 128, 256],
            "use_seq": True,
            "use_attention": True,
            "use_dropout": True,
            "epochs": 50,
            "batch_size": 4,
            "loss": "dice",
            "optimizer": "Adam",
            "scheduler": True,
            "early_stopper": False,
            "img_dims": (256, 256),
            "train_size": None, # 128, None
            "val_size": None, # 32, None
            "clicks": {"num": 5, "dst": 10},
            "cuts": {
                "num": 10,  # np.inf, 30
                "size": 40,
                "volumetric": False,
                "cut_depth": None
            },
            "random": True,
            "include_unchanged": True,
            "augment": True,
            "seed": 690,
        }


class TestCorrectionOptions:
    """Configuration used for training correction models."""

    def __init__(self, use_wand=False, name="") -> None:
        self.use_wandb = False
        self.name = name
        self.tags = ["correction"]
        self.config = {
            "lr": 1e-3,
            "model": "standard", # standard, multimodal
            "in_channels": [1, 2], # 3
            "img_channels": 3,
            "num_classes": 1,
            "conv_blocks": 3,
            "block_channels": [32, 64, 128, 256],
            "use_seq": True,
            "use_attention": True,
            "use_dropout": True,
            "epochs": 50,
            "batch_size": 2,
            "loss": "dice",
            "optimizer": "Adam",
            "scheduler": True,
            "early_stopper": False,
            "img_dims": (256, 256),
            "test_size": 30, # 128, None
            "clicks": {"num": 5, "dst": 10},
            "cuts": {
                "num": 10,  # np.inf, 30
                "size": 32,
                "volumetric": False,
                "cut_depth": None
            },
            "random": True,
            "include_unchanged": True,
            "augment": True,
            "seed": 690,
        }


class TrainCorrectionSweepOptions:
    """Configuration used for wandb sweep for training correction models."""

    def __init__(self, use_wand=False, name="") -> None:
        self.use_wandb = use_wand
        self.name = name
        self.tags = ["correction-sweep"]
        self.config = {
            "lr": { "values": [1e-3] },
            "img_channels": { "values": [3] },
            "num_classes": { "values": [1] },
            "conv_blocks": { "values": [3] },
            "use_seq": { "values": [True] },
            "epochs": { "values": [40] },
            "batch_size": { "values": [4] },
            "loss": { "values": ["correction", "dice"] },
            "optimizer": { "values": ["Adam"] },
            "scheduler": { "values": [True] },
            "early_stopper": { "values": [False] },
            "img_dims": { "values": [(256, 256)] },
            # "train_size": { "values": [32, 64, 80] },
            # "val_size": { "values": [8, 16, 32] },
            "train_size": { "values": [64] },
            "val_size": { "values": [16] },
            "clicks": { "values": [{"num": 5, "dst": 10}] },
            "cuts": { "values": [{"num": np.inf, "size": 32}, {"num": np.inf, "size": 40}, {"num": np.inf, "size": 64}] },
            "random": { "values": [False] },
            "include_unchanged": { "values": [True] },
            "augment": { "values": [True] },
            "seed": { "values": [690] },
        }
