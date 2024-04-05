class TrainOptions:
    def __init__(self, use_wand=False) -> None:
        self.use_wandb = use_wand
        self.config = {
            "lr": 1e-3,
            "img_channels": 2,
            "num_classes": 1,
            "conv_blocks": 3,  # 3 if device == 'cpu' else 4
            "dataset": "Schwannoma",
            "epochs": 40,
            "batch_size": 2,
            "loss": "dice",
            "optimizer": "Adam",
            "augment": False,
            "scheduler": True,
            "early_stopper": True,
            "img_dims": (40, 128, 128),  # (64, 80, 80) if device == 'cpu' else (64, 128, 128)
            "training": "clicks-pretraining",  # base, clicks-pretraining, clicks
            "clicks": {
                "use": False,
                "gen_fg": False,
                "gen_bg": False,
                "gen_border": True,
                "num": 10,
                "dst": 4,
            },
        }


class TrainCorrectionOptions:
    def __init__(self, use_wand) -> None:
        self.use_wandb = use_wand
        self.config = {
            "lr": 1e-3,
            "img_channels": 1,
            "num_classes": 1,
            "conv_blocks": 3,  # 3 if device == 'cpu' else 4
            "dataset": "Schwannoma",
            "epochs": 40,
            "batch_size": 2,
            "loss": "dice",
            "optimizer": "Adam",
            "augment": False,
            "scheduler": True,
            "early_stopper": True,
            "img_dims": (256, 256),
            "training": "base",  # base, clicks-pretraining, clicks
            "clicks": { "num": 3, "dst": 10 },
            "cuts": {
                "num": 12,  # np.inf
                "size": 32,
                "random": False,
            },
        }
