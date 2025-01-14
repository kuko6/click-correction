import argparse
import glob
import os

import numpy as np
import torch
import wandb

# import json
from sklearn.model_selection import train_test_split
from torchinfo import summary

from data.correction_generator import (
    CorrectionDataLoader,
    CorrectionMRIDataset,
    CorrectionMRIDatasetSequences,
)
from losses.correction import CorrectionLoss
from losses.dice import DiceLoss, dice_coefficient
from model.correction import CorrectionUnet
from options import TrainCorrectionSweepOptions
from utils import (
    EarlyStopper,
    make_output_dirs,
    preview_cuts,
    record_used_files,
    save_history,
)

opt = TrainCorrectionSweepOptions()
config = opt.config

device = torch.device("cuda" if torch.cuda.is_available() else "mps")
print(f"[Using {device} device]")


def prepare_data(data_dir: str) -> tuple[CorrectionDataLoader, CorrectionDataLoader]:
    """Loads the data from `data_dir` and returns `Dataset`."""

    t1_list = sorted(glob.glob(os.path.join(data_dir, "VS-*-*/vs_*/*_t1_*")))
    t2_list = sorted(glob.glob(os.path.join(data_dir, "VS-*-*/vs_*/*_t2_*")))
    seg_list = sorted(glob.glob(os.path.join(data_dir, "VS-*-*/vs_*/*_seg_*")))

    # seg_train, seg_val = train_test_split(seg_list, test_size=0.2, train_size=0.8, random_state=420)
    t1_train, t1_val, t2_train, t2_val, seg_train, seg_val = train_test_split(
        t1_list, t2_list, seg_list, test_size=0.2, train_size=0.8, random_state=420
    )

    if config["use_seq"]:
        print("Using dataset with sequences")
        train_data = CorrectionMRIDatasetSequences(
            t1_train[:config["train_size"]],
            t2_train[:config["train_size"]],
            seg_train[:config["train_size"]],
            config["img_dims"],
            clicks=config["clicks"],
            cuts=config["cuts"],
            random=config["random"],
            include_unchanged=config["include_unchanged"],
            augment=config["augment"],
            seed=config["seed"]
        )
        val_data = CorrectionMRIDatasetSequences(
            t1_val[:config["val_size"]],
            t2_val[:config["val_size"]],
            seg_val[:config["val_size"]],
            config["img_dims"],
            clicks=config["clicks"],
            cuts=config["cuts"],
            random=False,
            include_unchanged=config["include_unchanged"],
            augment=False,
            seed=config["seed"]
        )
    else:
        print("Using dataset without sequences")
        train_data = CorrectionMRIDataset(
            seg_train[:config["train_size"]],
            config["img_dims"],
            clicks=config["clicks"],
            cuts=config["cuts"],
            random=config["random"],
            include_unchanged=config["include_unchanged"],
            augment=config["augment"],
            seed=config["seed"]
        )
        val_data = CorrectionMRIDataset(
            seg_val[:config["val_size"]],
            config["img_dims"],
            clicks=config["clicks"],
            cuts=config["cuts"],
            random=False,
            include_unchanged=config["include_unchanged"],
            augment=False,
            seed=config["seed"]
        )
    print(len(train_data), len(val_data))

    train_dataloader = CorrectionDataLoader(train_data, batch_size=config["batch_size"])
    val_dataloader = CorrectionDataLoader(val_data, batch_size=config["batch_size"])
    print(f"~{len(train_dataloader)*2}, ~{len(val_dataloader)*2}")

    return train_dataloader, val_dataloader


def val(dataloader: CorrectionDataLoader, model: CorrectionUnet, loss_fn: torch.nn.Module, epoch: int) -> tuple[float, float]:
    """Validate model after each epoch on validation dataset, returns the avg. loss and avg. dice."""

    model.eval()
    avg_loss, avg_dice = 0, 0
    dataloader_size = len(dataloader)
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            y_pred = model(x)

            # Compute loss and dice coefficient
            if config["use_seq"]:
                y = y[:,0].unsqueeze(1)
                y = y[:,0].unsqueeze(1)

            loss = loss_fn(y_pred, y)
            dice = dice_coefficient(y_pred, y)

            avg_loss += loss.item()
            avg_dice += dice.item()

            print(f"validation step: {i+1}/{dataloader_size}, loss: {loss.item():>5f}, dice: {dice.item():>5f}", end="\r")

    avg_loss /= dataloader_size
    avg_dice /= dataloader_size
    print()

    return (avg_loss, avg_dice)


def train_one_epoch(dataloader: CorrectionDataLoader, model: CorrectionUnet, loss_fn: torch.nn.Module, optimizer, epoch) -> tuple[float, float]:
    """Train model for one epoch on the training dataset, returns the avg. loss and avg. dice."""

    model.train()
    avg_loss, avg_dice = 0, 0
    dataloader_size = len(dataloader)
    for i, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        # Get prediction
        y_pred = model(x)

        # Compute loss and dice coefficient
        if config["use_seq"]:
            y = y[:,0].unsqueeze(1)
            y = y[:,0].unsqueeze(1)
            
        loss = loss_fn(y_pred, y)
        dice = dice_coefficient(y_pred, y)

        avg_loss += loss.item()
        avg_dice += dice.item()

        # Update parameters
        loss.backward()
        optimizer.step()

        print(f"training step: {i+1}/{dataloader_size}, loss: {loss.item():>5f}, dice: {dice.item():>5f}", end="\r")

    avg_loss /= dataloader_size
    avg_dice /= dataloader_size
    print()

    return (avg_loss, avg_dice)


def train(
    train_dataloader: CorrectionDataLoader, 
    val_dataloader: CorrectionDataLoader,
    model: CorrectionUnet, 
    loss_fn, 
    optimizer, 
    scheduler
):
    """Run the training."""

    epochs = config["epochs"]
    train_history = {"loss": [], "dice": []}
    val_history = {"loss": [], "dice": []}
    best = {"loss": np.inf, "dice": 0, "epoch": 0}

    if config["early_stopper"]:
        early_stopper = EarlyStopper(patience=6, delta=0.01, mode="max")

    for epoch in range(epochs):
        print("===============================")
        print(f"[Epoch: {epoch}]")

        # Train and validate
        train_loss, train_dice = train_one_epoch(
            train_dataloader, model, loss_fn, optimizer, epoch
        )
        print("-------------------------------")
        val_loss, val_dice = val(val_dataloader, model, loss_fn, epoch)

        print("-------------------------------")
        print(f"loss: {train_loss:>5f} dice: {train_dice:>5f}")
        print(f"val loss: {val_loss:>5f} val dice: {val_dice:>5f}")

        # Log training and validation history
        train_history["loss"].append(train_loss)
        train_history["dice"].append(train_dice)

        val_history["loss"].append(val_loss)
        val_history["dice"].append(val_dice)

        # save_history(f"outputs/{opt.name}", train_history, val_history)

        # Save checkpoint
        # model_checkpoint = {
        #     "epoch": epoch,
        #     "model_state": model.state_dict(),
        #     "optimizer_state": optimizer.state_dict(),
        # }

        # torch.save(model_checkpoint, f"outputs/{opt.name}/checkpoint.pt")

        # Save best checkpoint
        if best["dice"] < val_dice:
            print("-------------------------------")
            print(f'new best!!! (loss: {best["loss"]:>5f} -> {val_loss:>5f}, dice: {best["dice"]:>5f} -> {val_dice:>5f})')

            # torch.save(model_checkpoint, f"outputs/{opt.name}/best.pt")

            best["dice"] = val_dice
            best["loss"] = val_loss
            best["epoch"] = epoch
            # best['model'] = model_checkpoint

        # Log to wandb
        if opt.use_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "loss": train_loss,
                    "dice": train_dice,
                    "val_loss": val_loss,
                    "val_dice": val_dice,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )
            # wandb.log({"preview": wandb.Image(f"outputs/images/{epoch}_preview.png")})

        # Run scheduler and early stopper
        if config["scheduler"]:
            scheduler.step(val_loss)

        if config["early_stopper"]:
            if early_stopper(val_dice):
                print("===============================")
                print("Stopping early!!!")
                break

    print("===============================")
    print(f'The best model was in epoch {best["epoch"]} with loss: {best["loss"]:>5f} and dice: {best["dice"]:>5f}')

    if opt.use_wandb:
        # artifact = wandb.Artifact("best_model", type="model", metadata={"val_dice": val_dice})
        # artifact.add_file(f"outputs/{opt.name}/best.pt")
        # wandb.run.log_artifact(artifact)
        wandb.finish()


def start_training():
    # Parse remaining arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, help="path to data")
    parser.add_argument("--model-path", type=str, help="path to pretrained model")
    parser.add_argument("--use-wandb", type=bool, help="whether to use wandb")
    parser.add_argument("--wandb-key", type=str, help="wandb id")
    parser.add_argument("--name", type=str, help="name of the experiment")
    args = parser.parse_args()

    # Prepare Datasets
    train_dataloader, val_dataloader = prepare_data(args.data_path)

    # # Initialize model
    model = CorrectionUnet(
        in_channels=config["img_channels"],
        out_channels=config["num_classes"],
        blocks=config["conv_blocks"],
    ).to(device)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    # Select loss function
    loss_functions = {
        "dice": DiceLoss(),
        "correction": CorrectionLoss(
            dims=(1, config["cuts"]["size"], config["cuts"]["size"]),
            device=device,
            batch_size=config["batch_size"],
            inverted=False,
        ),
        "invertedCorrection": CorrectionLoss(
            dims=(1, config["cuts"]["size"], config["cuts"]["size"]),
            device=device,
            batch_size=config["batch_size"],
            inverted=True,
        ),
    }

    loss_fn = loss_functions[config["loss"]]

    # Train :D
    train(train_dataloader, val_dataloader, model, loss_fn, optimizer, scheduler)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, help="path to data")
    parser.add_argument("--model-path", type=str, help="path to pretrained model")
    parser.add_argument("--use-wandb", type=bool, help="whether to use wandb")
    parser.add_argument("--wandb-key", type=str, help="wandb id")
    parser.add_argument("--name", type=str, help="name of the experiment")
    args = parser.parse_args()

    if not args.data_path:
        print("You need to specify datapath!!!! >:(")
        return

    global opt
    if args.use_wandb is not None or args.wandb_key is not None:
        opt.use_wandb = True
    
    if args.name is not None:
        opt.name = args.name

    sweep_config = {
        "method": "random",
        "name": opt.name,
        "metric": {'goal': 'maximize', 'name': 'val_dice'},
    }
    sweep_config['parameters'] = config

    print(sweep_config)

    wandb.login(key=args.wandb_key)
    sweep_id = wandb.sweep(sweep=sweep_config, entity="kuko", project="DP-sweep")
    print(sweep_id)
    wandb.agent(sweep_id, start_training)


if __name__ == "__main__":
    main()
