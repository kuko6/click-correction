import argparse
import os
import glob
import wandb
import json

from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchinfo import summary

from model.correction import CorrectionUnet
from data.correction_generator import CorrectionDataLoader, CorrectionMRIDataset
from utils import EarlyStopper
from losses.dice import dice_coefficient, DiceLoss
from losses.correction import CorrectionLoss
from options import TrainCorrectionOptions

opt = TrainCorrectionOptions(use_wand=True)
config = opt.config

device = torch.device("cuda" if torch.cuda.is_available() else "mps")
print(f"[Using {device} device]")


def prepare_data(data_dir: str) -> tuple[CorrectionDataLoader, CorrectionDataLoader]:
    """Loads the data from `data_dir` and returns `Dataset`."""

    seg_list = sorted(glob.glob(os.path.join(data_dir, "VS-*-*/vs_*/*_seg_*")))
    
    seg_train, seg_val = train_test_split(seg_list, test_size=0.2, train_size=0.8, random_state=420)
    seg_val.append(seg_train.pop(-1))

    # if config['clicks']['use']:
    #     preview_clicks(t1_list, t2_list, seg_list, config['clicks'])

    train_data = CorrectionMRIDataset(
        seg_train[:config["train_size"]],
        config["img_dims"],
        clicks=config["clicks"],
        cuts=config["cuts"],
        seed=config["seed"]
    )
    val_data = CorrectionMRIDataset(
        seg_val[:config["val_size"]],
        config["img_dims"],
        clicks=config["clicks"],
        cuts=config["cuts"],
        seed=config["seed"]
    )
    print(len(train_data), len(val_data))

    train_dataloader = CorrectionDataLoader(train_data, batch_size=config["batch_size"])
    val_dataloader = CorrectionDataLoader(val_data, batch_size=config["batch_size"])

    # print(train_data[0])
    # for i, (x, y) in enumerate(train_dataloader):
    #     print(i)
    #     break

    # TODO: format the output
    np.savetxt(
        "outputs/training_files.csv",
        [('seg')]+list(seg_train[:config["train_size"]]),
        delimiter =", ",
        fmt ='% s'
    )

    np.savetxt(
        "outputs/validation_files.csv",
        [('seg')]+list(seg_val[:config["val_size"]]),
        delimiter =", ",
        fmt ='% s'
    )

    return train_dataloader, val_dataloader


def val(dataloader: DataLoader, model: CorrectionUnet, loss_fn: torch.nn.Module, epoch: int) -> tuple[float, float]:
    """Validate model after each epoch on validation dataset, returns the avg. loss and avg. dice."""

    model.eval()
    avg_loss, avg_dice = 0, 0
    dataloader_size = len(dataloader)
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            y_pred = model(x)

            # Compute loss and dice coefficient
            loss = loss_fn(y_pred, y)
            dice = dice_coefficient(y_pred, y)

            avg_loss += loss.item()
            avg_dice += dice.item()

            print(f"validation step: {i+1}/{dataloader_size}, loss: {loss.item():>5f}, dice: {dice.item():>5f}", end="\r")

            # if i==0:
            #     # preview(y_pred[0], y[0], dice_coefficient2d2d(y_pred, y), epoch)
            #     preview(y_pred[0], y[0], dice.item(), epoch)

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
        loss = loss_fn(y_pred, y)
        dice = dice_coefficient(y_pred, y)
        # print(loss.shape)
        # print(dice.shape)

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
    val_dataloader: DataLoader,
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

        with open(f"outputs/{config['name']}/train_history.json", "w") as f:
            json.dump(train_history, f)
        with open(f"outputs/{config['name']}/val_history.json", "w") as f:
            json.dump(val_history, f)

        # Save checkpoint
        model_checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }

        torch.save(model_checkpoint, f"outputs/{config['name']}/checkpoint.pt")

        # Save best checkpoint
        if best["dice"] < val_dice:
            print("-------------------------------")
            print(f'new best!!! (loss: {best["loss"]:>5f} -> {val_loss:>5f}, dice: {best["dice"]:>5f} -> {val_dice:>5f})')

            torch.save(model_checkpoint, f"outputs/{config['name']}/best.pt")

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
        artifact = wandb.Artifact("best_model", type="model", metadata={"val_dice": val_dice})
        artifact.add_file(f"outputs/{config['name']}/best.pt")
        wandb.run.log_artifact(artifact)
        wandb.finish()


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, help="path to data")
    parser.add_argument("--model-path", type=str, help="path to pretrained model")
    parser.add_argument("--wandb", type=str, help="wandb id")

    args = parser.parse_args()
    # print(args.wandb)

    if not args.data_path:
        print("You need to specify datapath!!!! >:(")
        return

    wandb_key = args.wandb
    if opt.use_wandb and wandb_key:
        wandb.login(key=wandb_key)
        wandb.init(project="DP", entity="kuko", reinit=True, name=config['name'], config=config)

    data_dir = args.data_path
    print(os.listdir(data_dir))

    if not os.path.isdir("outputs"):
        os.mkdir("outputs")
    if not os.path.isdir(f"outputs/{config['name']}"):
        os.mkdir(f"outputs/{config['name']}")
    if not os.path.isdir("outputs/images"):
        os.mkdir("outputs/images")

    # Prepare Datasets
    train_dataloader, val_dataloader = prepare_data(data_dir)

    # # Initialize model
    model = CorrectionUnet(
        in_channels=config["img_channels"],
        out_channels=config["num_classes"],
        blocks=config["conv_blocks"],
    ).to(device)

    # writes model architecture to a file (just for experiment logging)
    with open("outputs/architecture.txt", "w") as f:
        model_summary = summary(
            CorrectionUnet(
                in_channels=config["img_channels"],
                out_channels=config["num_classes"],
                blocks=config["conv_blocks"],
            ),
            input_size=(
                config["batch_size"],
                config["img_channels"],
                *config["img_dims"],
            ),
            verbose=0,
        )
        f.write(str(model_summary))

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.1)

    # Select loss function
    loss_functions = {
        "dice": DiceLoss(),
        "correction": CorrectionLoss(dims=(1, 32, 32), device=device, batch_size=config["batch_size"], inverted=False),
        "invertedCorrection": CorrectionLoss(dims=(1, 32, 32), device=device, batch_size=config["batch_size"], inverted=True)
    }

    loss_fn = loss_functions[config["loss"]]

    # for i, (x, y) in enumerate(train_dataloader):
    #     print(i, y.shape)
    #     return

    # Train :D
    train(train_dataloader, val_dataloader, model, loss_fn, optimizer, scheduler)


if __name__ == "__main__":
    main()
