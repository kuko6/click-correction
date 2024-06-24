import argparse
import os
import glob
import wandb
# import json

from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torchinfo import summary

from model.segmentation import Unet
from model.correction import CorrectionUnet, MultiModalCorrectionUnet, MultiModal3BlockCorrectionUnet
# from data.correction_generator import CorrectionDataLoader, CorrectionMRIDataset, CorrectionMRIDatasetSequences
from data.data_generator import MRIDataset
from utils import EarlyStopper, make_output_dirs, preview_cuts, record_used_files, save_history
from losses.dice import dice_coefficient, DiceLoss
from losses.correction import CorrectionLoss
from options import FineTunningCorrectionOptions

from finetune_utils import simulate_clicks, generate_cuts, cut_volume

opt = FineTunningCorrectionOptions()
config = opt.config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Using {device} device]")

def prepare_cuts(segmentation_model, data, cut_size):
    with torch.no_grad():
        prepared_cuts = []
        for (x, y) in data:
            y_pred = segmentation_model(x.unsqueeze(0).to(device))
            y_pred = (y_pred > 0.6).type(torch.float32)
            # y_pred = y_pred.squeeze(0)

            y_pred = y_pred.cpu()
            new_clicks = simulate_clicks(y, y_pred[0], clicks_num=5, clicks_dst=10)
            seg_cuts, t1_cuts, t2_cuts = generate_cuts(
                y_pred[0, 0], x, new_clicks[0], cut_size=cut_size
            )

            true_seg_cuts = cut_volume(torch.stack((y[0], new_clicks[0])), cut_size=cut_size)
            # print(len(seg_cuts), len(t1_cuts), len(t2_cuts), len(true_seg_cuts))

            for seg_cut, t1_cut, t2_cut, true_seg_cut in zip(seg_cuts, t1_cuts, t2_cuts, true_seg_cuts):
                # training_cuts.append(torch.stack((seg_cut.squeeze(0), t1_cut.squeeze(0), t2_cut.squeeze(0))))
                prepared_cuts.append((
                    torch.stack((seg_cut.squeeze(0), t1_cut.squeeze(0), t2_cut.squeeze(0))),
                    true_seg_cut
                ))
    
    # print(len(prepared_cuts))
    return prepared_cuts


def create_batches(data, batch_size):
    batches = []
    x_batch = []
    y_batch = []
    for i, (x, y) in enumerate(data):    
        if i != 0 and i % batch_size == 0:
            batches.append((torch.stack(x_batch, dim=0), torch.stack(y_batch, dim=0)))
            x_batch = []
            y_batch = []

        x_batch.append(x)
        y_batch.append(y)

        if i == len(data) - 1:
            batches.append((torch.stack(x_batch, dim=0), torch.stack(y_batch, dim=0)))
    
    return batches


def prepare_data(data_dir: str, segmentation_model):
    """Loads the data from `data_dir` and returns `Dataset`."""

    t1_list = sorted(glob.glob(os.path.join(data_dir, "VS-*-*/vs_*/*_t1_*")))
    t2_list = sorted(glob.glob(os.path.join(data_dir, "VS-*-*/vs_*/*_t2_*")))
    seg_list = sorted(glob.glob(os.path.join(data_dir, "VS-*-*/vs_*/*_seg_*")))

    # seg_train, seg_val = train_test_split(seg_list, test_size=0.2, train_size=0.8, random_state=420)
    t1_train, t1_val, t2_train, t2_val, seg_train, seg_val = train_test_split(
        t1_list, t2_list, seg_list, test_size=0.2, train_size=0.8, random_state=420
    )

    train_data = MRIDataset(
        t1_train[: config["train_size"]],
        t2_train[: config["train_size"]],
        seg_train[: config["train_size"]],
        (48, 256, 256),
        clicks=None,
    )

    val_data = MRIDataset(
        t1_val[: config["val_size"]],
        t2_val[: config["val_size"]],
        seg_val[: config["val_size"]],
        (48, 256, 256),
        clicks=None,
    )

    training_cuts = prepare_cuts(segmentation_model, train_data, cut_size=48)
    validation_cuts = prepare_cuts(segmentation_model, val_data, cut_size=48)
    print(len(train_data), len(training_cuts), len(val_data), len(validation_cuts))

    train_batches = create_batches(training_cuts, batch_size=config["batch_size"])
    val_batches = create_batches(validation_cuts, batch_size=config["batch_size"])

    print(f"train size: {len(training_cuts)}, train batches: {len(train_batches)}")
    # for (x, y) in train_batches:
    #     print(x.shape, y.shape)
    
    print(f"val size: {len(validation_cuts)}, val batches: {len(val_batches)}")
    # for (x, y) in val_batches:
    #     print(x.shape, y.shape)

    if config["use_seq"]:
        record_used_files(
            path=f"outputs/{opt.name}", 
            labels=["t1", "t2", "seg"], 
            train_files=list(zip(t1_train[:config["train_size"]], t2_train[:config["train_size"]], seg_train[:config["train_size"]])),
            val_files=list(zip(t1_val[:config["val_size"]], t2_val[:config["val_size"]], seg_val[:config["val_size"]]))
        )

    return train_batches, val_batches

def val(data, model, loss_fn: torch.nn.Module, epoch: int) -> tuple[float, float]:
    """Validate model after each epoch on validation dataset, returns the avg. loss and avg. dice."""

    model.eval()
    avg_loss, avg_dice = 0, 0
    with torch.no_grad():
        for i, (x, y) in enumerate(data):
            x, y = x.to(device), y.to(device)
            y_pred = model(x)

            # Compute loss and dice coefficient
            loss = loss_fn(y_pred, y)
            dice = dice_coefficient(y_pred, y)

            avg_loss += loss.item()
            avg_dice += dice.item()

            print(f"validation step: {i+1}/{len(data)}, loss: {loss.item():>5f}, dice: {dice.item():>5f}", end="\r")

            if i==0 and not config["cuts"]["volumetric"]:
                preview_cuts(y_pred, x, y, dice.item(), output_path=f"outputs/images/{epoch}_preview.png")

    avg_loss /= len(data)
    avg_dice /= len(data)
    print()

    return (avg_loss, avg_dice)


def train_one_epoch(data, model, loss_fn: torch.nn.Module, optimizer, epoch) -> tuple[float, float]:
    """Train model for one epoch on the training dataset, returns the avg. loss and avg. dice."""

    model.train()
    avg_loss, avg_dice = 0, 0
    for i, (x, y) in enumerate(data):
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

        print(f"training step: {i+1}/{len(data)}, loss: {loss.item():>5f}, dice: {dice.item():>5f}", end="\r")

    avg_loss /= len(data)
    avg_dice /= len(data)
    print()

    return (avg_loss, avg_dice)


def train(train_data, val_data, model, loss_fn, optimizer, scheduler):
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
            train_data, model, loss_fn, optimizer, epoch
        )
        print("-------------------------------")
        val_loss, val_dice = val(val_data, model, loss_fn, epoch)

        print("-------------------------------")
        print(f"loss: {train_loss:>5f} dice: {train_dice:>5f}")
        print(f"val loss: {val_loss:>5f} val dice: {val_dice:>5f}")

        # Log training and validation history
        train_history["loss"].append(train_loss)
        train_history["dice"].append(train_dice)

        val_history["loss"].append(val_loss)
        val_history["dice"].append(val_dice)

        save_history(f"outputs/{opt.name}", train_history, val_history)

        # Save checkpoint
        model_checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }

        torch.save(model_checkpoint, f"outputs/{opt.name}/checkpoint.pt")

        # Save best checkpoint
        if best["dice"] < val_dice:
            print("-------------------------------")
            print(f'new best!!! (loss: {best["loss"]:>5f} -> {val_loss:>5f}, dice: {best["dice"]:>5f} -> {val_dice:>5f})')

            torch.save(model_checkpoint, f"outputs/{opt.name}/best.pt")

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
        artifact.add_file(f"outputs/{opt.name}/best.pt")
        wandb.run.log_artifact(artifact)
        wandb.finish()


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, help="path to data")
    parser.add_argument("--seg-model", type=str, help="path to pretrained segmentation model")
    parser.add_argument("--corr-model", type=str, help="path to pretrained correction model")
    parser.add_argument("--use-wandb", type=str, help="whether to use wandb")
    parser.add_argument("--wandb-key", type=str, help="wandb id")
    parser.add_argument("--name", type=str, help="name of the experiment")

    args = parser.parse_args()
    # print(args.wandb)

    if not args.data_path:
        print("You need to specify datapath!!!! >:(")
        return

    # wandb_key = args.wandb_key
    global opt
    if args.use_wandb is not None or args.wandb_key is not None:
        opt.use_wandb = True
    
    if args.name is not None:
        opt.name = args.name
    
    if opt.use_wandb:
        wandb.login(key=args.wandb_key)
        wandb.init(
            project="DP",
            entity="kuko",
            reinit=True,
            name=opt.name,
            config=config,
            tags=opt.tags,
        )

    data_dir = args.data_path
    print(os.listdir(data_dir))

    make_output_dirs(["outputs", f"outputs/{opt.name}", "outputs/images"])

    # Load pretrained seg. model
    segmentation_model = Unet(in_channels=2, out_channels=1, blocks=3).to(device)
    checkpoint = torch.load(args.seg_model, map_location=device)
    segmentation_model.load_state_dict(checkpoint["model_state"])

    # Initialize model
    if config["model"] == "standard":
        model = CorrectionUnet(
            in_channels=config["img_channels"],
            out_channels=config["num_classes"],
            blocks=config["conv_blocks"],
            volumetric=config["cuts"]["volumetric"],
            block_channels=config["block_channels"],
            use_attention=config["use_attention"],
            use_dropout=config["use_dropout"]
        )
    elif config["model"] == "multimodal":
        if config["conv_blocks"] == 3:
            model = MultiModal3BlockCorrectionUnet(
                in_channels=config["in_channels"],
                out_channels=config["num_classes"],
                blocks=config["conv_blocks"],
                encoders=2,
                volumetric=config["cuts"]["volumetric"],
                block_channels=config["block_channels"],
                use_dropout=config["use_dropout"]
            )
        else:
            model = MultiModalCorrectionUnet(
                in_channels=config["in_channels"],
                out_channels=config["num_classes"],
                blocks=config["conv_blocks"],
                encoders=2,
                volumetric=config["cuts"]["volumetric"],
                block_channels=config["block_channels"],
                use_dropout=config["use_dropout"]
            )

    # model = MultiModalCorrectionUnet(in_channels=[1, 2], out_channels=1, blocks=3, encoders=2, block_channels=[32, 64, 128, 256], use_dropout=True).to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    model.to(device)

    # writes model architecture to a file (just for experiment logging)
    with open(f"outputs/{opt.name}/architecture.txt", "w") as f:
        sample_input = [config["batch_size"], config["img_channels"], config["cuts"]["size"], config["cuts"]["size"]]
        if config["cuts"]["volumetric"]:
            sample_input.insert(2, config["cuts"]["cut_depth"]*2)
        model_summary = summary(
            model,
            input_size=sample_input,
            verbose=0,
        )
        f.write(str(model_summary))
    
    # Load pretrained corr. model
    print("Using pretrained model")
    checkpoint = torch.load(args.corr_model, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    
    # Select loss function
    loss_functions = {
        "dice": DiceLoss(),
        "correction": CorrectionLoss(
            dims=(1, config["cuts"]["size"], config["cuts"]["size"]),
            device=device,
            inverted=False,
        ),
        "invertedCorrection": CorrectionLoss(
            dims=(1, config["cuts"]["size"], config["cuts"]["size"]),
            device=device,
            inverted=True,
        ),
    }

    loss_fn = loss_functions[config["loss"]]

    # Prepare Datasets
    train_data, val_data = prepare_data(data_dir, segmentation_model)

    # Train :D
    train(train_data, val_data, model, loss_fn, optimizer, scheduler)


if __name__ == "__main__":
    main()
