import argparse
import glob
import os

import torch
from torchinfo import summary

from data.data_generator import MRIDataset
from finetune_utils import cut_volume, generate_cuts, simulate_clicks
from losses.correction import CorrectionLoss
from losses.dice import DiceLoss, dice_coefficient
from model.correction import (
    CorrectionUnet,
    MultiModal3BlockCorrectionUnet,
    MultiModalCorrectionUnet,
)
from model.segmentation import Unet
from options import TestCorrectionOptions
from utils import make_output_dirs

opt = TestCorrectionOptions()
config = opt.config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Using {device} device]")

def prepare_cuts(segmentation_model, data, cut_size):
    with torch.no_grad():
        prepared_cuts = []
        for (x, y) in data:
            y_pred = segmentation_model(x.unsqueeze(0).to(device))
            y_pred = (y_pred > 0.6).type(torch.float32)

            y_pred = y_pred.cpu()
            new_clicks = simulate_clicks(y, y_pred[0], clicks_num=5, clicks_dst=10, seed=420)
            seg_cuts, t1_cuts, t2_cuts = generate_cuts(
                y_pred[0, 0], x, new_clicks[0], cut_size=cut_size
            )

            true_seg_cuts = cut_volume(torch.stack((y[0], new_clicks[0])), cut_size=cut_size)

            for seg_cut, t1_cut, t2_cut, true_seg_cut in zip(seg_cuts, t1_cuts, t2_cuts, true_seg_cuts):
                prepared_cuts.append((
                    torch.stack((seg_cut.squeeze(0), t1_cut.squeeze(0), t2_cut.squeeze(0))),
                    true_seg_cut
                ))
    
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

    test_data = MRIDataset(
        t1_list[: config["test_size"]],
        t2_list[: config["test_size"]],
        seg_list[: config["test_size"]],
        (48, 256, 256),
        clicks=None,
    )

    testing_cuts = prepare_cuts(segmentation_model, test_data, cut_size=48)
    print(len(test_data), len(testing_cuts))

    test_batches = create_batches(testing_cuts, batch_size=config["batch_size"])
    print(f"train size: {len(testing_cuts)}, train batches: {len(test_batches)}")

    return test_batches


def test(data, model, loss_fn: torch.nn.Module) -> tuple[float, float]:
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

    avg_loss /= len(data)
    avg_dice /= len(data)
    print()

    return (avg_loss, avg_dice)


def evaluate(test_data, model, loss_fn):
    """Run the evaluation."""

    # for epoch in range(epochs):
    print("===============================")

    # Evaluate
    test_loss, test_dice = test(test_data, model, loss_fn)

    print("-------------------------------")
    print(f"average test loss: {test_loss:>5f} average test dice: {test_dice:>5f}")

    print("===============================")


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

    global opt

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

    # Initialize optimizer
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
    test_data = prepare_data(data_dir, segmentation_model)

    # Evaluate :D
    evaluate(test_data, model, loss_fn)


if __name__ == "__main__":
    main()
