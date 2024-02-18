import argparse
import sys
import os
import glob
import wandb
import json

from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchinfo import summary

from model import Unet
from data_generator import MRIDataset
from utils import EarlyStopper, preview, preview_clicks

from losses.dice import dice_coefficient, DiceLoss, DiceBCELoss
from losses.focal_tversky import FocalTverskyLoss, FocalLoss, TverskyLoss
from losses.clicks import DistanceLoss


use_wandb = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[Using {device} device]')

config = {
    'img_channels': 2,
    'num_classes': 1,
    'conv_blocks': 3, # 3 if device == 'cpu' else 4
    'dataset': 'Schwannoma',
    'batch_size': 2,
    'loss': 'distance', 
    'img_dims': (40, 128, 128), # (64, 80, 80) if device == 'cpu' else (64, 128, 128)
    'clicks': {
        'use': True,
        'gen_fg': False,
        'gen_bg': False,
        'gen_border': True,
        'num': 40,
        'size': 1
    }
}
print(f"Training with clicks: {config['clicks']['use']}")

def prepare_data(data_dir: str) -> MRIDataset:
    """ Loads the data from `data_dir` and returns `Dataset` """

    t1_list = sorted(glob.glob(os.path.join(data_dir, 'VS-*-*/vs_*/*_t1_*')))
    t2_list = sorted(glob.glob(os.path.join(data_dir, 'VS-*-*/vs_*/*_t2_*')))
    seg_list = sorted(glob.glob(os.path.join(data_dir, 'VS-*-*/vs_*/*_seg_*')))

    # if config['clicks']['use']:
    #     preview_clicks(t1_list, t2_list, seg_list, config['clicks'])
    
    val_data = MRIDataset(t1_list, t2_list, seg_list, config['img_dims'], clicks=config['clicks'])
    
    print(len(val_data))
    val_dataloader = DataLoader(val_data, batch_size=config['batch_size'], shuffle=False)

    return val_dataloader


def evaluate(dataloader: DataLoader, model: Unet, loss_fn: torch.nn.Module) -> tuple[float, float]:
    """ Validate model after each epoch on validation dataset, returns the avg. loss and avg. dice """
    
    model.eval()
    avg_loss, avg_dice = 0, 0
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            y_pred = model(x)

            # Compute loss
            loss = loss_fn(y_pred, y)
            avg_loss += loss.item()

            # Compute the dice coefficient
            dice = dice_coefficient(y_pred, y).item()
            avg_dice += dice

            print(f'validation step: {i+1}/{len(dataloader)}, loss: {loss.item():>5f}, dice: {dice:>5f}', end='\r')

            # if i==0:
            #   preview(y_pred[0], y[0], dice_coefficient(y_pred, y))

    avg_loss /= len(dataloader)
    avg_dice /= len(dataloader)
    print()

    return (avg_loss, avg_dice)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, help='path to data')
    parser.add_argument('--model-path', type=str, help='path to pretrained model')
    parser.add_argument('--wandb', type=str, help='wandb id')
    
    args = parser.parse_args()
    # print(args.wandb)

    if not args.data_path:
        print('You need to specify datapath!!!! >:(')
        return

    # wandb_key = args.wandb
    # if use_wandb and wandb_key:
    #     wandb.login(key=wandb_key)
    #     wandb.init(project='DP', entity='kuko', reinit=True, config=config)    

    data_dir = args.data_path
    print(os.listdir(data_dir))
    
    if not os.path.isdir('outputs'):
        os.mkdir('outputs')
    if not os.path.isdir('outputs/images'):
        os.mkdir('outputs/images')

    # Prepare Datasets
    val_dataloader = prepare_data(data_dir)

    # Initialize model
    model = Unet(
        in_channels=config['img_channels'], 
        out_channels=config['num_classes'], 
        blocks=config['conv_blocks']
    ).to(device)

    # writes model architecture to a file (just for experiment logging)
    with open('outputs/architecture.txt', 'w') as f:
        model_summary = summary(
            Unet(in_channels=config['img_channels'], 
                out_channels=config['num_classes'], 
                blocks=config['conv_blocks'],
            ), 
            input_size=(config['batch_size'], config['img_channels'], *config['img_dims']), 
            verbose=0
        )
        f.write(str(model_summary))
    
    # Load pretrained model
    # print('Using pretrained model')
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])

    # Select loss function
    classes = {0: 64115061, 1: 396939}
    # total_pixels = classes[0] + classes[1]
    # weight = torch.tensor(total_pixels/classes[0]).to(device)
    weight = torch.tensor(classes[1]/classes[0]).to(device)

    loss_functions = {
        'bce': torch.nn.BCELoss(weight=weight),
        'dice': DiceLoss(),
        'dicebce': DiceBCELoss(weight=weight),
        'focal': FocalLoss(alpha=weight, gamma=2),
        'tversky': TverskyLoss(alpha=.3, beta=.7),
        'focaltversky': FocalTverskyLoss(alpha=.3, beta=.7, gamma=.75),
        'distance': DistanceLoss(thresh_val=10.0, probs=True, preds_threshold=0.7)
    }
    loss_fn = loss_functions[config['loss']]

    # Evaluate :D
    avg_loss, avg_dice = evaluate(val_dataloader, model, loss_fn)
    print(f'Avg loss: {avg_loss:>5f} avg dice: {avg_dice:>5f}')


if __name__ == '__main__': 
    main()