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

import scipy
from losses.dice import dice_coefficient
from utils import get_glioma_indices

use_wandb = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[Using {device} device]')

config = {
    'lr': 1e-3,
    'img_channels': 2,
    'num_classes': 1,
    'conv_blocks': 3, # 3 if device == 'cpu' else 4
    'dataset': 'Schwannoma',
    'epochs': 30,
    'batch_size': 2,
    'loss': 'distance', 
    'optimizer': 'Adam',
    'augment': False,
    'scheduler': True,
    'early_stopper': True,
    'img_dims': (40, 128, 128), # (64, 80, 80) if device == 'cpu' else (64, 128, 128)
    'training': 'clicks', # base, clicks-pretraining, clicks
    'clicks': {
        'use': True,
        'gen_fg': False,
        'gen_bg': False,
        'gen_border': True,
        'num': 5,
        'size': 1
    }
}


# def get_dst_map(y_pred, thresh_val: None | int, thresh_mode='max', probs=True, preds_threshold=0.7):
#     combined = torch.zeros_like(y_pred)
        
#     if probs:
#         # threshold the probabilities
#         y_threshed = (y_pred > preds_threshold).type(torch.float32).detach().cpu()

#     for seg_idx in range(len(y_pred)):
#         first, last = get_glioma_indices(y_threshed[seg_idx])
        
#         for slice_idx in range(first, last+1):
#             dst = torch.as_tensor(
#                 scipy.ndimage.distance_transform_edt(y_threshed[seg_idx,0,slice_idx,:,:]), 
#                 dtype=torch.float32, device=y_pred.device
#             )

#             inverted_dst = torch.as_tensor(
#                 scipy.ndimage.distance_transform_edt(1 - y_threshed[seg_idx,0,slice_idx,:,:]), 
#                 dtype=torch.float32, device=y_pred.device
#             )

#             combined[seg_idx,:,slice_idx,:,:] = dst + inverted_dst

#         # thresholding
#         # thresh_val = 5.
#         if thresh_mode == 'max' or thresh_val == None:
#             thresh_val = torch.max(dst).item()
#         combined[seg_idx][combined[seg_idx] > thresh_val] = thresh_val

#     combined.requires_grad_()
#     return combined


def prepare_data(data_dir: str) -> MRIDataset:
    """ Loads the data from `data_dir` and returns `Dataset` """

    t1_list = sorted(glob.glob(os.path.join(data_dir, 'VS-*-*/vs_*/*_t1_*')))
    t2_list = sorted(glob.glob(os.path.join(data_dir, 'VS-*-*/vs_*/*_t2_*')))
    seg_list = sorted(glob.glob(os.path.join(data_dir, 'VS-*-*/vs_*/*_seg_*')))

    t1_train, t1_val, t2_train, t2_val, seg_train, seg_val = train_test_split(t1_list, t2_list, seg_list, test_size=0.2, train_size=0.8, random_state=420)
    t1_val.append(t1_train.pop(-1))
    t2_val.append(t2_train.pop(-1))
    seg_val.append(seg_train.pop(-1))

    # if config['clicks']['use']:
    #     preview_clicks(t1_list, t2_list, seg_list, config['clicks'])
    
    if config['training'] == 'base':
        train_data = MRIDataset(t1_train, t2_train, seg_train, config['img_dims'], clicks=config['clicks'])
        val_data = MRIDataset(t1_val, t2_val, seg_val, config['img_dims'], clicks=config['clicks'])
    elif config['training'] == 'clicks':
        train_data = MRIDataset(t1_train[40:], t2_train[40:], seg_train[40:], config['img_dims'], clicks=config['clicks'])
        val_data = MRIDataset(t1_val[10:], t2_val[10:], seg_val[10:], config['img_dims'], clicks=config['clicks'])
        # train_data = MRIDataset(t1_train, t2_train, seg_train, config['img_dims'], clicks=config['clicks'])
        # val_data = MRIDataset(t1_val, t2_val, seg_val, config['img_dims'])
    elif config['training'] == 'clicks-pretraining':
        train_data = MRIDataset(t1_train[:40], t2_train[:40], seg_train[:40], config['img_dims'], clicks=False)
        val_data = MRIDataset(t1_val[:10], t2_val[:10], seg_val[:10], config['img_dims'])
    
    print(len(train_data), len(val_data))
    # print(len(t1_train), len(t2_train), len(seg_train))
    # print(len(t1_val), len(t2_val), len(seg_val))

    train_dataloader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=config['batch_size'], shuffle=False)

    return train_dataloader, val_dataloader


def val(dataloader: DataLoader, model: Unet, loss_fn: torch.nn.Module, epoch: int) -> tuple[float, float]:
    """ Validate model after each epoch on validation dataset, returns the avg. loss and avg. dice """
    
    model.eval()
    avg_loss, avg_dice = 0, 0
    with torch.no_grad():
        for i, (x, y, z) in enumerate(dataloader):
            x, y, z = x.to(device), y.to(device), z.to(device)
            y_pred = model(x)

            # Compute loss
            # combined = get_dst_map(y_pred, thresh_val=10.0, probs=True, preds_threshold=0.7)
            # loss = loss_fn(y_pred, y, combined)
            loss = loss_fn(y_pred, y)
            avg_loss += loss.item()

            # Compute the dice coefficient
            dice = dice_coefficient(y_pred, z).item()
            avg_dice += dice

            print(f'validation step: {i+1}/{len(dataloader)}, loss: {loss.item():>5f}, dice: {dice:>5f}', end='\r')

            if i==0:
              preview(y_pred[0], y[0], z[0], dice_coefficient(y_pred, z), epoch)

    avg_loss /= len(dataloader)
    avg_dice /= len(dataloader)
    print()

    return (avg_loss, avg_dice)


def train_one_epoch(dataloader: DataLoader, model: Unet, loss_fn, optimizer, epoch) -> tuple[float, float]:
  """ Train model for one epoch on the training dataset, returns the avg. loss and avg. dice """

  model.train()
  avg_loss, avg_dice = 0, 0

  for i, (x, y, z) in enumerate(dataloader):
    x, y, z = x.to(device), y.to(device), z.to(device)
    optimizer.zero_grad()

    # Get prediction
    y_pred = model(x)

    # Compute loss
    # combined = get_dst_map(y_pred, thresh_val=10.0, probs=True, preds_threshold=0.7)
    # loss = loss_fn(y_pred, y, combined)
    loss = loss_fn(y_pred, y)
    avg_loss += loss.item()

    # Compute the dice coefficient
    dice = dice_coefficient(y_pred, z).item()
    avg_dice += dice

    # Update parameters
    loss.backward()
    optimizer.step()

    print(f'training step: {i+1}/{len(dataloader)}, loss: {loss.item():>5f}, dice: {dice:>5f}', end='\r')

  avg_loss /= len(dataloader)
  avg_dice /= len(dataloader)
  print()

  return (avg_loss, avg_dice)


def train(train_dataloader: DataLoader, val_dataloader: DataLoader, model: Unet, loss_fn, optimizer, scheduler):
    """ Run the training """
    
    epochs = config['epochs']
    train_history = {'loss': [], 'dice': []}
    val_history = {'loss': [], 'dice': []}
    best = {'loss': np.inf, 'dice': 0, 'epoch': 0}

    if config['early_stopper']:
        early_stopper = EarlyStopper(patience=6, delta=0.01, mode='max')

    for epoch in range(epochs):
        print('===============================')
        print(f'[Epoch: {epoch}]')

        # Train and validate
        train_loss, train_dice = train_one_epoch(train_dataloader, model, loss_fn, optimizer, epoch)
        print('-------------------------------')
        val_loss, val_dice = val(val_dataloader, model, loss_fn, epoch)

        print('-------------------------------')
        print(f'loss: {train_loss:>5f} dice: {train_dice:>5f}')
        print(f'val loss: {val_loss:>5f} val dice: {val_dice:>5f}')
        
        # Log training and validation history
        train_history['loss'].append(train_loss)
        train_history['dice'].append(train_dice)

        val_history['loss'].append(val_loss)
        val_history['dice'].append(val_dice)

        with open('outputs/train_history.json', 'w') as f:
            json.dump(train_history, f)
        with open('outputs/val_history.json', 'w') as f:
            json.dump(val_history, f)
        
        # Save checkpoint
        model_checkpoint = {
            'epoch': epoch, 
            'model_state': model.state_dict(), 
            'optimizer_state': optimizer.state_dict()
        }

        torch.save(model_checkpoint, 'outputs/checkpoint.pt')
        
        # Save best checkpoint
        if best['dice'] < val_dice:
            print('-------------------------------')
            print(f'new best!!! (loss: {best["loss"]:>5f} -> {val_loss:>5f}, dice: {best["dice"]:>5f} -> {val_dice:>5f})')
            
            torch.save(model_checkpoint, 'outputs/best.pt')    
            
            best['dice'] = val_dice
            best['loss'] = val_loss
            best['epoch'] = epoch
            # best['model'] = model_checkpoint

        # Log to wandb
        if use_wandb:
            wandb.log({
                'epoch': epoch, 'loss': train_loss, 'dice': train_dice, 
                'val_loss':val_loss, 'val_dice': val_dice, 'lr': optimizer.param_groups[0]["lr"]
            })
            wandb.log({'preview': wandb.Image(f'outputs/images/{epoch}_preview.png')})
        
        # Run scheduler and early stopper
        if config['scheduler']:
            scheduler.step(val_loss)

        if config['early_stopper']:
            if early_stopper(val_dice):
                print('===============================')
                print('Stopping early!!!')
                break
    
    print('===============================')
    print(f'The best model was in epoch {best["epoch"]} with loss: {best["loss"]:>5f} and dice: {best["dice"]:>5f}')

    if use_wandb:
        artifact = wandb.Artifact('best_model', type='model', metadata={'val_dice': val_dice})
        artifact.add_file('outputs/best.pt')
        wandb.run.log_artifact(artifact)
        wandb.finish()


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

    wandb_key = args.wandb
    if use_wandb and wandb_key:
        wandb.login(key=wandb_key)
        wandb.init(project='DP', entity='kuko', reinit=True, config=config)

    data_dir = args.data_path
    print(os.listdir(data_dir))
    
    if not os.path.isdir('outputs'):
        os.mkdir('outputs')
    if not os.path.isdir('outputs/images'):
        os.mkdir('outputs/images')

    # Prepare Datasets
    train_dataloader, val_dataloader = prepare_data(data_dir)

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

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.1)
    
    # Load pretrained model
    if args.model_path and config['training'] == 'clicks':
        print('Using pretrained model')
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])

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

    # Train :D
    train(train_dataloader, val_dataloader, model, loss_fn, optimizer, scheduler)


if __name__ == '__main__': 
    main()