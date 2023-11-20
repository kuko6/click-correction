import argparse
import sys
import os
import glob
import wandb
import json

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchinfo import summary

from model import Unet
from data_generator import MRIDataset
from utils import get_glioma_indices, EarlyStopper

from losses.dice import dice_coefficient, DiceLoss, DiceBCELoss
from losses.focal_tversky import FocalTverskyLoss, FocalLoss, TverskyLoss

# print(sys.version_info)
# print(torch.__version__)

# print(torch.backends.cuda.is_built())
# print(torch.backends.cudnn.is_available())
# print(torch.cuda.is_available())

use_wandb = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} device')

config = {
    "lr": 1e-3,
    "num_classes": 1,
    "img_channels": 2,
    "dataset": "Schwannoma",
    "epochs": 50,
    "batch_size": 2,
    "loss": "focaltversky", 
    "optimizer": "Adam",
    "augment": False,
    "scheduler": True,
}

def prepare_data(data_dir: str) -> MRIDataset:
    """ Loads the data from `data_dir` and returns `Dataset` """

    t1_list = sorted(glob.glob(os.path.join(data_dir, 'VS-*-*/vs_*/*_t1_*')))
    t2_list = sorted(glob.glob(os.path.join(data_dir, 'VS-*-*/vs_*/*_t2_*')))
    seg_list = sorted(glob.glob(os.path.join(data_dir, 'VS-*-*/vs_*/*_seg_*')))

    t1_train, t1_val, t2_train, t2_val, seg_train, seg_val = train_test_split(t1_list, t2_list, seg_list, test_size=0.2, train_size=0.8, random_state=420)
    train_data = MRIDataset(t1_train, t2_train, seg_train, (40, 80, 80))
    val_data = MRIDataset(t1_val, t2_val, seg_val, (40, 80, 80))
  
    print(len(train_data), len(val_data))
    print(len(t1_train), len(t2_train), len(seg_train))
    print(len(t1_val), len(t2_val), len(seg_val))

    train_dataloader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=config['batch_size'], shuffle=False)

    return train_dataloader, val_dataloader


def preview(y_pred: torch.Tensor, y: torch.Tensor, dice: torch.Tensor, epoch=0):
    """ Saves a png of sample prediction `y_pred` for scan `y` """

    # Compute number of slices with the tumour
    first, last = get_glioma_indices(y)
    length = (last-first+1)
    n_graphs = (length*2)//6
    rows = n_graphs
    cols = 6
    res = cols if cols > rows else rows

    # Plot them
    fig, axs = plt.subplots(rows, cols, figsize=(res*2, res*2))
    axs = axs.flatten()
    j = 0
    for i in range(first, last):
        if j >= len(axs): break
        axs[j].imshow(y[0,i,:,:].cpu().detach(), cmap='magma')
        axs[j].axis('off')
        axs[j].set_title(f'mask slice {i}', fontsize=9)
        axs[j+1].imshow(y_pred[0,i,:,:].cpu().detach(), cmap='magma')
        axs[j+1].axis('off')
        axs[j+1].set_title(f'pred slice {i}', fontsize=9)
        j += 2
    fig.suptitle(f'Dice: {dice.item()}', fontsize=10)
    plt.subplots_adjust(top=0.9)

    fig.savefig(f'outputs/{epoch}_preview.png')
    plt.close(fig)


def val(dataloader: DataLoader, model: Unet, loss_fn: torch.nn.Module, epoch: int) -> tuple[float, float]:
    """ Validate model after each epoch on validation dataset, returns the avg. loss and avg. dice """
    
    model.eval()
    avg_loss, avg_dice = 0, 0
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            y_pred = model(x)

            # avg_loss += loss_fn(y_pred, y).item()
            # avg_dice += dice_coefficient(y_pred, y).item()

            # Compute loss
            loss = loss_fn(y_pred, y)
            avg_loss += loss.item()

            # Compute the dice coefficient
            dice = dice_coefficient(y_pred, y).item()
            avg_dice += dice

            print(f'validation step: {i+1}/{len(dataloader)}, loss: {loss.item():>5f}, dice: {dice}', end='\r')

            if i==0:
              preview(y_pred[0], y[0], dice_coefficient(y_pred, y), epoch)

    avg_loss /= len(dataloader)
    avg_dice /= len(dataloader)
    print()

    return (avg_loss, avg_dice)


def train_one_epoch(dataloader: DataLoader, model: Unet, loss_fn, optimizer) -> tuple[float, float]:
  """ Train model for one epoch on the training dataset, returns the avg. loss and avg. dice """

  model.train()
  avg_loss, avg_dice = 0, 0

  for i, (x, y) in enumerate(dataloader):
    x, y = x.to(device), y.to(device)

    # print(x.shape, y.shape)
    # print(x.dtype, y.dtype)

    optimizer.zero_grad()

    # Get prediction
    y_pred = model(x)

    # Compute loss
    loss = loss_fn(y_pred, y)
    # print(loss)
    avg_loss += loss.item()

    # Compute the dice coefficient
    dice = dice_coefficient(y_pred, y).item()
    avg_dice += dice

    # Update parameters
    loss.backward()
    optimizer.step()

    print(f'training step: {i+1}/{len(dataloader)}, loss: {loss.item():>5f}, dice: {dice}', end='\r')

  avg_loss /= len(dataloader)
  avg_dice /= len(dataloader)
  print()

  return (avg_loss, avg_dice)


def train(train_dataloader: DataLoader, val_dataloader: DataLoader, model: Unet, loss_fn, optimizer, scheduler):
    """ Run the training """
    
    epochs = config['epochs']
    train_history = {'loss': [], 'dice': []}
    val_history = {'loss': [], 'dice': []}

    early_stopper = EarlyStopper(patience=5, delta=0, mode='min')

    for epoch in range(epochs):
        print('-------------------------------')
        print(f'epoch: {epoch}')

        train_loss, train_dice = train_one_epoch(train_dataloader, model, loss_fn, optimizer)
        val_loss, val_dice = val(val_dataloader, model, loss_fn, epoch)

        train_history['loss'].append(train_loss)
        train_history['dice'].append(train_dice)

        val_history['loss'].append(val_loss)
        val_history['dice'].append(val_dice)

        if config['scheduler']:
            scheduler.step(val_loss)

        print(f'loss: {train_loss:>5f} dice: {train_dice:>5f}')
        print(f'val loss: {val_loss:>5f} val dice: {val_dice:>5f}')

        with open('outputs/train_history.json', 'w') as f:
            json.dump(train_history, f)
        with open('outputs/val_history.json', 'w') as f:
            json.dump(val_history, f)

        torch.save({
            'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()
        }, 'outputs/checkpoint.pt')

        if use_wandb:
            wandb.log({
                'epoch': epoch, 'loss': train_loss, 'dice': train_dice, 
                'val_loss':val_loss, 'val_dice': val_dice, 'lr': optimizer.param_groups[0]["lr"]
            })
            wandb.log({'preview': wandb.Image(f'outputs/{epoch}_preview.png')})

        # artifact = wandb.Artifact('checkpoint', type='model', metadata={'val_dice': val_dice})
        # artifact.add_file('../outputs/checkpoint.pt')
        # wandb.run.log_artifact(artifact)
        
        if early_stopper(val_loss):
            print('Stopping early!!!')
            break
        
    
    if use_wandb:
        wandb.finish()


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, help='path to data')
    parser.add_argument('--wandb', type=str, help='wandb id')
    
    args = parser.parse_args()
    # print(args.wandb)

    if not args.data_path:
        print('You need to specify datapath!!!! >:(')

    wandb_key = args.wandb
    if use_wandb:
        wandb.login(key=wandb_key)
        wandb.init(project='DP', entity='kuko', reinit=True, config=config)    

    data_dir = args.data_path
    print(os.listdir(data_dir))
    
    if not os.path.isdir('outputs'):
        os.mkdir('outputs')

    # Prepare Datasets
    train_dataloader, val_dataloader = prepare_data(data_dir)

    # Initialize model and optimizer
    model = Unet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.1)

    # Select loss function
    classes = {0: 64115061, 1: 396939}
    total_pixels = classes[0] + classes[1]
    # weight = torch.tensor(total_pixels/classes[0]).to(device)
    # weight = torch.tensor(classes[0]/classes[1]).to(device)
    # loss_fn = nn.BCELoss(weight=weight)
    # loss_fn = DiceLoss()
    # loss_fn = DiceBCELoss(weight=weight)
    # loss_fn = FocalLoss(alpha=weight, gamma=2)
    # loss_fn = TverskyLoss(alpha=.3, beta=.7)
    loss_fn = FocalTverskyLoss(alpha=.3, beta=.7)

    # Train :D
    train(train_dataloader, val_dataloader, model, loss_fn, optimizer, scheduler)


if __name__ == '__main__': 
    main()
    