import argparse
import os
import glob
import sys

import torch

print(sys.version_info)
print(torch.__version__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} device')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, help='path to data')
    parser.add_argument('--wandb', type=str, help='wandb id')
    
    args = parser.parse_args()

    if not args.data_path:
        print('You need to specify datapath!!!! >:(')

    data = args.data_path
    print(os.listdir(data))
    print(args.wandb)


if __name__ == '__main__':
    main()