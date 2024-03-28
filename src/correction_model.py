import torch
from torch import nn
from torchinfo import summary


def build_conv_block(in_channels: tuple[int], out_channels: tuple[int]):
    """ 
    Creates a convolution block with: \n
    `2dConv -> 2dBatchNorm -> ReLu -> 2dConv -> 2dBatchNorm -> ReLu` 
    """
    
    return nn.Sequential(
        nn.Conv2d(in_channels[0], out_channels[0], kernel_size=3, padding=1, padding_mode='zeros'),
        nn.BatchNorm2d(out_channels[0]),
        nn.ReLU(),
        nn.Conv2d(in_channels[1], out_channels[1], kernel_size=3, padding=1, padding_mode='zeros'),
        nn.BatchNorm2d(out_channels[1]),
        nn.ReLU(),
    )


class DownBlock(nn.Module):
    """ 
    A downsample block built with one convolution block 
    and a `2dMaxPool` layer.
    """

    def __init__(self, in_channels: tuple[int], out_channels: tuple[int]):
        super().__init__()
        self.conv = build_conv_block(
            in_channels=[in_channels[0], in_channels[1]], 
            out_channels=[out_channels[0], out_channels[1]]
        )
        self.down = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        out = self.conv(x)
        downscaled = self.down(out)

        return out, downscaled


class UpBlock(nn.Module):
    """
    An upsample block built with one `2dConvTranspose` followed
    by one convolution block.
    """
    def __init__(self, in_channels: tuple[int], out_channels: tuple[int]):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels=in_channels[0], out_channels=out_channels[0], kernel_size=2, stride=2)
        # self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv = build_conv_block(in_channels=[in_channels[0], in_channels[1]], out_channels=[out_channels[1], out_channels[1]])

    def forward(self, x, skip):
        upscaled = self.up(x)
        concat = torch.cat([skip, upscaled], dim=1)
        out = self.conv(concat)

        return out


class CorrectionUnet(nn.Module):
    """
    Unet model with 3 downsampling blocks and 3 upsampling blocks, where
    one block consists of: \n
    `2dConv -> 2dBatchNorm -> ReLu -> 2dConv -> 2dBatchNorm -> ReLu`
    and a sampling operation (`2dMaxPool`/`2dConvTranspose`).
    """
    def __init__(self, in_channels, out_channels, blocks=3):
        super().__init__()
        self.downsampling_path = nn.ModuleList()
        self.upsampling_path = nn.ModuleList()

        possible_channels = [in_channels, 16, 32, 64, 128]
        channels = possible_channels[:blocks+1]
        
        # Downsampling path
        for i in range(0, len(channels)-1):
            self.downsampling_path.append(
                DownBlock(in_channels=[channels[i], channels[i+1]], out_channels=[channels[i+1], channels[i+1]])
            )
        
        # Bottle neck
        self.bottle_neck = build_conv_block(in_channels=[channels[-1], channels[-1]*2], out_channels=[channels[-1]*2, channels[-1]*2])
        
        # Upsampling path
        for i in range(len(channels)-1, 0, -1):
            self.upsampling_path.append(
                UpBlock(in_channels=[channels[i]*2, channels[i]], out_channels=[channels[i], channels[i]])
            )

        # Output
        self.output = nn.Sequential(
            nn.Conv2d(in_channels=channels[1], out_channels=out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = x

        # Downsampling path
        skips = []
        for block in self.downsampling_path:
            skip, out = block(out)
            skips.append(skip)
        
        # Bottle neck
        out = self.bottle_neck(out)

        # Upsampling path
        skips.reverse()
        for (block, skip) in zip(self.upsampling_path, skips):
            out = block(out, skip)

        # Output
        out = self.output(out)
        return out


if __name__ == '__main__':
    summary(CorrectionUnet(in_channels=1, out_channels=1, blocks=3), input_size=(2, 1, 32, 32))
    # print(Unet()._modules)
    # summary(Unet(block_num=4), input_size=(2, 2, 64, 128, 128))