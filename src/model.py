import torch
from torch import nn
from torchinfo import summary


def build_conv_block(in_channels: tuple[int], out_channels: tuple[int]):
    """ 
    Creates a convolution block with: \n
    `3dConv -> 3dBatchNorm -> ReLu -> 3dConv -> 3dBatchNorm -> ReLu` 
    """
    
    return nn.Sequential(
        nn.Conv3d(in_channels[0], out_channels[0], kernel_size=3, padding=1, padding_mode='zeros'),
        nn.BatchNorm3d(out_channels[0]),
        nn.ReLU(),
        nn.Conv3d(in_channels[1], out_channels[1], kernel_size=3, padding=1, padding_mode='zeros'),
        nn.BatchNorm3d(out_channels[1]),
        nn.ReLU(),
    )


class DownBlock(nn.Module):
    """ 
    A downsample block built with one convolution block 
    and a `3dMaxPool` layer.
    """

    def __init__(self, in_channels: tuple[int], out_channels: tuple[int]):
        super().__init__()
        self.conv = build_conv_block(
            in_channels=[in_channels[0], in_channels[1]], 
            out_channels=[out_channels[0], out_channels[1]]
        )
        self.down = nn.MaxPool3d(kernel_size=2)

    def forward(self, x):
        out = self.conv(x)
        downscaled = self.down(out)

        return out, downscaled


class UpBlock(nn.Module):
    """
    An upsample block built with one `3dConvTranspose` followed
    by one convolution block.
    """
    def __init__(self, in_channels: tuple[int], out_channels: tuple[int]):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels=in_channels[0], out_channels=out_channels[0], kernel_size=2, stride=2)
        # self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv = build_conv_block(in_channels=[in_channels[0], in_channels[1]], out_channels=[out_channels[1], out_channels[1]])

    def forward(self, x, skip):
        upscaled = self.up(x)
        # print(upscaled.shape, skip.shape)
        concat = torch.cat([skip, upscaled], dim=1)
        # print(concat.shape)
        out = self.conv(concat)
        # print(out.shape)

        # print()
        return out


class Unet(nn.Module):
    """
    Unet model with 3 downsampling blocks and 3 upsampling blocks, where
    one block consists of: \n
    `3dConv -> 3dBatchNorm -> ReLu -> 3dConv -> 3dBatchNorm -> ReLu`
    and a sampling operation (`3dMaxPool`/`3dConvTranspose`).
    """
    def __init__(self, in_channels, out_channels, blocks=3):
        super().__init__()
        self.downsampling_path = nn.ModuleList()
        self.upsampling_path = nn.ModuleList()

        if blocks == 3:
            channels = [in_channels, 32, 64, 128]
        elif blocks == 4:
            channels = [in_channels, 32, 64, 128, 256]

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
            nn.Conv3d(in_channels=channels[1], out_channels=out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # # Downsampling path
        # skip1, down1 = self.down1(x)
        # skip2, down2 = self.down2(down1)
        # skip3, down3 = self.down3(down2)
        # # skip4, down4 = self.down4(down3)

        # # Bottle neck
        # bottom = self.bottle_neck(down3)

        # # Expanding path
        # # up1 = self.up1(bottom, skip4)
        # up2 = self.up2(bottom, skip3)
        # up3 = self.up3(up2, skip2)
        # up4 = self.up4(up3, skip1)

        # # Output
        # out = self.output(up4)

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
    summary(Unet(in_channels=2, out_channels=1), input_size=(2, 2, 40, 80, 80))
    # print(Unet()._modules)
    # summary(Unet(block_num=4), input_size=(2, 2, 64, 128, 128))