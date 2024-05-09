import torch
from torch import nn
from torchinfo import summary


def build_conv_block(in_channels: tuple[int], out_channels: tuple[int]) -> nn.Sequential:
    """
    Creates a 2d convolution block with:

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


def build_3dconv_block(in_channels: tuple[int], out_channels: tuple[int]) -> nn.Sequential:
    """
    Creates a 3d convolution block with:

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
    and a `2dMaxPool` layer.
    """

    def __init__(self, in_channels: tuple[int], out_channels: tuple[int], volumetric=False):
        super().__init__()
        if volumetric:
            self.conv = build_3dconv_block(
                in_channels=[in_channels[0], in_channels[1]],
                out_channels=[out_channels[0], out_channels[1]],
            )
            self.down = nn.MaxPool3d(kernel_size=2)
        else:
            self.conv = build_conv_block(
                in_channels=[in_channels[0], in_channels[1]],
                out_channels=[out_channels[0], out_channels[1]],
            )
            self.down = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        out = self.conv(x)
        downscaled = self.down(out)

        return out, downscaled


class UpBlock(nn.Module):
    """
    An upsample block built with one transposed convolution followed
    by one convolution block.
    """

    def __init__(self, in_channels: tuple[int], out_channels: tuple[int], volumetric=False):
        super().__init__()
        if volumetric:
            self.up = nn.ConvTranspose3d(in_channels=in_channels[0], out_channels=out_channels[0], kernel_size=2, stride=2)
            # self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = build_3dconv_block(
                in_channels=[in_channels[0], in_channels[1]],
                out_channels=[out_channels[1], out_channels[1]],
            )
        else:    
            self.up = nn.ConvTranspose2d(in_channels=in_channels[0], out_channels=out_channels[0], kernel_size=2, stride=2)
            # self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = build_conv_block(
                in_channels=[in_channels[0], in_channels[1]],
                out_channels=[out_channels[1], out_channels[1]],
            )

    def forward(self, x, skip):
        upscaled = self.up(x)
        concat = torch.cat([skip, upscaled], dim=1)
        out = self.conv(concat)

        return out


class BottleNeck(nn.Module):
    def __init__(self, in_channels: tuple[int], out_channels: tuple[int], volumetric=False):
        super().__init__()
        if volumetric:
            self.conv = build_3dconv_block(
                in_channels=[in_channels[0], in_channels[1]],
                out_channels=[out_channels[0], out_channels[1]],
            )
        else:
            self.conv = build_conv_block(
                in_channels=[in_channels[0], in_channels[1]],
                out_channels=[out_channels[0], out_channels[1]],
            )

    def forward(self, x):
        out = self.conv(x)

        return out

class CorrectionUnet(nn.Module):
    """
    Unet model with 3 downsampling blocks and 3 upsampling blocks, where
    one block consists of:
    
    `2dConv -> 2dBatchNorm -> ReLu -> 2dConv -> 2dBatchNorm -> ReLu`
    and a sampling operation (`2dMaxPool`/`2dConvTranspose`).
    """

    def __init__(self, in_channels, out_channels, blocks=3, block_channels=[16, 32, 64, 128], volumetric=False):
        super().__init__()
        self.downsampling_path = nn.ModuleList()
        self.upsampling_path = nn.ModuleList()

        # possible_channels = [in_channels, 24, 48, 96, 192]
        possible_channels = [in_channels] + block_channels
        channels = possible_channels[: blocks + 1]

        # Downsampling path
        for i in range(0, len(channels) - 1):
            self.downsampling_path.append(
                DownBlock(
                    in_channels=[channels[i], channels[i + 1]],
                    out_channels=[channels[i + 1], channels[i + 1]],
                    volumetric=volumetric
                )
            )

        # Bottle neck
        self.bottle_neck = BottleNeck(
            in_channels=[channels[-1], channels[-1] * 2],
            out_channels=[channels[-1] * 2, channels[-1] * 2],
            volumetric=volumetric
        )

        # Upsampling path
        for i in range(len(channels) - 1, 0, -1):
            self.upsampling_path.append(
                UpBlock(
                    in_channels=[channels[i] * 2, channels[i]],
                    out_channels=[channels[i], channels[i]],
                    volumetric=volumetric
                )
            )

        # Output
        if volumetric:
            self.output = nn.Sequential(
                nn.Conv3d(in_channels=channels[1], out_channels=out_channels, kernel_size=1),
                nn.Sigmoid(),
            )
        else:
            self.output = nn.Sequential(
                nn.Conv2d(in_channels=channels[1], out_channels=out_channels, kernel_size=1),
                nn.Sigmoid(),
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
        for block, skip in zip(self.upsampling_path, skips):
            out = block(out, skip)

        # Output
        out = self.output(out)
        return out


class MultiModalCorrectionUnet(nn.Module):
    """
    Unet with multiple encoders...
    """

    def __init__(self, in_channels, out_channels, blocks=3, encoders=2, block_channels=[16, 32, 64, 128], volumetric=False):
        super().__init__()
        self.downsampling_paths = nn.ModuleList()
        self.bottle_necks = nn.ModuleList()
        self.upsampling_path = nn.ModuleList()
        self.encoders = encoders

        # possible_channels = [in_channels, 24, 48, 96, 192]

        # Downsampling path
        for enc_i in range(encoders):
            possible_channels = [in_channels[enc_i]] + block_channels
            channels = possible_channels[: blocks + 1]
            # print(channels)

            downsampling_path = nn.ModuleList()
            for i in range(0, blocks):
                downsampling_path.append(
                    DownBlock(
                        in_channels=[channels[i], channels[i + 1]],
                        out_channels=[channels[i + 1], channels[i + 1]],
                        volumetric=volumetric
                    )
                )
            self.downsampling_paths.append(downsampling_path)
            
        channels = [0, block_channels[1], block_channels[2], block_channels[3]]
        channels = channels[: blocks + 1]
        
        # Bottle neck
        self.bottle_neck = BottleNeck(
            in_channels=[channels[-1], channels[-1] * 2],
            out_channels=[channels[-1] * 2, channels[-1] * 2],
            volumetric=volumetric
        )
        
        # Upsampling path
        for i in range(len(channels) - 1, 0, -1):
            self.upsampling_path.append(
                UpBlock(
                    in_channels=[channels[i], channels[i]],
                    out_channels=[channels[i], channels[i]],
                    volumetric=volumetric
                )
            )

        # Output
        if volumetric:
            self.output = nn.Sequential(
                nn.Conv3d(in_channels=channels[0], out_channels=out_channels, kernel_size=1),
                nn.Sigmoid(),
            )
        else:
            self.output = nn.Sequential(
                nn.Conv2d(in_channels=channels[0], out_channels=out_channels, kernel_size=1),
                nn.Sigmoid(),
            )

    def forward(self, x):
        # out = x
        if self.encoders == 2:
            inputs = [x[:,0].unsqueeze(1), x[:,1:]]
        else:
            inputs = [x[:,i].unsqueeze(1) for i in range(x.shape[1])]
        # print(len(inputs), inputs[0].shape, inputs[1].shape)

        # Downsampling path
        skips = []
        # print(self.downsampling_paths)
        # print(self.encoders, len(self.bottle_necks), len(self.downsampling_paths))
        for i in range(self.encoders):
            path_skips = []
            for down_block in self.downsampling_paths[i]:  
                skip, inputs[i] = down_block(inputs[i]) 
                path_skips.append(skip)
            skips.append(path_skips)

        out = torch.cat(inputs, dim=1)
        print(out.shape)
        
        # Bottle neck
        # for i in range(self.encoders):
        #     inputs[i] = self.bottle_necks[i](inputs[i])
        #     print(inputs[i].shape)

        out = self.bottle_neck(out)
        print(inputs[0].shape)
        
        # Upsampling path
        # print(len(skips), len(skips[0]))
        # skips.reverse()
        new_skips = []
        for path_skips in zip(*skips):
            print(path_skips[0].shape, path_skips[1].shape, torch.cat(path_skips, dim=1).shape)
            new_skips.append(torch.cat(path_skips, dim=1))

        # print(new_skips[0].shape, new_skips[1].shape, new_skips[2].shape)

        new_skips.reverse()
        for up_block, skip in zip(self.upsampling_path, new_skips):
            print(out.shape, skip.shape)
            out = up_block(out, skip)

        # print(self.upsampling_path)
        
        # Output
        # out = self.output(out)
        # return out

class OGCorrectionUnet(nn.Module):
    """
    Unet model with 3 downsampling blocks and 3 upsampling blocks, where
    one block consists of:
    
    `2dConv -> 2dBatchNorm -> ReLu -> 2dConv -> 2dBatchNorm -> ReLu`
    and a sampling operation (`2dMaxPool`/`2dConvTranspose`).
    """

    def __init__(self, in_channels, out_channels, blocks=3):
        super().__init__()
        self.downsampling_path = nn.ModuleList()
        self.upsampling_path = nn.ModuleList()

        possible_channels = [in_channels, 16, 32, 64, 128]
        channels = possible_channels[: blocks + 1]

        # Downsampling path
        for i in range(0, len(channels) - 1):
            self.downsampling_path.append(
                DownBlock(
                    in_channels=[channels[i], channels[i + 1]],
                    out_channels=[channels[i + 1], channels[i + 1]],
                )
            )

        # Bottle neck
        self.bottle_neck = build_conv_block(
            in_channels=[channels[-1], channels[-1] * 2],
            out_channels=[channels[-1] * 2, channels[-1] * 2],
        )

        # Upsampling path
        for i in range(len(channels) - 1, 0, -1):
            self.upsampling_path.append(
                UpBlock(
                    in_channels=[channels[i] * 2, channels[i]],
                    out_channels=[channels[i], channels[i]],
                )
            )

        # Output
        self.output = nn.Sequential(
            nn.Conv2d(in_channels=channels[1], out_channels=out_channels, kernel_size=1),
            nn.Sigmoid(),
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
        for block, skip in zip(self.upsampling_path, skips):
            out = block(out, skip)

        # Output
        out = self.output(out)
        return out


if __name__ == "__main__":
    model = MultiModalCorrectionUnet(in_channels=[1, 2], encoders=2, out_channels=1, blocks=3)
    x = torch.rand(size=(2, 3, 32, 32))
    model(x)

    # summary(
    #     CorrectionUnet(in_channels=1, out_channels=1, blocks=3, volumetric=True),
    #     input_size=(2, 1, 16, 32, 32),
    # )

    # summary(
    #     CorrectionUnet(
    #         in_channels=3,
    #         out_channels=1,
    #         blocks=4,
    #         block_channels=[12, 24, 48, 96],
    #         volumetric=False,
    #     ),
    #     input_size=(2, 3, 48, 48),
    # )

    # summary(
    #     OGCorrectionUnet(in_channels=1, out_channels=1, blocks=3),
    #     input_size=(2, 1, 32, 32),
    # )
    # print(Unet()._modules)
    # summary(Unet(block_num=4), input_size=(2, 2, 64, 128, 128))
