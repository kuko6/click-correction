import torch
from torch import nn
from torchinfo import summary


def build_conv_block(in_channels: tuple[int], out_channels: tuple[int]) -> nn.Sequential:
    """
    Creates a 2d convolution block with:

    `2dConv -> 2dBatchNorm -> ReLu -> 2dConv -> 2dBatchNorm -> ReLu`
    """

    layers = nn.Sequential(
        nn.Conv2d(in_channels[0], out_channels[0], kernel_size=3, padding=1, padding_mode='zeros'),
        nn.BatchNorm2d(out_channels[0]),
        nn.ReLU(),
        nn.Conv2d(in_channels[1], out_channels[1], kernel_size=3, padding=1, padding_mode='zeros'),
        nn.BatchNorm2d(out_channels[1]),
        nn.ReLU(),
    )
    
    return layers


def build_3dconv_block(in_channels: tuple[int], out_channels: tuple[int]) -> nn.Sequential:
    """
    Creates a 3d convolution block with:

    `3dConv -> 3dBatchNorm -> ReLu -> 3dConv -> 3dBatchNorm -> ReLu`
    """

    layers =  nn.Sequential(
        nn.Conv3d(in_channels[0], out_channels[0], kernel_size=3, padding=1, padding_mode='zeros'),
        nn.BatchNorm3d(out_channels[0]),
        nn.ReLU(),
        nn.Conv3d(in_channels[1], out_channels[1], kernel_size=3, padding=1, padding_mode='zeros'),
        nn.BatchNorm3d(out_channels[1]),
        nn.ReLU(),
    )

    return layers


class DownBlock(nn.Module):
    """
    A downsample block built with one convolution block
    and a `2dMaxPool` layer.
    """

    def __init__(self, in_channels: tuple[int], out_channels: tuple[int], volumetric=False, use_dropout=False):
        super().__init__()
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.4)

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
        if self.use_dropout:
            out = self.dropout(out)

        return out, downscaled


class UpBlock(nn.Module):
    """
    An upsample block built with one transposed convolution followed
    by one convolution block.
    """

    def __init__(self, in_channels: tuple[int], out_channels: tuple[int], volumetric=False, multi_enc=False, use_dropout=False):
        super().__init__()
        self.multi_enc = multi_enc

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.4)

        if volumetric:
            self.up = nn.ConvTranspose3d(in_channels=in_channels[0], out_channels=out_channels[0], kernel_size=2, stride=2)
            self.conv = build_3dconv_block(
                in_channels=[in_channels[0], in_channels[1]],
                out_channels=[out_channels[1], out_channels[1]],
            )
            if multi_enc:
                self.resampler = nn.Conv3d(in_channels[0], out_channels[0], kernel_size=1, stride=1, padding=0)
        else:    
            self.up = nn.ConvTranspose2d(in_channels=in_channels[0], out_channels=out_channels[0], kernel_size=2, stride=2)
            self.conv = build_conv_block(
                in_channels=[in_channels[0], in_channels[1]],
                out_channels=[out_channels[1], out_channels[1]],
            )
            if multi_enc:
                self.resampler = nn.Conv2d(in_channels[0], out_channels[0], kernel_size=1, stride=1, padding=0)

    def forward(self, x, skip):
        upscaled = self.up(x)

        if self.multi_enc:
            skip = self.resampler(skip)

        concat = torch.cat([skip, upscaled], dim=1)
        out = self.conv(concat)

        if self.use_dropout:
            out = self.dropout(out)

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
    

class Output(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, volumetric=False):
        super().__init__()
        if volumetric:
            self.conv = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1),
                nn.Sigmoid(),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.Sigmoid(),
            )

    def forward(self, x):
        out = self.conv(x)

        return out


class AttentionBlock(nn.Module):
    def __init__(self, fg, fx, volumetric=False):
        super(AttentionBlock, self).__init__()

        if volumetric:
            self.wg = nn.Conv3d(in_channels=fg, out_channels=fg, kernel_size=1, stride=1, padding=0)
            self.wx = nn.Conv3d(in_channels=fx, out_channels=fg, kernel_size=1, stride=2, padding=0)    
            self.psi = nn.Sequential(
                nn.Conv3d(in_channels=fg, out_channels=1, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid()
            )
            self.resampler = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True) 
        else:
            self.wg = nn.Conv2d(in_channels=fg, out_channels=fg, kernel_size=1, stride=1, padding=0)
            self.wx = nn.Conv2d(in_channels=fx, out_channels=fg, kernel_size=1, stride=2, padding=0)
            self.psi = nn.Sequential(
                nn.Conv2d(in_channels=fg, out_channels=1, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid()
            )
            self.resampler = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.relu = nn.ReLU()

    def forward(self, g, x):
        g = self.wg(g) # up
        xl = self.wx(x) # skip
        
        alpha = self.psi(self.relu(xl + g))
        alpha = self.resampler(alpha)

        return torch.mul(x, alpha)


class CorrectionUnet(nn.Module):
    """
    Unet model with 3 downsampling blocks and 3 upsampling blocks, where
    one block consists of:
    
    `2dConv -> 2dBatchNorm -> ReLu -> 2dConv -> 2dBatchNorm -> ReLu`
    and a sampling operation (`2dMaxPool`/`2dConvTranspose`).
    """

    def __init__(self, in_channels, out_channels, blocks=3, block_channels=[32, 64, 128, 256], use_attention=False, use_dropout=False, volumetric=False):
        super().__init__()

        self.use_attention = use_attention
        self.blocks = blocks
        self.downsampling_path = nn.ModuleList()
        self.upsampling_path = nn.ModuleList()
        self.attention_modules = nn.ModuleList()

        # possible_channels = [in_channels, 24, 48, 96, 192]
        possible_channels = [in_channels] + block_channels
        channels = possible_channels[: blocks + 1]

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.4)

        # Downsampling path
        for i in range(0, len(channels) - 1):
            self.downsampling_path.append(
                DownBlock(
                    in_channels=[channels[i], channels[i + 1]],
                    out_channels=[channels[i + 1], channels[i + 1]],
                    volumetric=volumetric,
                    use_dropout=self.use_dropout
                )
            )

        # Bottle neck
        self.bottle_neck = BottleNeck(
            in_channels=[channels[-1], channels[-1] * 2],
            out_channels=[channels[-1] * 2, channels[-1] * 2],
            volumetric=volumetric
        ) 

        # Upsampling path
        block_i = 0
        for i in range(len(channels) - 1, 0, -1):
            if use_attention:
                self.upsampling_path.append(
                    nn.ModuleList([
                        AttentionBlock(fg=channels[i] * 2, fx=channels[i], volumetric=volumetric),
                        UpBlock(
                            in_channels=[channels[i] * 2, channels[i]],
                            out_channels=[channels[i], channels[i]],
                            volumetric=volumetric,
                            use_dropout= self.use_dropout if block_i < self.blocks - 1 else False
                        )
                    ])
                )
            else:
                self.upsampling_path.append(
                    UpBlock(
                        in_channels=[channels[i] * 2, channels[i]],
                        out_channels=[channels[i], channels[i]],
                        volumetric=volumetric,
                        use_dropout= self.use_dropout if block_i < self.blocks - 1 else False
                    )
                )

            block_i += 1

        # Output
        self.output = Output(channels[1], out_channels, volumetric=volumetric)

    def forward(self, x):
        out = x

        # Downsampling path
        skips = []
        for block in self.downsampling_path:
            skip, out = block(out)
            skips.append(skip)
            # if self.use_dropout:
            #     out = self.dropout(out)

        # Bottle neck
        out = self.bottle_neck(out)

        # Upsampling path
        skips.reverse()
        if self.use_attention:
            # block_i = 0
            for (attention, block), skip in zip(self.upsampling_path, skips):
                skip = attention(g=out, x=skip)
                out = block(out, skip)
                # if block_i != self.blocks - 1 and self.use_dropout:
                #     out = self.dropout(out)
                # block_i += 1
                                       
        else:
            for block, skip in zip(self.upsampling_path, skips):
                out = block(out, skip)

        # Output
        out = self.output(out)
        return out


class MultiModalCorrectionUnet(nn.Module):
    """
    Unet with multiple encoders...
    """

    def __init__(self, in_channels, out_channels, blocks=3, encoders=2, block_channels=[32, 64, 128, 256], volumetric=False, use_dropout=False):
        super().__init__()
        self.encoders = encoders
        self.use_dropout = use_dropout
        self.blocks = blocks

        # Downsampling path
        self.down1_1 = DownBlock(
            in_channels=[in_channels[0], block_channels[0]],
            out_channels=[block_channels[0], block_channels[0]],
            volumetric=volumetric,
            use_dropout=use_dropout,
        )
        self.down1_2 = DownBlock(
            in_channels=[block_channels[0], block_channels[1]],
            out_channels=[block_channels[1], block_channels[1]],
            volumetric=volumetric,
            use_dropout=use_dropout,
        )
        self.down1_3 = DownBlock(
            in_channels=[block_channels[1], block_channels[2]],
            out_channels=[block_channels[2], block_channels[2]],
            volumetric=volumetric,
            use_dropout=use_dropout,
        )
        self.down1_4 = DownBlock(
            in_channels=[block_channels[2], block_channels[3]],
            out_channels=[block_channels[3], block_channels[3]],
            volumetric=volumetric,
            use_dropout=use_dropout,
        )

        self.down2_1 = DownBlock(
            in_channels=[in_channels[1], block_channels[0]],
            out_channels=[block_channels[0], block_channels[0]],
            volumetric=volumetric,
            use_dropout=use_dropout,
        )
        self.down2_2 = DownBlock(
            in_channels=[block_channels[0], block_channels[1]],
            out_channels=[block_channels[1], block_channels[1]],
            volumetric=volumetric,
            use_dropout=use_dropout,
        )
        self.down2_3 = DownBlock(
            in_channels=[block_channels[1], block_channels[2]],
            out_channels=[block_channels[2], block_channels[2]],
            volumetric=volumetric,
            use_dropout=use_dropout,
        )
        self.down2_4 = DownBlock(
            in_channels=[block_channels[2], block_channels[3]],
            out_channels=[block_channels[3], block_channels[3]],
            volumetric=volumetric,
            use_dropout=use_dropout,
        )

        # Bottle neck
        self.bottle_neck = BottleNeck(
            in_channels=[block_channels[3] * 2, block_channels[3] * 2],
            out_channels=[block_channels[3] * 2, block_channels[3] * 2],
            volumetric=volumetric,
        )

        # Upsampling path
        self.up4 = UpBlock(
            in_channels=[block_channels[3] * 2, block_channels[3]],
            out_channels=[block_channels[3], block_channels[3]],
            volumetric=volumetric,
            multi_enc=True,
            use_dropout=use_dropout,
        )
        self.att1_4 = AttentionBlock(fg=block_channels[3] * 2, fx=block_channels[3], volumetric=volumetric)
        self.att2_4 = AttentionBlock(fg=block_channels[3] * 2, fx=block_channels[3], volumetric=volumetric)

        self.up3 = UpBlock(
            in_channels=[block_channels[3], block_channels[2]],
            out_channels=[block_channels[2], block_channels[2]],
            volumetric=volumetric,
            multi_enc=True,
            use_dropout=use_dropout,
        )
        self.att1_3 = AttentionBlock(fg=block_channels[3], fx=block_channels[2], volumetric=volumetric)
        self.att2_3 = AttentionBlock(fg=block_channels[3], fx=block_channels[2], volumetric=volumetric)

        self.up2 = UpBlock(
            in_channels=[block_channels[2], block_channels[1]],
            out_channels=[block_channels[1], block_channels[1]],
            volumetric=volumetric,
            multi_enc=True,
            use_dropout=use_dropout,
        )
        self.att1_2 = AttentionBlock(fg=block_channels[2], fx=block_channels[1], volumetric=volumetric)
        self.att2_2 = AttentionBlock(fg=block_channels[2], fx=block_channels[1], volumetric=volumetric)

        self.up1 = UpBlock(
            in_channels=[block_channels[1], block_channels[0]],
            out_channels=[block_channels[0], block_channels[0]],
            volumetric=volumetric,
            multi_enc=True,
            use_dropout=False,
        )
        self.att1_1 = AttentionBlock(fg=block_channels[1], fx=block_channels[0], volumetric=volumetric)
        self.att2_1 = AttentionBlock(fg=block_channels[1], fx=block_channels[0], volumetric=volumetric)

        # Output
        self.output = Output(block_channels[0], out_channels, volumetric=volumetric)

    def forward(self, x):
        if self.encoders == 2:
            inputs = [x[:,0].unsqueeze(1), x[:,1:]]
        else:
            inputs = [x[:,i].unsqueeze(1) for i in range(x.shape[1])]
        # print(len(inputs), inputs[0].shape, inputs[1].shape)

        # Downsampling path
        skip1_1, out1_1 = self.down1_1(inputs[0])
        skip1_2, out1_2 = self.down1_2(out1_1)
        skip1_3, out1_3 = self.down1_3(out1_2)
        skip1_4, out1_4 = self.down1_4(out1_3)

        skip2_1, out2_1 = self.down2_1(inputs[1])
        skip2_2, out2_2 = self.down2_2(out2_1)
        skip2_3, out2_3 = self.down2_3(out2_2)
        skip2_4, out2_4 = self.down2_4(out2_3)

        # Bottle neck
        out = torch.cat([out1_4, out2_4], dim=1)
        # out = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0)(out)
        out = self.bottle_neck(out)

        # Upsampling path
        skip1 = self.att1_4(out, skip1_4)
        skip2 = self.att2_4(out, skip2_4)
        skip = torch.cat([skip1, skip2], dim=1)
        out = self.up4(out, skip)

        skip1 = self.att1_3(out, skip1_3)
        skip2 = self.att2_3(out, skip2_3)
        skip = torch.cat([skip1, skip2], dim=1)
        out = self.up3(out, skip)

        skip1 = self.att1_2(out, skip1_2)
        skip2 = self.att2_2(out, skip2_2)
        skip = torch.cat([skip1, skip2], dim=1)
        out = self.up2(out, skip)

        skip1 = self.att1_1(out, skip1_1)
        skip2 = self.att2_1(out, skip2_1)
        skip = torch.cat([skip1, skip2], dim=1)
        out = self.up1(out, skip)

        # Output
        out = self.output(out)
        return out


class MultiModal3BlockCorrectionUnet(nn.Module):
    """
    Unet with multiple encoders...
    """

    def __init__(self, in_channels, out_channels, blocks=3, encoders=2, block_channels=[32, 64, 128], volumetric=False, use_dropout=False):
        super().__init__()
        self.encoders = encoders
        self.use_dropout = use_dropout
        self.blocks = blocks

        # Downsampling path
        self.down1_1 = DownBlock(
            in_channels=[in_channels[0], block_channels[0]],
            out_channels=[block_channels[0], block_channels[0]],
            volumetric=volumetric,
            use_dropout=use_dropout,
        )
        self.down1_2 = DownBlock(
            in_channels=[block_channels[0], block_channels[1]],
            out_channels=[block_channels[1], block_channels[1]],
            volumetric=volumetric,
            use_dropout=use_dropout,
        )
        self.down1_3 = DownBlock(
            in_channels=[block_channels[1], block_channels[2]],
            out_channels=[block_channels[2], block_channels[2]],
            volumetric=volumetric,
            use_dropout=use_dropout,
        )

        self.down2_1 = DownBlock(
            in_channels=[in_channels[1], block_channels[0]],
            out_channels=[block_channels[0], block_channels[0]],
            volumetric=volumetric,
            use_dropout=use_dropout,
        )
        self.down2_2 = DownBlock(
            in_channels=[block_channels[0], block_channels[1]],
            out_channels=[block_channels[1], block_channels[1]],
            volumetric=volumetric,
            use_dropout=use_dropout,
        )
        self.down2_3 = DownBlock(
            in_channels=[block_channels[1], block_channels[2]],
            out_channels=[block_channels[2], block_channels[2]],
            volumetric=volumetric,
            use_dropout=use_dropout,
        )

        # Bottle neck
        self.bottle_neck = BottleNeck(
            in_channels=[block_channels[2] * 2, block_channels[2] * 2],
            out_channels=[block_channels[2] * 2, block_channels[2] * 2],
            volumetric=volumetric,
        )

        # Upsampling path
        self.up3 = UpBlock(
            in_channels=[block_channels[2] * 2, block_channels[2]],
            out_channels=[block_channels[2], block_channels[2]],
            volumetric=volumetric,
            multi_enc=True,
            use_dropout=use_dropout,
        )
        self.att1_3 = AttentionBlock(fg=block_channels[2] * 2, fx=block_channels[2], volumetric=volumetric)
        self.att2_3 = AttentionBlock(fg=block_channels[2] * 2, fx=block_channels[2], volumetric=volumetric)

        self.up2 = UpBlock(
            in_channels=[block_channels[2], block_channels[1]],
            out_channels=[block_channels[1], block_channels[1]],
            volumetric=volumetric,
            multi_enc=True,
            use_dropout=use_dropout,
        )
        self.att1_2 = AttentionBlock(fg=block_channels[2], fx=block_channels[1], volumetric=volumetric)
        self.att2_2 = AttentionBlock(fg=block_channels[2], fx=block_channels[1], volumetric=volumetric)

        self.up1 = UpBlock(
            in_channels=[block_channels[1], block_channels[0]],
            out_channels=[block_channels[0], block_channels[0]],
            volumetric=volumetric,
            multi_enc=True,
            use_dropout=False,
        )
        self.att1_1 = AttentionBlock(fg=block_channels[1], fx=block_channels[0], volumetric=volumetric)
        self.att2_1 = AttentionBlock(fg=block_channels[1], fx=block_channels[0], volumetric=volumetric)

        # Output
        self.output = Output(block_channels[0], out_channels, volumetric=volumetric)

    def forward(self, x):
        if self.encoders == 2:
            inputs = [x[:,0].unsqueeze(1), x[:,1:]]
        else:
            inputs = [x[:,i].unsqueeze(1) for i in range(x.shape[1])]
        # print(len(inputs), inputs[0].shape, inputs[1].shape)

        # Downsampling path
        skip1_1, out1_1 = self.down1_1(inputs[0])
        skip1_2, out1_2 = self.down1_2(out1_1)
        skip1_3, out1_3 = self.down1_3(out1_2)

        skip2_1, out2_1 = self.down2_1(inputs[1])
        skip2_2, out2_2 = self.down2_2(out2_1)
        skip2_3, out2_3 = self.down2_3(out2_2)

        # Bottle neck
        out = torch.cat([out1_3, out2_3], dim=1)
        # out = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0)(out)
        out = self.bottle_neck(out)

        # Upsampling path
        skip1 = self.att1_3(out, skip1_3)
        skip2 = self.att2_3(out, skip2_3)
        skip = torch.cat([skip1, skip2], dim=1)
        out = self.up3(out, skip)

        skip1 = self.att1_2(out, skip1_2)
        skip2 = self.att2_2(out, skip2_2)
        skip = torch.cat([skip1, skip2], dim=1)
        out = self.up2(out, skip)

        skip1 = self.att1_1(out, skip1_1)
        skip2 = self.att2_1(out, skip2_1)
        skip = torch.cat([skip1, skip2], dim=1)
        out = self.up1(out, skip)

        # Output
        out = self.output(out)
        return out
    

class OGCorrectionUnet(nn.Module):
    """
    Original old correction Unet with 3 downsampling blocks and 3 upsampling blocks, where
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
    # model = MultiModalCorrectionUnet(in_channels=[1, 2], encoders=2, out_channels=1, blocks=4)
    # x = torch.rand(size=(2, 3, 48, 48))
    # model(x)

    summary(
        MultiModal3BlockCorrectionUnet(in_channels=[1, 2], encoders=2, out_channels=1, blocks=3, volumetric=True, use_dropout=True),
        input_size=(2, 3, 16, 48, 48),
        depth=2
    )

    # summary(
    #     CorrectionUnet(in_channels=1, out_channels=1, blocks=3, use_attention=True, volumetric=False, use_dropout=True, block_channels=[32, 64, 128, 256]),
    #     input_size=(2, 1, 48, 48),
    #     depth=3
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
