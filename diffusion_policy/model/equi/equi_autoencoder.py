from escnn import gspaces, nn
import torch


class EquiResBlock(torch.nn.Module):
    def __init__(
        self,
        group: gspaces.GSpace2D,
        input_channels: int,
        hidden_dim: int,
        kernel_size: int = 3,
        stride: int = 1,
        initialize: bool = True,
    ):
        super(EquiResBlock, self).__init__()
        self.group = group
        rep = self.group.regular_repr

        feat_type_in = nn.FieldType(self.group, input_channels * [rep])
        feat_type_hid = nn.FieldType(self.group, hidden_dim * [rep])

        self.layer1 = nn.SequentialModule(
            nn.R2Conv(
                feat_type_in,
                feat_type_hid,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                stride=stride,
                initialize=initialize,
            ),
            nn.ReLU(feat_type_hid, inplace=True),
        )

        self.layer2 = nn.SequentialModule(
            nn.R2Conv(
                feat_type_hid,
                feat_type_hid,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                initialize=initialize,
            ),
        )
        self.relu = nn.ReLU(feat_type_hid, inplace=True)

        self.upscale = None
        if input_channels != hidden_dim or stride != 1:
            self.upscale = nn.SequentialModule(
                nn.R2Conv(feat_type_in, feat_type_hid, kernel_size=1, stride=stride, bias=False, initialize=initialize),
            )
        
        self.batch_norm = nn.InnerBatchNorm(feat_type_hid)

    def forward(self, xx: nn.GeometricTensor) -> nn.GeometricTensor:
        residual = xx
        out = self.layer1(xx)
        out = self.layer2(out)
        if self.upscale:
            out += self.upscale(residual)
        else:
            out += residual
        out = self.relu(self.batch_norm(out))

        return out
    
class EquivResEnc96to24(torch.nn.Module):
    def __init__(self, 
                 in_channels: int = 3, 
                 out_channels: int = 3, 
                 initialize: bool = True, 
                 N=8,
                 NUM_CHANNEL_1=32,
                 NUM_CHANNEL_2=64
                 ):
        super(EquivResEnc96to24, self).__init__()
        self.in_channels = in_channels
        self.group = gspaces.rot2dOnR2(N)
        self.conv = torch.nn.Sequential(
            # 3x96x96
            nn.R2Conv(
                nn.FieldType(self.group, in_channels * [self.group.trivial_repr]),
                nn.FieldType(self.group, NUM_CHANNEL_1 * [self.group.regular_repr]),
                kernel_size=3,
                stride=1,
                padding=1,
                initialize=initialize,
            ),
            # 32x96x96
            nn.ReLU(nn.FieldType(self.group, NUM_CHANNEL_1 * [self.group.regular_repr]), inplace=True),
            EquiResBlock(self.group, NUM_CHANNEL_1, NUM_CHANNEL_2, initialize=True),
            EquiResBlock(self.group, NUM_CHANNEL_2, NUM_CHANNEL_2, initialize=True),
            nn.PointwiseMaxPool(nn.FieldType(self.group, NUM_CHANNEL_2 * [self.group.regular_repr]), 2),
            # 64x48x48
            EquiResBlock(self.group, NUM_CHANNEL_2, NUM_CHANNEL_2, initialize=True),
            EquiResBlock(self.group, NUM_CHANNEL_2, NUM_CHANNEL_1, initialize=True),    
            nn.PointwiseMaxPool(nn.FieldType(self.group, NUM_CHANNEL_1 * [self.group.regular_repr]), 2),
            # 32x24x24
            nn.R2Conv(
                nn.FieldType(self.group, NUM_CHANNEL_1 * [self.group.regular_repr]),
                nn.FieldType(self.group, out_channels * [self.group.trivial_repr]),
                kernel_size=3,
                stride=1,
                padding=1,
                initialize=initialize,
            ),
            nn.ReLU(nn.FieldType(self.group, out_channels * [self.group.trivial_repr]), inplace=True),
            # 1x24x24
        )

    def forward(self, x) -> torch.Tensor:
        if type(x) is torch.Tensor:
            x = nn.GeometricTensor(x, nn.FieldType(self.group, self.in_channels * [self.group.trivial_repr]))
        return self.conv(x).tensor
    

class EquiResUpBlock(torch.nn.Module):
    def __init__(self, group, in_channels, out_channels, kernel_size=3, scale_factor=2, initialize=True):
        super(EquiResUpBlock, self).__init__()
        rep = group.regular_repr
        
        feat_type_in = nn.FieldType(group, in_channels * [rep])
        feat_type_out = nn.FieldType(group, out_channels * [rep])
        
        self.upsample = nn.R2Upsampling(
            in_type=feat_type_in,
            scale_factor=scale_factor,
            mode="bilinear",
            align_corners=True
        )
        
        self.layer1 = nn.SequentialModule(
            nn.R2Conv(
                feat_type_in, feat_type_out,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                stride=1,
                initialize=initialize
            ),
            nn.ReLU(feat_type_out, inplace=True)
        )
        
        self.layer2 = nn.SequentialModule(
            nn.R2Conv(
                feat_type_out, feat_type_out,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                stride=1,
                initialize=initialize
            )
        )
        
        self.skip = None
        if in_channels != out_channels:
            self.skip = nn.SequentialModule(
                nn.R2Conv(
                    feat_type_in, feat_type_out,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                    initialize=initialize
                )
            )
            
        self.batch_norm = nn.InnerBatchNorm(feat_type_out)
        
        self.relu = nn.ReLU(feat_type_out, inplace=True)
        self.out_type = feat_type_out
        
    def forward(self, x: nn.GeometricTensor) -> nn.GeometricTensor:
        x_upsampled = self.upsample(x)
        out = self.layer1(x_upsampled)
        out = self.layer2(out)
        if self.skip is not None:
            shortcut = self.skip(x_upsampled)
        else:
            shortcut = x_upsampled
        out = out + shortcut
        out = self.batch_norm(out)
        out = self.relu(out)
        return out


class EquivResDec24to96(torch.nn.Module):
    def __init__(self, 
                 in_channels: int = 3, 
                 out_channels: int = 3, 
                 initialize: bool = True,
                 N: int = 8,
                 NUM_CHANNEL_1=32,
                 NUM_CHANNEL_2=64
                 ):
        super(EquivResDec24to96, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.group = gspaces.rot2dOnR2(N)
        
        self.up_blocks = torch.nn.Sequential(
            # 1x24x24
            nn.R2Conv(
                nn.FieldType(self.group, in_channels * [self.group.trivial_repr]),
                nn.FieldType(self.group, NUM_CHANNEL_1 * [self.group.regular_repr]),
                kernel_size=3,
                stride=1,
                padding=1,
                initialize=initialize,
            ),
            # 32x24x24
            EquiResUpBlock(self.group, NUM_CHANNEL_1, NUM_CHANNEL_2, initialize=initialize),
            # 64x48x48
            EquiResUpBlock(self.group, NUM_CHANNEL_2, NUM_CHANNEL_1, initialize=initialize),
            # 64x96x96
            nn.R2Conv(
                nn.FieldType(self.group, NUM_CHANNEL_1 * [self.group.regular_repr]),
                nn.FieldType(self.group, out_channels * [self.group.trivial_repr]),
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
                initialize=initialize
            ),
        )
        self.tanh = torch.nn.Tanh()
        
    def forward(self, x) -> torch.Tensor:
        if type(x) is torch.Tensor:
            x = nn.GeometricTensor(x, nn.FieldType(self.group, self.in_channels * [self.group.trivial_repr]))
        return self.tanh(self.up_blocks(x).tensor)
        return self.up_blocks(x).tensor
    
    
    
    
####################################################################################################################################################
####################################################### Not Equivariant Version ####################################################################
class ResBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
    ):
        super(ResBlock, self).__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                stride=stride,
            ),
            torch.nn.ReLU(inplace=True),
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                stride=1
            ),
        )

        self.upscale = None
        if in_channels != out_channels or stride != 1:
            self.upscale = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels, 
                    out_channels, 
                    kernel_size=1, 
                    stride=stride, 
                    bias=False, 
                ),
            )
        self.relu = torch.nn.ReLU(inplace=True)
        self.batch_norm = torch.nn.BatchNorm2d(out_channels)

    def forward(self, xx: torch.Tensor) -> torch.Tensor:
        residual = xx
        out = self.layer1(xx)
        out = self.layer2(out)
        if self.upscale:
            out += self.upscale(residual)
        else:
            out += residual
        out = self.relu(self.batch_norm(out))

        return out
    
class ResEnc96to24(torch.nn.Module):
    def __init__(self, 
                 in_channels: int = 3, 
                 out_channels: int = 3,
                 NUM_CHANNEL_1=32,
                 NUM_CHANNEL_2=64
                 ):
        super(ResEnc96to24, self).__init__()
        self.in_channels = in_channels
        self.conv = torch.nn.Sequential(
            # 3x96x96
            torch.nn.Conv2d(
                in_channels,
                NUM_CHANNEL_1,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            # 32x48x48
            torch.nn.ReLU(inplace=True),
            ResBlock(NUM_CHANNEL_1, NUM_CHANNEL_2),
            ResBlock(NUM_CHANNEL_2, NUM_CHANNEL_2),
            torch.nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
            # 64x48x48
            ResBlock(NUM_CHANNEL_2, NUM_CHANNEL_2),
            ResBlock(NUM_CHANNEL_2, NUM_CHANNEL_1),    
            torch.nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
            # 32x24x24
            torch.nn.Conv2d(
                NUM_CHANNEL_1,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            # 3x24x24
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x) -> torch.Tensor:
        return self.conv(x)
    

class ResUpBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, scale_factor=2):
        super(ResUpBlock, self).__init__()
        
        self.upsample = torch.nn.Upsample(
            scale_factor=scale_factor,
            mode='bilinear',
            align_corners=True
        )
        
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
            ),
            torch.nn.ReLU(inplace=True)
        )
        
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
            ),
        )
        
        self.skip = None
        if in_channels != out_channels:
            self.skip = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False
                ),
            )
            
        self.batch_norm = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_upsampled = self.upsample(x)
        out = self.layer1(x_upsampled)
        out = self.layer2(out)
        if self.skip is not None:
            shortcut = self.skip(x_upsampled)
        else:
            shortcut = x_upsampled
        out = out + shortcut
        out = self.batch_norm(out)
        out = self.relu(out)
        return out


class ResDec24to96(torch.nn.Module):
    def __init__(self, 
                 in_channels: int = 3, 
                 out_channels: int = 3,
                 NUM_CHANNEL_1=32,
                 NUM_CHANNEL_2=64
                 ):
        super(ResDec24to96, self).__init__()
        
        self.up_blocks = torch.nn.Sequential(
            # 3x24x24
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=NUM_CHANNEL_1,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            # 32x24x24
            ResUpBlock(NUM_CHANNEL_1, NUM_CHANNEL_2),
            # 64x48x48
            ResUpBlock(NUM_CHANNEL_2, NUM_CHANNEL_1),
            # 64x96x96
            torch.nn.Conv2d(
                in_channels=NUM_CHANNEL_1,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            torch.nn.Tanh(),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up_blocks(x)

class ResEncoder(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=3, channels_list=[8, 16, 32, 16, 8]):
        super().__init__()

        layers = []

        # Initial conv layer
        layers.append(torch.nn.Conv2d(in_channels, channels_list[0], kernel_size=3, stride=1, padding=1))
        layers.append(torch.nn.ReLU(inplace=True))

        # Residual blocks with pooling
        for i in range(len(channels_list)-1):
            layers.append(ResBlock(channels_list[i], channels_list[i + 1]))
            layers.append(torch.nn.MaxPool2d(kernel_size=2))

        # Final conv to project to out_channels
        layers.append(torch.nn.Conv2d(channels_list[-1], out_channels, kernel_size=3, stride=1, padding=1))
        layers.append(torch.nn.ReLU(inplace=True))  # Optional

        self.encoder = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)

class ResDecoder(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=3, channels_list=[8, 16, 32, 16, 8]):
        super().__init__()
        
        layers = []

        # Initial conv to expand channels
        layers.append(torch.nn.Conv2d(in_channels, channels_list[0], kernel_size=3, padding=1))
        layers.append(torch.nn.ReLU(inplace=True))

        # ResUpBlocks with upsampling
        for i in range(len(channels_list) - 1):
            layers.append(ResUpBlock(channels_list[i], channels_list[i + 1]))

        # Final projection
        layers.append(torch.nn.Conv2d(channels_list[-1], out_channels, kernel_size=1))
        layers.append(torch.nn.Tanh())  # or Sigmoid / ReLU depending on your use case

        self.decoder = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)