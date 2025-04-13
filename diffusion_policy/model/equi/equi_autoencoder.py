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

    def forward(self, xx: nn.GeometricTensor) -> nn.GeometricTensor:
        residual = xx
        out = self.layer1(xx)
        out = self.layer2(out)
        if self.upscale:
            out += self.upscale(residual)
        else:
            out += residual
        out = self.relu(out)

        return out
    
class EquivariantResEncoder(torch.nn.Module):
    def __init__(self, in_channels: int = 2, n_out: int = 128, initialize: bool = True, N=8):
        super().__init__()
        self.in_channels = in_channels
        self.group = gspaces.rot2dOnR2(N)
        self.conv = torch.nn.Sequential(
            # 96x96
            nn.R2Conv(
                nn.FieldType(self.group, in_channels * [self.group.trivial_repr]),
                nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]),
                kernel_size=3,
                stride=2,
                padding=1,
                initialize=initialize,
            ),
            # 48x48
            nn.ReLU(nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]), inplace=True),
            # EquiResBlock(self.group, self.in_channels, n_out // 8, initialize=True),
            # EquiResBlock(self.group, n_out // 8, n_out // 8, initialize=True),
            # nn.PointwiseMaxPool(nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]), 2),
            # # 48x48
            EquiResBlock(self.group, n_out // 8, n_out // 4, initialize=True),
            EquiResBlock(self.group, n_out // 4, n_out // 4, initialize=True),
            nn.PointwiseMaxPool(nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]), 2),
            # 24x24
            EquiResBlock(self.group, n_out // 4, n_out // 2, initialize=True),
            EquiResBlock(self.group, n_out // 2, n_out // 2, initialize=True),
            nn.PointwiseMaxPool(nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]), 2),
            # 12x12
        )

    def forward(self, x) -> nn.GeometricTensor:
        if type(x) is torch.Tensor:
            x = nn.GeometricTensor(x, nn.FieldType(self.group, self.in_channels * [self.group.trivial_repr]))
        return self.conv(x)
    

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
        out = self.relu(out)
        return out


class EquivariantResDecoder(torch.nn.Module):
    def __init__(self, in_channels: int = 64, out_channels: int = 3, initialize: bool = True, N: int = 8):
        super(EquivariantResDecoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.group = gspaces.rot2dOnR2(N)
        
        self.up_blocks = torch.nn.Sequential(
            # 12x12
            EquiResUpBlock(self.group, in_channels, in_channels // 2, initialize=initialize),
            # 24x24
            EquiResUpBlock(self.group, in_channels // 2, in_channels // 4, initialize=initialize),
            # 48x48
            EquiResUpBlock(self.group, in_channels // 4, in_channels // 4, initialize=initialize),
            # 96x96
            nn.R2Conv(
                nn.FieldType(self.group, in_channels // 4 * [self.group.regular_repr]),
                nn.FieldType(self.group, out_channels * [self.group.trivial_repr]),
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
                initialize=initialize
            ),
            nn.ReLU(nn.FieldType(self.group, out_channels * [self.group.trivial_repr]), inplace=True),
        )
        
    def forward(self, x: nn.GeometricTensor) -> nn.GeometricTensor:
        if type(x) is torch.Tensor:
            x = nn.GeometricTensor(x, nn.FieldType(self.group, self.in_channels * [self.group.regular_repr]))
        return self.up_blocks(x)