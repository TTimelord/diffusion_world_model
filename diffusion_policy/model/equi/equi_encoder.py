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
            EquiResBlock(self.group, self.in_channels, n_out // 8, initialize=True),
            EquiResBlock(self.group, n_out // 8, n_out // 8, initialize=True),
            nn.PointwiseMaxPool(nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]), 2),
            # 48x48
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
    

class EquiResDecBlock(torch.nn.Module):
    def __init__(
        self,
        group: gspaces.GSpace2D,
        input_channels: int,
        hidden_dim: int,
        kernel_size: int = 3,
        stride: int = 1,
        initialize: bool = True,
    ):
        super(EquiResDecBlock, self).__init__()
        self.group = group
        rep = self.group.regular_repr

        feat_type_in = nn.FieldType(self.group, input_channels * [rep])
        feat_type_hid = nn.FieldType(self.group, hidden_dim * [rep])
        
        # Choose the convolution operator based on stride:
        # For upsampling (stride > 1) use R2ConvTranspose; otherwise use a standard R2Conv.
        conv1 = nn.R2ConvTranspose if stride > 1 else nn.R2Conv

        self.layer1 = nn.SequentialModule(
            conv1(
                feat_type_in,
                feat_type_hid,
                kernel_size=kernel_size,
                stride=stride,
                # For simplicity we use the same padding calculation,
                # but note that in practice padding for transposed convolutions may require further tuning.
                padding=(kernel_size - 1) // 2,
                initialize=initialize,
            ),
            nn.ReLU(feat_type_hid, inplace=True),
        )
        
        # Second layer always uses a standard convolution (stride=1).
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
            # For the residual connection, mirror the conv type used in layer1.
            if stride > 1:
                self.upscale = nn.SequentialModule(
                    nn.R2ConvTranspose(
                        feat_type_in,
                        feat_type_hid,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                        initialize=initialize,
                    ),
                )
            else:
                self.upscale = nn.SequentialModule(
                    nn.R2Conv(
                        feat_type_in,
                        feat_type_hid,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                        initialize=initialize,
                    ),
                )

    def forward(self, x: nn.GeometricTensor) -> nn.GeometricTensor:
        residual = x
        out = self.layer1(x)
        out = self.layer2(out)
        if self.upscale:
            out += self.upscale(residual)
        else:
            out += residual
        out = self.relu(out)
        return out


class EquivariantResDecoder(torch.nn.Module):
    def __init__(self, in_channels: int = 64, out_channels: int = 3, initialize: bool = True, N: int = 8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.group = gspaces.rot2dOnR2(N)
        rep = self.group.regular_repr

        self.trans_conv = torch.nn.Sequential(
            # 12x12
            EquiResDecBlock(self.group, self.in_channels, self.in_channels // 2, initialize=initialize),
            EquiResDecBlock(self.group, self.in_channels // 2, self.in_channels // 2, kernel_size=3, stride=2, initialize=initialize),
            # 24x24
            EquiResDecBlock(self.group, self.in_channels // 2, self.in_channels // 4, initialize=initialize),
            EquiResDecBlock(self.group, self.in_channels // 4, self.in_channels // 4, kernel_size=3, stride=2, initialize=initialize),
            # 48x48
            EquiResDecBlock(self.group, self.in_channels // 4, self.in_channels // 8, initialize=initialize),
            EquiResDecBlock(self.group, self.in_channels // 8, self.in_channels // 8, kernel_size=3, stride=2, initialize=initialize),
            # 96x96
            nn.R2Conv(
                nn.FieldType(self.group, (self.in_channels // 8) * [rep]),
                nn.FieldType(self.group, self.out_channels * [self.group.trivial_repr]),
                kernel_size=3,
                padding=1,
                initialize=initialize,
            ),
            nn.ReLU(nn.FieldType(self.group, self.out_channels * [self.group.trivial_repr]), inplace=True),
        )

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            latent_type = nn.FieldType(self.group, (x.shape[1]) * [self.group.regular_repr])
            x = nn.GeometricTensor(x, latent_type)
        return self.trans_conv(x)
