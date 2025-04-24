import math
import torch
import torch.nn as nn
from escnn import gspaces
import escnn.nn as enn

# Define the rotation group on the plane (e.g., 8-fold symmetry)
G = gspaces.rot2dOnR2(N=8)

class EquiAdaNorm(nn.Module):
    """
    Adaptive normalization for geometric tensors using InnerBatchNorm and learned scale/shift from conditioning.
    """
    def __init__(self, field_type: enn.FieldType, cond_dim: int):
        super().__init__()
        self.norm = enn.InnerBatchNorm(field_type)
        # one scale and shift per channel in field_type
        num_channels = field_type.size
        self.linear = nn.Linear(cond_dim, num_channels * 2)

    def forward(self, x: enn.GeometricTensor, cond: torch.Tensor) -> enn.GeometricTensor:
        # x.tensor shape: [B, C, H, W]
        x = self.norm(x)
        B, C, H, W = x.tensor.shape
        # cond: [B, cond_dim]
        scale, shift = self.linear(cond).view(B, C, 2).permute(2, 0, 1).chunk(2, dim=0)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        y = x.tensor * (1 + scale) + shift
        return enn.GeometricTensor(y, x.type)

class EquiDownsample(nn.Module):
    """Equivariant downsampling by stride-2 convolution"""
    def __init__(self, in_type: enn.FieldType):
        super().__init__()
        self.conv = enn.R2Conv(in_type, in_type, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x: enn.GeometricTensor) -> enn.GeometricTensor:
        return self.conv(x)

class EquiUpsample(nn.Module):
    """Equivariant upsampling by interpolation + pointwise conv"""
    def __init__(self, in_type: enn.FieldType):
        super().__init__()
        self.conv = enn.R2Conv(in_type, in_type, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x: enn.GeometricTensor) -> enn.GeometricTensor:
        y = x.tensor
        y = nn.functional.interpolate(y, scale_factor=2.0, mode='nearest')
        return enn.GeometricTensor(self.conv(enn.GeometricTensor(y, x.type)).tensor, x.type)

class EquiResBlock(nn.Module):
    """Equivariant residual block with adaptive normalization and ReLU gating"""
    def __init__(self,
                 in_type: enn.FieldType,
                 out_type: enn.FieldType,
                 cond_dim: int):
        super().__init__()
        # project if necessary
        self.need_proj = in_type != out_type
        if self.need_proj:
            self.proj = enn.R2Conv(in_type, out_type, kernel_size=1, stride=1, padding=0, bias=False)
        # layers
        self.norm1 = EquiAdaNorm(in_type, cond_dim)
        self.conv1 = enn.R2Conv(in_type, out_type, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = EquiAdaNorm(out_type, cond_dim)
        self.conv2 = enn.R2Conv(out_type, out_type, kernel_size=3, stride=1, padding=1, bias=False)
        # initialize second conv to zero for identity init
        nn.init.zeros_(self.conv2.tensor.weight)
        self.relu = enn.PointwiseNonLinearity(out_type, 'relu')

    def forward(self, x: enn.GeometricTensor, cond: torch.Tensor) -> enn.GeometricTensor:
        identity = x
        if self.need_proj:
            identity = self.proj(x)
        out = self.norm1(x, cond)
        out = self.conv1(out)
        out = self.norm2(out, cond)
        out = self.conv2(out)
        out = enn.GeometricTensor(out.tensor + identity.tensor, out.type)
        return self.relu(out)

class EquiResBlocks(nn.Module):
    """Sequence of equivariant residual blocks"""
    def __init__(self,
                 types_in: list,
                 types_out: list,
                 cond_dim: int):
        super().__init__()
        assert len(types_in) == len(types_out)
        self.blocks = nn.ModuleList([
            EquiResBlock(t_in, t_out, cond_dim)
            for t_in, t_out in zip(types_in, types_out)
        ])

    def forward(self, x: enn.GeometricTensor, cond: torch.Tensor) -> tuple[enn.GeometricTensor, list]:
        outputs = []
        for block in self.blocks:
            x = block(x, cond)
            outputs.append(x)
        return x, outputs

class EquiConditionalUNet2D(nn.Module):
    def __init__(self,
                 cond_dim: int,
                 depths: list,
                 channels: list,
                 action_ch: int):
        super().__init__()
        assert len(depths) == len(channels)
        self.num_levels = len(channels)
        # define field types for features and actions
        rep = G.regular_repr
        feat_types = [enn.FieldType(G, c * [rep]) for c in channels]
        action_type = enn.FieldType(G, action_ch * [rep])

        # action convs per level
        self.action_convs = nn.ModuleList([
            enn.R2Conv(action_type, feat_types[i], kernel_size=3, stride=1, padding=1, bias=False)
            for i in range(self.num_levels)
        ])
        # mid action conv
        self.mid_action_conv = enn.R2Conv(action_type, feat_types[-1], kernel_size=3, stride=1, padding=1, bias=False)

        # down and up blocks
        self.d_blocks = nn.ModuleList()
        self.u_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList([nn.Identity()])
        self.upsamples = nn.ModuleList([nn.Identity()])
        for i in range(self.num_levels):
            # down path
            types_in = [feat_types[i]] * depths[i]
            types_out = [feat_types[i]] * depths[i]
            self.d_blocks.append(EquiResBlocks(types_in, types_out, cond_dim))
            if i > 0:
                self.downsamples.append(EquiDownsample(feat_types[i]))
        # middle blocks
        mid_types_in = [feat_types[-1]] * depths[-1] + [feat_types[-1]] * depths[-1]
        mid_types_out = [feat_types[-1]] * (depths[-1] * 2)
        self.mid_blocks = EquiResBlocks(mid_types_in, mid_types_out, cond_dim)

        for i in range(self.num_levels):
            # up path
            idx = self.num_levels - 1 - i
            self.upsamples.append(EquiUpsample(feat_types[idx]))
            types_in = [feat_types[idx]] * depths[idx]
            types_out = [feat_types[idx]] * depths[idx]
            self.u_blocks.append(EquiResBlocks(types_in, types_out, cond_dim))

    def forward(self, x: torch.Tensor, cond: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # wrap into GeometricTensor
        in_type = enn.FieldType(G, x.size(1) * [G.regular_repr])
        x = enn.GeometricTensor(x, in_type)
        cond = cond
        # action tensor
        a = enn.GeometricTensor(action, enn.FieldType(G, action.size(1) * [G.regular_repr]))

        skips = []
        # down path
        for down, block, a_conv in zip(self.downsamples, self.d_blocks, self.action_convs):
            x = down(x)
            a_map = enn.GeometricTensor(
                nn.functional.interpolate(a.tensor, size=(x.tensor.size(2), x.tensor.size(3)), mode='bilinear', align_corners=False),
                a.type)
            a_map = a_conv(a_map)
            x = enn.GeometricTensor(torch.cat([x.tensor, a_map.tensor], dim=1), x.type)
            x, out = block(x, cond)
            skips.append(x)

        # middle
        a_map = enn.GeometricTensor(
            nn.functional.interpolate(a.tensor, size=(x.tensor.size(2), x.tensor.size(3)), mode='bilinear', align_corners=False),
            a.type)
        a_map = self.mid_action_conv(a_map)
        x = enn.GeometricTensor(torch.cat([x.tensor, a_map.tensor], dim=1), x.type)
        x, _ = self.mid_blocks(x, cond)

        # up path
        for up, block, skip, a_conv in zip(self.upsamples, self.u_blocks, reversed(skips), reversed(self.action_convs)):
            x = up(x)
            a_map = enn.GeometricTensor(
                nn.functional.interpolate(a.tensor, size=(x.tensor.size(2), x.tensor.size(3)), mode='bilinear', align_corners=False),
                a.type)
            a_map = a_conv(a_map)
            x = enn.GeometricTensor(torch.cat([x.tensor, skip.tensor, a_map.tensor], dim=1), x.type)
            x, _ = block(x, cond)

        return x.tensor
