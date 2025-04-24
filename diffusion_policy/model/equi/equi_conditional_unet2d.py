from functools import partial
import math
from typing import List, Optional
from einops import rearrange
import torch
from torch import Tensor
from torch.nn import functional as F
import escnn.nn as nn
import escnn
from escnn import gspaces


from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb

# Settings for GroupNorm and Attention

GN_GROUP_SIZE = 32
GN_EPS = 1e-5
ATTN_HEAD_DIM = 8

# Convs

Conv1x1 = partial(torch.nn.Conv2d, kernel_size=1, stride=1, padding=0)
Conv3x3 = partial(torch.nn.Conv2d, kernel_size=3, stride=1, padding=1)


EquiConv1x1 = partial(nn.R2Conv, kernel_size=1, stride=1, padding=0, initialize=True)
EquiConv3x3 = partial(nn.R2Conv, kernel_size=3, stride=1, padding=1, initialize=True)


class EquiAdaGroupNorm(torch.nn.Module):
    """
    Equivariant AdaGroupNorm, equivariant to x, invariant to cond;
    AdaGroupNorm is equivariant itself
    """
    def __init__(self, group: gspaces.GSpace2D, sample_channels: int, cond_channels: int) -> None:
        super().__init__()
        self.group = group
        self.sample_channels = sample_channels
        self.num_groups = max(1, sample_channels // GN_GROUP_SIZE)
        self.linear = torch.nn.Linear(cond_channels, sample_channels * 2)
        self.field_norm = nn.FieldNorm(nn.FieldType(self.group, self.sample_channels * [self.group.trivial_repr]), eps=GN_EPS)

    def forward(self, x: nn.GeometricTensor, cond: Tensor) -> nn.GeometricTensor:
        # trivial in, trivial out
        assert x.shape[1] == self.sample_channels
        x = self.field_norm(x)
        scale, shift = self.linear(cond)[:, :, None, None].chunk(2, dim=1)
        return x * (1 + scale) + shift


class EquiDownsample(torch.nn.Module):
    def __init__(self, group: gspaces.GSpace2D, channels: int) -> None:
        super().__init__()
        self.group = group

        self.maxpooling = nn.PointwiseMaxPool2D(nn.FieldType(self.group, channels*[self.group.trivial_repr]), 2)
        self.conv = nn.R2Conv(
            in_type=nn.FieldType(self.group, channels*[self.group.trivial_repr]),
            out_type=nn.FieldType(self.group, channels*[self.group.trivial_repr]),
            kernel_size=3,
            stride=1,
            padding=1,
            initialize=True
        )

    def forward(self, x: nn.GeometricTensor) -> nn.GeometricTensor:
        return self.maxpooling(self.conv(x))


class EquiUpsample(torch.nn.Module):
    def __init__(self, group: gspaces.GSpace2D, channels: int) -> None:
        super().__init__()
        self.group = group
        self.conv = nn.R2Conv(
            in_type=nn.FieldType(self.group, channels*[self.group.trivial_repr]),
            out_type=nn.FieldType(self.group, channels*[self.group.trivial_repr]),
            kernel_size=3,
            stride=1,
            padding=1,
            initialize=True
        )

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)
    

class EquiResBlock(torch.nn.Module):
    def __init__(self, group: gspaces.GSpace2D, in_channels: int, out_channels: int, cond_features: int, N=4) -> None:
        super().__init__()
        
        self.group = group
        
        self.conv1 = nn.R2Conv(
            in_type=nn.FieldType(self.group, in_channels*[self.group.trivial_repr]),
            out_type=nn.FieldType(self.group, out_channels*[self.group.trivial_repr]),
            kernel_size=3,
            stride=1,
            padding=1,
            initialize=True,
        ),
        self.group_norm1 = EquiAdaGroupNorm(self.group, out_channels, cond_features),
        self.relu1 = nn.ReLU(nn.FieldType(self.group, out_channels*[self.group.trivial_repr])),
        self.conv2 = nn.R2Conv(
            in_type=nn.FieldType(self.group, out_channels*[self.group.trivial_repr]),
            out_type=nn.FieldType(self.group, out_channels*[self.group.trivial_repr]),
            kernel_size=3,
            stride=1,
            padding=1,
            initialize=True,
        ),
        self.group_norm2 = EquiAdaGroupNorm(self.group, out_channels, cond_features),
        self.relu2 = nn.ReLU(nn.FieldType(self.group, out_channels*[self.group.trivial_repr])),
        self.proj = EquiConv1x1(
            in_type=nn.FieldType(self.group, in_channels*[self.group.trivial_repr]),
            out_type=nn.FieldType(self.group, out_channels*[self.group.trivial_repr])
        ) if in_channels != out_channels else nn.IdentityModule(nn.FieldType(self.group, in_channels*[self.group.trivial_repr]))

    def forward(self, x: nn.GeometricTensor, cond: Tensor) -> nn.GeometricTensor:
        # trivial in, trivial out
        r = self.proj(x)
        x = self.relu1(self.group_norm1(self.conv1(x), cond))
        x = self.relu2(self.group_norm2(self.conv2(x), cond))
        x = x + r
        return x
    

class EquiResBlocks(torch.nn.Module):
    def __init__(
        self,
        group,
        list_in_channels: List[int],
        list_out_channels: List[int],
        cond_features: int,
        N=4,
    ) -> None:
        super().__init__()
        assert len(list_in_channels) == len(list_out_channels)
        
        self.resblocks = torch.nn.ModuleList()
        for i in range(len(list_in_channels)):
            self.resblocks.append(
                EquiResBlock(
                    group=group,
                    in_channels=list_in_channels[i],
                    out_channels=list_out_channels[i],
                    cond_features=cond_features,
                    N=N
                )
            )

    def forward(self, x: nn.GeometricTensor, cond: nn.GeometricTensor) -> nn.GeometricTensor:
        # trivial in, trivial out
        outputs = []
        for i, resblock in enumerate(self.resblocks):
            x = resblock(x, cond)
            outputs.append(x)
        return x #, outputs
    

class EquiActionEmbedding(torch.nn.Module):
    """
    Equivariant action embedding, embeds an action from 1D to 2D, such that k*pi/2 rotations on the 1D-input are equivelant to rotations on the 2D-output
    """
    def __init__(self, group: gspaces.GSpace2D, in_channels, out_channels):
        super().__init__()
        self.group = group
        self.out_channels = out_channels
        # self.C, self.H, self.W = out_shape
        
        self.conv = nn.R2Conv(
            in_type=nn.FieldType(self.group, in_channels * [self.group.trivial_repr]),
            out_type=nn.FieldType(self.group, out_channels * [self.group.trivial_repr]),
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, action: torch.Tensor, out_size) -> torch.Tensor:
        """
        action: (B, 2),
        output: (B, C, H, W)
        """
        action = F.interpolate(action, size=out_size, mode='bilinear', align_corners=False)
        action = nn.GeometricTensor(action, nn.FieldType(self.group, action.shape[1] * [self.group.trivial_repr]))
        return self.conv(action).tensor
        
        

class EquiConditionalUNet2D(torch.nn.Module):
    def __init__(
        self,
        group: gspaces.GSpace2D,
        cond_features: int,
        depths: List[int],
        channels: List[int],
        action_in_ch: int,
        action_embedding_channels: List[int],
        N=4,
    ) -> None:
        super().__init__()
        assert len(depths)==len(channels)==len(action_embedding_channels), \
            "channels, depths, attn_depths, and action_embedding_channels must have same length"

        self.group = group
        self.num_levels = len(channels)
        
        # down action convs
        self.action_embeddings = torch.nn.ModuleList([
            EquiActionEmbedding(
                self.group,
                action_in_ch,
                action_embedding_channels[i]
            )
            for i in range(self.num_levels)
        ])
        # mid-level conv
        self.mid_action_embeddings = EquiActionEmbedding(
            self.group,
            action_in_ch,
            action_embedding_channels[-1]
        )

        # build UNet blocks
        self.d_blocks = torch.nn.ModuleList()
        self.u_blocks = torch.nn.ModuleList()
        for i in range(self.num_levels):
            in_ch = channels[max(0, i-1)] + action_embedding_channels[i]
            out_ch = channels[i]
            self.d_blocks.append(EquiResBlocks(
                group=self.group,
                list_in_channels=[in_ch]*depths[i],
                list_out_channels=[out_ch]*depths[i],
                cond_features=cond_features,
                N=N
            ))
            prev_ch = channels[max(0, i-1)]
            up_in = out_ch + prev_ch + action_embedding_channels[i]
            self.u_blocks.append(EquiResBlocks(
                group=self.group,
                list_in_channels=[up_in]*depths[i],
                list_out_channels=[prev_ch]*depths[i],
                cond_features=cond_features,
                N=N
            ))
        self.mid_blocks = EquiResBlocks(
            group=self.group,
            list_in_channels=[channels[-1] + action_embedding_channels[-1]]*2,
            list_out_channels=[channels[-1]]*2,
            cond_features=cond_features,
            N=N
        )
        self.downsamples = torch.nn.ModuleList([torch.nn.Identity()] + [EquiDownsample(self.group, channels[i]) for i in range(self.num_levels-1)])
        self.upsamples = torch.nn.ModuleList([torch.nn.Identity()] + [EquiUpsample(self.group, channels[-1-i]) for i in range(self.num_levels-1)])

    def forward(self, x: Tensor, cond: Tensor, action: Tensor) -> Tensor:
        B, _, h, w = x.shape
        n = self.num_levels - 1
        pad_h = math.ceil(h/2**n)*2**n - h
        pad_w = math.ceil(w/2**n)*2**n - w
        x = F.pad(x, (0, pad_w, 0, pad_h))
        
        # action = torch.tensor([action[0]])
        x = nn.GeometricTensor(x, nn.FieldType(self.group, x.shape[1] * [self.group.trivial_repr]))

        skips = []
        # Down path
        for i, (down, block) in enumerate(zip(self.downsamples, self.d_blocks)):
            x = down(x)
            act_embds = self.action_embeddings[i](action, (x.shape[2], x.shape[3]))
            x = torch.cat([x.tensor, act_embds], dim=1)
            x = nn.GeometricTensor(x, nn.FieldType(self.group, x.shape[1] * [self.group.trivial_repr]))
            x = block(x, cond)
            skips.append(x.tensor)

        # Middle
        act_embeds = self.mid_action_embeddings(action, (x.shape[2], x.shape[3]))
        x = torch.cat([x.tensor, act_embeds], dim=1)
        x = nn.GeometricTensor(x, nn.FieldType(self.group, x.shape[1] * [self.group.trivial_repr]))
        x = self.mid_blocks(x, cond)

        # Up path
        for j, (up, block) in enumerate(zip(self.upsamples, self.u_blocks)):
            x = up(x)
            skip = skips[-1-j]
            act_embeds = self.action_embeddings[self.num_levels-1-j](action, (x.size(2), x.size(3)))
            x = torch.cat([x.tensor, skip, act_embeds], dim=1)
            x = nn.GeometricTensor(x, nn.FieldType(self.group, x.shape[1] * [self.group.trivial_repr]))
            x = block(x, cond)
            
        return x, None, None
