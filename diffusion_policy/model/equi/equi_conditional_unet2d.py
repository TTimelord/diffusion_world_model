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

GN_GROUP_SIZE = 16
GN_EPS = 1e-1
ATTN_HEAD_DIM = 8

# Convs

Conv1x1 = partial(torch.nn.Conv2d, kernel_size=1, stride=1, padding=0)
Conv3x3 = partial(torch.nn.Conv2d, kernel_size=3, stride=1, padding=1)


EquiConv1x1 = partial(nn.R2Conv, kernel_size=1, stride=1, padding=0, initialize=True)
EquiConv3x3 = partial(nn.R2Conv, kernel_size=3, stride=1, padding=1, initialize=True)

def action1Dto2D(action: torch.Tensor):
    x = action[:, 0]
    y = action[:, 1]
    action2d = torch.stack([
        torch.stack([x, -y], dim=-1),
        torch.stack([y, -x], dim=-1)
    ], dim=1).unsqueeze(1)
    
    return action2d


class EquiAdaGroupNorm(torch.nn.Module):
    """
    Equivariant AdaGroupNorm, equivariant to x, invariant to cond;
    AdaGroupNorm is equivariant itself
    """
    def __init__(self, group: gspaces.GSpace2D, sample_channels: int, cond_channels: int, repr, N=4) -> None:
        super().__init__()
        self.group = group
        self.repr = repr
        self.sample_channels = sample_channels
        self.num_groups = max(1, sample_channels // GN_GROUP_SIZE)
        n = N if repr == self.group.regular_repr else 1
        self.linear = torch.nn.Linear(cond_channels, sample_channels * 2 * n)
        self.field_norm = nn.FieldNorm(nn.FieldType(self.group, self.sample_channels * [self.repr]), eps=GN_EPS)
        self.batch_norm = nn.InnerBatchNorm(nn.FieldType(self.group, self.sample_channels * [self.repr]))

    def forward(self, x: nn.GeometricTensor, cond: Tensor) -> nn.GeometricTensor:
        # trivial in, trivial out
        # assert x.shape[1] == self.sample_channels
        x = self.batch_norm(x)
        scale, shift = self.linear(cond)[:, :, None, None].chunk(2, dim=1)
        x = x.tensor * (1 + scale) + shift
        return nn.GeometricTensor(x, nn.FieldType(self.group, self.sample_channels * [self.repr]))


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
        self.upsampling = nn.R2Upsampling(nn.FieldType(self.group, channels*[self.group.trivial_repr]), 2)
        self.conv = nn.R2Conv(
            in_type=nn.FieldType(self.group, channels*[self.group.trivial_repr]),
            out_type=nn.FieldType(self.group, channels*[self.group.trivial_repr]),
            kernel_size=3,
            stride=1,
            padding=1,
            initialize=True
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.upsampling(x)
        return self.conv(x)
    

class EquiResBlock(torch.nn.Module):
    def __init__(self, group: gspaces.GSpace2D, in_channels: int, out_channels: int, cond_features: int, in_repr, out_repr, N=4) -> None:
        super().__init__()
        
        self.group = group
        
        self.conv1 = nn.R2Conv(
            in_type=nn.FieldType(self.group, in_channels*[in_repr]),
            out_type=nn.FieldType(self.group, out_channels*[self.group.regular_repr]),
            kernel_size=3,
            stride=1,
            padding=1,
            initialize=True,
        )
        self.group_norm1 = EquiAdaGroupNorm(self.group, out_channels, cond_features, self.group.regular_repr, N=N)
        self.relu1 = nn.ReLU(nn.FieldType(self.group, out_channels*[self.group.regular_repr]))
        self.conv2 = nn.R2Conv(
            in_type=nn.FieldType(self.group, out_channels*[self.group.regular_repr]),
            out_type=nn.FieldType(self.group, out_channels*[out_repr]),
            kernel_size=3,
            stride=1,
            padding=1,
            initialize=True,
        )
        self.group_norm2 = EquiAdaGroupNorm(self.group, out_channels, cond_features, out_repr, N=N)
        self.relu2 = nn.ReLU(nn.FieldType(self.group, out_channels*[out_repr]))
        self.proj = nn.R2Conv(
            in_type=nn.FieldType(self.group, in_channels*[in_repr]),
            out_type=nn.FieldType(self.group, out_channels*[out_repr]),
            kernel_size=1,
            stride=1, 
            padding=0,
            initialize=True
        ) if in_channels != out_channels or in_repr != out_repr else nn.IdentityModule(nn.FieldType(self.group, in_channels*[in_repr]))

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
        group: gspaces.GSpace2D,
        list_in_channels: List[int],
        list_out_channels: List[int],
        cond_features: int,
        N=4,
    ) -> None:
        super().__init__()
        assert len(list_in_channels) == len(list_out_channels)
        
        self.resblocks = torch.nn.ModuleList()
        if len(list_in_channels) == 1:
            self.resblocks.append(
                EquiResBlock(
                    group=group,
                    in_channels=list_in_channels[0],
                    out_channels=list_out_channels[0],
                    cond_features=cond_features,
                    in_repr=group.trivial_repr,
                    out_repr=group.trivial_repr,
                    N=N
                )
            )
            return
        self.resblocks.append(
            EquiResBlock(
                group=group,
                in_channels=list_in_channels[0],
                out_channels=list_out_channels[0],
                cond_features=cond_features,
                in_repr=group.trivial_repr,
                out_repr=group.regular_repr,
                N=N
            )
        )
        for i in range(1, len(list_in_channels)-1):
            self.resblocks.append(
                EquiResBlock(
                    group=group,
                    in_channels=list_in_channels[i],
                    out_channels=list_out_channels[i],
                    cond_features=cond_features,
                    in_repr=group.regular_repr,
                    out_repr=group.regular_repr,
                    N=N
                )
            )
        self.resblocks.append(
            EquiResBlock(
                group=group,
                in_channels=list_in_channels[-1],
                out_channels=list_out_channels[-1],
                cond_features=cond_features,
                in_repr=group.regular_repr,
                out_repr=group.trivial_repr,
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
    def __init__(self, group: gspaces.GSpace2D, out_channels):
        super().__init__()
        self.group = group
        self.out_channels = out_channels
        # self.C, self.H, self.W = out_shape
        
        self.conv = nn.R2Conv(
            in_type=nn.FieldType(self.group, 1 * [self.group.trivial_repr]),
            out_type=nn.FieldType(self.group, out_channels * [self.group.trivial_repr]),
            kernel_size=3,
            stride=1,
            padding=1,
            initialize=True
        )

    def forward(self, action: torch.Tensor, out_size) -> torch.Tensor:
        """
        action: (B, 1, 2, 2),
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
                action_embedding_channels[i]
            )
            for i in range(self.num_levels)
        ])
        # mid-level conv
        self.mid_action_embeddings = EquiActionEmbedding(
            self.group,
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
                list_in_channels=[in_ch] + [out_ch]*(depths[i]-1),
                list_out_channels=[out_ch]*depths[i],
                cond_features=cond_features,
                N=N
            ))
            prev_ch = channels[max(0, i-1)]
            up_in = out_ch + prev_ch + action_embedding_channels[i]
            self.u_blocks.append(EquiResBlocks(
                group=self.group,
                list_in_channels=[up_in] + [prev_ch]*(depths[i]-1),
                list_out_channels=[prev_ch]*depths[i],
                cond_features=cond_features,
                N=N
            ))
        self.mid_blocks = EquiResBlocks(
            group=self.group,
            list_in_channels=[channels[-1] + action_embedding_channels[-1], channels[-1]],
            list_out_channels=[channels[-1]]*2,
            cond_features=cond_features,
            N=N
        )
        self.downsamples = torch.nn.ModuleList([torch.nn.Identity()] + [EquiDownsample(self.group, channels[i]) for i in range(self.num_levels-1)])
        self.upsamples = torch.nn.ModuleList([torch.nn.Identity()] + [EquiUpsample(self.group, channels[-1-i]) for i in range(self.num_levels-1)])

    def forward(self, x: nn.GeometricTensor, cond: Tensor, action: Tensor) -> nn.GeometricTensor:
        B, _, h, w = x.shape
        n = self.num_levels - 1
        pad_h = math.ceil(h/2**n)*2**n - h
        pad_w = math.ceil(w/2**n)*2**n - w
        assert(pad_h == 0 and pad_w == 0)
        # x = F.pad(x, (0, pad_w, 0, pad_h))
        
        action = action1Dto2D(action)  # (B, 1, 2, 2)
        # x = nn.GeometricTensor(x, nn.FieldType(self.group, x.shape[1] * [self.group.trivial_repr]))

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
            act_embeds = self.action_embeddings[self.num_levels-1-j](action, (x.shape[2], x.shape[3]))
            x = torch.cat([x.tensor, skip, act_embeds], dim=1)
            x = nn.GeometricTensor(x, nn.FieldType(self.group, x.shape[1] * [self.group.trivial_repr]))
            x = block(x, cond)
        # x = x.tensor
        
        return x
