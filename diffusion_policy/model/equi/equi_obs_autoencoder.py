import torch
from torchvision import models as vision_models
from escnn import gspaces, nn
from escnn.group import CyclicGroup
from einops import rearrange
from robomimic.models.base_nets import SpatialSoftmax
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.model.equi.equi_encoder import EquivariantResEncoder, EquivariantResDecoder
# from diffusion_policy.model.vision.voxel_crop_randomizer import VoxelCropRandomizer
from diffusion_policy.model.common.rotation_transformer import RotationTransformer

class Identity(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
    
class InHandEncoder(torch.nn.Module):
    def __init__(self, out_size):
        super().__init__()
        net = vision_models.resnet18(norm_layer=Identity)
        self.resnet = torch.nn.Sequential(*(list(net.children())[:-2]))
        self.spatial_softmax = SpatialSoftmax([512, 3, 3], num_kp=out_size//2)

    def forward(self, ih):
        batch_size = ih.shape[0]
        return self.spatial_softmax(self.resnet(ih)).reshape(batch_size, -1)


class EquivariantObsEnc(ModuleAttrMixin):
    def __init__(
        self,
        obs_shape=(3, 96, 96),
        crop_shape=(96, 96),
        out_channel=128,
        N=8,
        initialize=True,
    ):
        super().__init__()
        obs_channel = obs_shape[0]
        self.out_channel = out_channel
        self.N = N
        self.group = gspaces.no_base_space(CyclicGroup(self.N))
        self.enc_obs = EquivariantResEncoder(obs_channel, self.out_channel, initialize)
        
        self.quaternion_to_sixd = RotationTransformer('quaternion', 'rotation_6d')

        self.gTgc = torch.Tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

        self.crop_randomizer = dmvc.CropRandomizer(
            input_shape=obs_shape,
            crop_height=crop_shape[0],
            crop_width=crop_shape[1],
        )

        
    def forward(self, nobs):
        obs = nobs["agentview_image"]
        # B, T, C, H, W
        batch_size = obs.shape[0]
        t = obs.shape[1]
        obs = rearrange(obs, "b t c h w -> (b t) c h w")
        obs = self.crop_randomizer(obs)

        enc_out = self.enc_obs(obs).tensor  # b d
        return rearrange(enc_out, "(b t) c h w -> b t c h w", b=batch_size)


class EquivariantObsDec(ModuleAttrMixin):
    def __init__(
        self,
        obs_shape=(3, 96, 96),
        crop_shape=(96, 96),
        out_channel=128,
        N=8,
        initialize=True,
    ):
        super().__init__()
        obs_channel = obs_shape[0]
        self.out_channel = out_channel
        self.N = N
        self.group = gspaces.no_base_space(CyclicGroup(self.N))
        self.enc_obs = EquivariantResEncoder(obs_channel, self.out_channel, initialize)
        
        self.quaternion_to_sixd = RotationTransformer('quaternion', 'rotation_6d')

        self.gTgc = torch.Tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

        self.crop_randomizer = dmvc.CropRandomizer(
            input_shape=obs_shape,
            crop_height=crop_shape[0],
            crop_width=crop_shape[1],
        )

        
    def forward(self, nobs):
        obs = nobs["agentview_image"]
        # B, T, C, H, W
        batch_size = obs.shape[0]
        t = obs.shape[1]
        obs = rearrange(obs, "b t c h w -> (b t) c h w")
        obs = self.crop_randomizer(obs)

        enc_out = self.enc_obs(obs).tensor  # b d
        return rearrange(enc_out, "(b t) c h w -> b t c h w", b=batch_size)