import torch
from piq.ms_ssim import multi_scale_ssim
from torchvision import models as vision_models
from escnn import gspaces, nn
from escnn.group import CyclicGroup
from einops import rearrange
from typing import Dict

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.model.equi.equi_autoencoder import EquivResEnc96to24, EquivResDec24to96, ResEnc96to24, ResDec24to96
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.model.common.normalizer import LinearNormalizer




class EquivariantAutoencoder(ModuleAttrMixin):
    def __init__(
        self,
        obs_channels=3,
        lats_channels=3,
        N=8,
        initialize=True,
    ):
        super().__init__()
        self.encoder = EquivResEnc96to24(obs_channels, lats_channels, initialize, N)
        self.decoder = EquivResDec24to96(lats_channels, obs_channels, initialize, N)
        self.normalizer = LinearNormalizer()
    
    def encode(self, obs):
        if isinstance(obs, nn.GeometricTensor):
            obs = obs.tensor
        return self.encoder(obs)
        
    def decode(self, lats):
        if isinstance(lats, nn.GeometricTensor):
            lats = lats.tensor
        return self.decoder(lats)
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]):
        nobs = self.normalizer.normalize(batch['obs'])
        obs = nobs['image'].squeeze(1)
        reconstructions = self.decoder(self.encoder(obs))
        
        # loss = multi_scale_ssim(
        #     obs, reconstructions,
        #     data_range=1.0, 
        #     reduction='mean', 
        #     kernel_size=3,
        #     scale_weights=torch.tensor([0.5, 0.3, 0.2], device=obs.device, dtype=torch.float)
        # )
        
        loss = torch.nn.functional.mse_loss(obs, reconstructions, reduce='mean')
        
        return loss
        
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
        
        
class Autoencoder(ModuleAttrMixin):
    def __init__(
        self,
        obs_channels=3,
        lats_channels=3,
    ):
        super().__init__()
        self.encoder = ResEnc96to24(obs_channels, lats_channels)
        self.decoder = ResDec24to96(lats_channels, obs_channels)
        self.normalizer = LinearNormalizer()
    
    def encode(self, obs):
        return self.encoder(obs)
        
    def decode(self, lats):
        return self.decoder(lats)
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]):
        nobs = self.normalizer.normalize(batch['obs'])
        obs = nobs['image'].squeeze(1)
        reconstructions = self.decoder(self.encoder(obs))
        
        loss = torch.nn.functional.mse_loss(obs, reconstructions, reduce='mean')
        
        return loss
        
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
        
        


# class EquivariantObsEnc(ModuleAttrMixin):
#     def __init__(
#         self,
#         obs_shape=(3, 96, 96),
#         crop_shape=(96, 96),
#         lats_channel=64,
#         N=8,
#         initialize=True,
#     ):
#         super().__init__()
#         obs_channel = obs_shape[0]
#         self.enc_obs = EquivariantResEncoder(obs_channel, lats_channel, initialize, N)

#         self.is_crop = True if obs_shape[1:] != crop_shape else False
#         if self.is_crop:
#             self.crop_randomizer = dmvc.CropRandomizer(
#                 input_shape=obs_shape,
#                 crop_height=crop_shape[0],
#                 crop_width=crop_shape[1],
#             )

        
#     def forward(self, obs):
#         if isinstance(obs, nn.GeometricTensor):
#             obs = obs.tensor
#         batch_size = obs.shape[0]
#         obs = rearrange(obs, "b t c h w -> (b t) c h w")
#         if self.is_crop:
#             obs = self.crop_randomizer(obs)

#         enc_out = self.enc_obs(obs).tensor
#         return rearrange(enc_out, "(b t) c h w -> b t c h w", b=batch_size)


# class EquivariantObsDec(ModuleAttrMixin):
#     def __init__(
#         self,
#         lats_channel=64,
#         obs_channel=3,
#         N=8,
#         initialize=True,
#     ):
#         super().__init__()
#         self.decoder = EquivariantResDecoder(lats_channel, obs_channel, initialize, N)

        
#     def forward(self, x):
#         if isinstance(x, nn.GeometricTensor):
#             x = x.tensor
#         batch_size = x.shape[0]
#         x = rearrange(x, "b t c h w -> (b t) c h w")
#         dec_out = self.decoder(x).tensor
#         return rearrange(dec_out, "(b t) c h w -> b t c h w", b=batch_size)