import torch
# from piq.ms_ssim import multi_scale_ssim
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
    def __init__(self,
                 obs_channels=3,
                 lats_channels=1,
                 N=8,
                 initialize=True,
                 NUM_CHANNEL_1=32,
                 NUM_CHANNEL_2=64,
                 l2_loss_weight=0.00001,
                 recursive_steps=1,
                 recursive_weight=0.5,
                 latent_noise_std=None,
                 latent_norm_regularization_r=None,
                 latent_norm_regularization_weight=None,
                 ):
        super().__init__()
        self.encoder = EquivResEnc96to24(obs_channels, lats_channels, initialize, N, NUM_CHANNEL_1, NUM_CHANNEL_2)
        self.decoder = EquivResDec24to96(lats_channels, obs_channels, initialize, N, NUM_CHANNEL_1, NUM_CHANNEL_2)
        self.normalizer = LinearNormalizer()

        self.obs_channels = obs_channels
        self.lats_channels = lats_channels
        
        self.l2_loss_weight = l2_loss_weight
        self.recursive_steps = recursive_steps
        self.recursive_weight = recursive_weight
        self.latent_noise_std = latent_noise_std
        self.latent_norm_regularization_r = latent_norm_regularization_r
        self.latent_norm_regularization_weight = latent_norm_regularization_weight
    
    def encode(self, obs):
        if isinstance(obs, nn.GeometricTensor):
            obs = obs.tensor
        out = self.encoder(obs)
        return out
        
    def decode(self, lats):
        if isinstance(lats, nn.GeometricTensor):
            lats = lats.tensor
        out = self.decoder(lats)
        return out
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]):        
        nobs = self.normalizer.normalize(batch['obs'])
        obs = nobs['image'].squeeze(1)
        latent = self.encode(obs)
        flattened_latent = latent.view(latent.size(0), -1)
        dimension = flattened_latent.size(1)

        if self.latent_noise_std is not None:
            noise = torch.randn_like(latent) * self.latent_noise_std * self.latent_norm_regularization_r
            latent = latent + noise
        reconstructions = self.decode(latent)
        
        loss = torch.nn.functional.mse_loss(obs, reconstructions, reduction='mean') \
            # + self.l2_loss_weight * torch.nn.functional.mse_loss(latent, torch.zeros_like(latent), reduction='mean')
        
        if self.latent_norm_regularization_r is not None and self.latent_norm_regularization_weight is not None:
            latent_norm_loss = torch.mean((torch.sum(flattened_latent ** 2, dim=1) - self.latent_norm_regularization_r * dimension)**2)
            loss += self.latent_norm_regularization_weight * latent_norm_loss
        
        return loss
        
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
        
        
class Autoencoder(ModuleAttrMixin):
    def __init__(self,
                 obs_channels=3,
                 lats_channels=1,
                 NUM_CHANNEL_1=32,
                 NUM_CHANNEL_2=64,
                 l2_loss_weight=0.05,
                 recursive_steps=1,
                 recursive_weight=0.5,
                 ):
        super().__init__()
        self.encoder = ResEnc96to24(obs_channels, lats_channels, NUM_CHANNEL_1, NUM_CHANNEL_2)
        self.decoder = ResDec24to96(lats_channels, obs_channels, NUM_CHANNEL_1, NUM_CHANNEL_2)
        self.normalizer = LinearNormalizer()

        self.obs_channels = obs_channels
        self.lats_channels = lats_channels
        self.l2_loss_weight = l2_loss_weight
        self.recursive_steps = recursive_steps
        self.recursive_weight = recursive_weight
    
    def encode(self, obs):
        return self.encoder(obs)
        
    def decode(self, lats):
        return self.decoder(lats)
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]):
        nobs = self.normalizer.normalize(batch['obs'])
        obs = nobs['image'].squeeze(1)
        latent = self.encode(obs)
        reconstructions = self.decode(latent)
        
        loss = torch.nn.functional.mse_loss(obs, reconstructions, reduction='mean') \
            + self.l2_loss_weight * torch.nn.functional.mse_loss(latent, torch.zeros_like(latent), reduction='mean')
        
        return loss
    
    def compute_finetune_loss(self, batch: Dict[str, torch.Tensor]):
        nobs = self.normalizer.normalize(batch['obs'])
        obs = nobs['image'].squeeze(1)
        latent = self.encode(obs)
        reconstructions = self.decode(latent)

        loss = torch.nn.functional.mse_loss(obs, reconstructions, reduction='mean') \
            + self.l2_loss_weight * torch.nn.functional.mse_loss(latent, torch.zeros_like(latent), reduction='mean')

        # Compute the recursive loss
        for i in range(self.recursive_steps):
            latent = self.encode(reconstructions)
            reconstructions = self.decode(latent)
            loss += self.recursive_weight ** (i + 1) * torch.nn.functional.mse_loss(obs, reconstructions, reduction='mean') \
                + self.l2_loss_weight * torch.nn.functional.mse_loss(latent, torch.zeros_like(latent), reduction='mean')
        
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