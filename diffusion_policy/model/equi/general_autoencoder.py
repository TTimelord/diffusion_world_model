import torch
from typing import Dict

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.model.equi.equi_autoencoder import ResEncoder, ResDecoder

class Autoencoder(ModuleAttrMixin):
    def __init__(self,
                 obs_channels=3,
                 lats_channels=1,
                 encoder_channels=[8, 16, 32, 16, 8],
                 decoder_channels=[8, 16, 32, 16, 8],
                 l2_loss_weight=0.00001,
                 recursive_steps=1,
                 recursive_weight=0.5,
                 ):
        super().__init__()
        self.encoder = ResEncoder(obs_channels, lats_channels, encoder_channels)
        self.decoder = ResDecoder(lats_channels, obs_channels, decoder_channels)
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
