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
                 latent_noise_std=None,
                 latent_norm_regularization_r=None,
                 latent_norm_regularization_weight=None,
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
        self.latent_noise_std = latent_noise_std
        self.latent_norm_regularization_r = latent_norm_regularization_r
        self.latent_norm_regularization_weight = latent_norm_regularization_weight
    
    def encode(self, obs):
        return self.encoder(obs)
        
    def decode(self, lats):
        return self.decoder(lats)
    
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
            + self.l2_loss_weight * torch.nn.functional.mse_loss(latent, torch.zeros_like(latent), reduction='mean')
        
        if self.latent_norm_regularization_r is not None and self.latent_norm_regularization_weight is not None:
            latent_norm_loss = torch.mean((torch.sum(flattened_latent ** 2, dim=1) - self.latent_norm_regularization_r * dimension)**2)
            loss += self.latent_norm_regularization_weight * latent_norm_loss
        
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


class VAE(ModuleAttrMixin):
    def __init__(self,
                 obs_channels=3,
                 lats_channels=1,
                 encoder_channels=[8,16,32,16,8],
                 decoder_channels=[8,16,32,16,8],
                 recursive_steps=1,
                 recursive_weight=0.5,
                 beta=1.0,
                 fixed_logvar = None):
        super().__init__()
        # Encoder now outputs 2 * lats_channels so we can split into mu & logvar
        if fixed_logvar is None:
            self.encoder = ResEncoder(obs_channels, 2 * lats_channels, encoder_channels)
        else:
            self.encoder = ResEncoder(obs_channels, lats_channels, encoder_channels)
        self.decoder = ResDecoder(lats_channels, obs_channels, decoder_channels)
        self.normalizer = LinearNormalizer()
        self.lats_channels = lats_channels
        self.recursive_steps = recursive_steps
        self.recursive_weight = recursive_weight
        self.beta = beta
        self.fixed_logvar = fixed_logvar

    def encode(self, obs):
        if self.fixed_logvar is None:
            stats = self.encoder(obs)
            mu, logvar = torch.chunk(stats, 2, dim=1)
        else:
            mu = self.encoder(obs)
            logvar = torch.full_like(mu, self.fixed_logvar)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def compute_kl(self, mu, logvar):
        # KL per example, then mean over batch
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        return kl.sum(dim=[1,2,3]).mean()

    def compute_loss(self, batch: Dict[str, torch.Tensor]):
        nobs = self.normalizer.normalize(batch['obs'])
        obs = nobs['image'].squeeze(1)

        mu, logvar = self.encode(obs)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)

        recon_loss = torch.nn.functional.mse_loss(obs, recon, reduction='mean')
        kl_loss    = self.compute_kl(mu, logvar)

        return recon_loss + self.beta * kl_loss

    def compute_finetune_loss(self, batch: Dict[str, torch.Tensor]):
        nobs = self.normalizer.normalize(batch['obs'])
        obs = nobs['image'].squeeze(1)

        mu, logvar = self.encode(obs)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)

        loss = torch.nn.functional.mse_loss(obs, recon, reduction='mean') \
             + self.compute_kl(mu, logvar)

        for i in range(self.recursive_steps):
            mu, logvar = self.encode(recon)
            z = self.reparameterize(mu, logvar)
            recon = self.decode(z)
            loss += (self.recursive_weight**(i+1)) * torch.nn.functional.mse_loss(obs, recon, reduction='mean')
            loss += self.compute_kl(mu, logvar)

        return loss

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
