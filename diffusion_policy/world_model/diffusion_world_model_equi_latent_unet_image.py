from typing import Dict, Optional
import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from einops import rearrange
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.world_model.base_world_model import BaseWorldModel
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb
from diffusion_policy.model.equi.equi_conditional_unet2d import ConditionalUNet2D, FourierFeatures, Conv3x3, GroupNorm


GN_GROUP_SIZE = 32
GN_EPS = 1e-5
ATTN_HEAD_DIM = 8



class DiffusionWorldModelImageLatentUnet(BaseWorldModel):
    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DDPMScheduler,
        auto_encoder: ModuleAttrMixin,
        pretrained_auto_encoder_path: str,
        n_obs_steps: int,
        n_future_steps: int = 1,
        num_inference_steps: Optional[int] = None,
        cond_channels: int = 256,
        action_map_channels: int = 32,
        depths = [2,2,2,2],
        channels = [64,64,64,64],
        attn_depths = [0,0,0,0],
        **kwargs
    ):
        super().__init__(shape_meta, **kwargs)
        # Load and freeze AE
        self.auto_encoder = auto_encoder
        ckpt = torch.load(pretrained_auto_encoder_path, map_location=self.device)
        self.auto_encoder.load_state_dict(ckpt['state_dicts']['model'], strict=True)
        self.auto_encoder.to(self.device).eval()
        for p in self.auto_encoder.parameters(): p.requires_grad = False

        lats_channels = self.auto_encoder.lats_channels
        self.n_obs_steps = n_obs_steps
        self.n_future_steps = n_future_steps

        # timestep encoder
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(cond_channels),
            nn.Linear(cond_channels, cond_channels * 4),
            nn.SiLU(),
            nn.Linear(cond_channels * 4, cond_channels)
        )

        # build input conv: latent_history + noise
        in_ch = (n_obs_steps * lats_channels) + (n_future_steps * lats_channels)
        self.conv_in = Conv3x3(in_ch, channels[0])

        # UNet (now will internally convolve and concat action)
        action_in_ch = shape_meta['action']['shape'][0]
        self.unet = ConditionalUNet2D(
            cond_channels=cond_channels,
            depths=depths,
            channels=channels,
            attn_depths=attn_depths,
            action_in_ch=action_in_ch,
            action_map_channels=action_map_channels
        )

        self.norm_out = GroupNorm(channels[0])
        self.conv_out = Conv3x3(channels[0], lats_channels)
        nn.init.zeros_(self.conv_out.weight)

        # scheduler
        self.noise_scheduler = noise_scheduler
        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        # normalizer
        self.normalizer = LinearNormalizer()

    def predict_future(
        self,
        obs_dict: Dict[str, torch.Tensor],
        action: torch.Tensor,
        last_latent: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        with torch.inference_mode():
            # encode history
            if last_latent is None:
                nobs = self.normalizer.normalize(obs_dict)
                imgs = nobs['image']  # (B, T, C, H, W)
                B, T, C, H, W = imgs.shape
                imgs = rearrange(imgs, 'B T C H W -> (B T) C H W')
                latent_history = self.auto_encoder.encode(imgs)
            else:
                latent_history = last_latent
                B, _, H, W = latent_history.shape

            # reshape history & init noise
            latent_history = rearrange(latent_history, '(B T) C H W -> B (T C) H W', B=B, T=self.n_obs_steps)
            noisy = torch.randn((B, self.n_future_steps * latent_history.size(1) // B, H, W), device=self.device)

            # diffusion loop
            self.noise_scheduler.set_timesteps(self.num_inference_steps, device=self.device)
            for t in self.noise_scheduler.timesteps:
                ts = t[None].to(self.device)
                cond = self.diffusion_step_encoder(ts)

                x_in = torch.cat((latent_history, noisy), dim=1)
                x = self.conv_in(x_in)
                x, _, _ = self.unet(x, cond, action)
                x = self.conv_out(F.silu(self.norm_out(x)))
                noisy = self.noise_scheduler.step(x, t, noisy).prev_sample

            out = rearrange(noisy, '(B T) C H W -> B T C H W', B=B, T=self.n_future_steps)
            decoded = self.auto_encoder.decode(rearrange(out, 'B T C H W -> (B T) C H W'))
            pred = rearrange(decoded, '(B T) C H W -> B T C H W', B=B, T=self.n_future_steps)
            return { 'predicted_future': self.normalizer['image'].unnormalize(pred),
                     'new_latent_history': out }

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        nobs = self.normalizer.normalize(batch['obs'])
        imgs = nobs['image']
        hist = imgs[:, :self.n_obs_steps]
        fut = imgs[:, self.n_obs_steps:self.n_obs_steps+self.n_future_steps]
        B, Tf, C, H, W = fut.shape

        # encode
        x0 = self.auto_encoder.encode(rearrange(fut, 'B T C H W -> (B T) C H W'))
        hist_lat = self.auto_encoder.encode(rearrange(hist, 'B T C H W -> (B T) C H W'))
        x0 = rearrange(x0, '(B T) C H W -> B (T C) H W', B=B, T=self.n_future_steps)
        hist_lat = rearrange(hist_lat, '(B T) C H W -> B (T C) H W', B=B, T=self.n_obs_steps)

        # noise
        t = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        noisy = self.noise_scheduler.add_noise(x0, noise, t)

        # forward
        cond = self.diffusion_step_encoder(t)
        x_in = torch.cat((hist_lat, noisy), dim=1)
        x = self.conv_in(x_in)
        x, _, _ = self.unet(x, cond, batch['action'])
        x = self.conv_out(F.silu(self.norm_out(x)))

        target = noise if self.noise_scheduler.config.prediction_type=='epsilon' else x0
        return F.mse_loss(x, target, reduction='mean')