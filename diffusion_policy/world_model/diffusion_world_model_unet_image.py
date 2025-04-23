from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.world_model.base_world_model import BaseWorldModel
from diffusion_policy.common.pytorch_util import dict_apply

# Example diffusion model architecture
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.conditional_unet2d import ConditionalUNet2D, FourierFeatures, Conv3x3, GroupNorm
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb

class DiffusionWorldModelImageUnet(BaseWorldModel):
    """
    Diffusion-based image world model. Predicts future images conditioned on
    (past images, action sequence).

    This is analogous to DiffusionUnetImagePolicy, but for modeling the environment
    rather than producing actions.
    """

    def __init__(self,
                 shape_meta: dict,
                 noise_scheduler: DDPMScheduler,
                 n_obs_steps: int,
                 n_future_steps : int = 1,
                 num_inference_steps=None,
                 cond_channels=256,
                 depths = [2,2,2,2],
                 channels= [64,64,64,64],
                 attn_depths= [0,0,0,0],
                 **kwargs):
        """
        shape_meta:
            includes info about 'obs_image' shape: e.g. (3, H, W)
            includes info about 'action' shape: e.g. (action_dim,)
        noise_scheduler:
            A diffusion scheduler instance (DDPMScheduler, DDIM, etc.)
        n_obs_steps:
            How many past image frames are used as context
        n_future_steps:
            How many future frames we want to predict
        """
        super().__init__(shape_meta, **kwargs)

        # get feature dim
        self.n_future_steps = n_future_steps

        # parse action dimension
        action_dim = shape_meta['action']['shape'][0]

        # get img channels
        img_channels = shape_meta['obs']['image']['shape'][0]

        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(cond_channels),
            nn.Linear(cond_channels, cond_channels * 4),
            nn.Mish(),
            nn.Linear(cond_channels * 4, cond_channels),
        )
        self.act_proj = nn.Sequential(
            nn.Linear(action_dim * (n_future_steps), cond_channels),
            nn.SiLU(),
            nn.Linear(cond_channels, cond_channels),
        )
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_channels, cond_channels),
            nn.SiLU(),
            nn.Linear(cond_channels, cond_channels),
        )
        self.conv_in = Conv3x3((n_obs_steps + n_future_steps) * img_channels, channels[0])

        self.unet = ConditionalUNet2D(
            cond_channels = cond_channels,
            depths = depths, 
            channels = channels, 
            attn_depths = attn_depths
        )

        self.norm_out = GroupNorm(channels[0])
        self.conv_out = Conv3x3(channels[0], n_future_steps * img_channels)
        nn.init.zeros_(self.conv_out.weight)

        trainable_params = sum(p.numel() for p in self.unet.parameters() if p.requires_grad)
        print(f'Trainable parameters in unet: {trainable_params}')

        # self.obs_encoder = obs_encoder

        # trainable_params = sum(p.numel() for p in self.obs_encoder.parameters() if p.requires_grad)
        # print(f'Trainable parameters in resnet: {trainable_params}')

        self.noise_scheduler = noise_scheduler
        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        # normalizer for images if needed
        self.normalizer = LinearNormalizer()
        self.n_obs_steps = n_obs_steps

    def predict_future(self, obs_dict: Dict[str, torch.Tensor], action) -> Dict[str, torch.Tensor]:
        """
        Inference method:
          - 'obs_dict' should contain "obs" => (B, n_obs_steps, C, H, W)
            and "action" => (B, n_future_steps, action_dim)
          - returns a dict with "predicted_future" => (B, n_future_steps, C, H, W)
        """
        with torch.inference_mode():
            nobs = self.normalizer.normalize(obs_dict)
            nactions = self.normalizer['action'].normalize(action)
            history_imgs = nobs['image']  # shape (B, n_obs_steps, C, H, W)
            action_seq = nactions  # shape (B, n_future_steps, Da)
            device = history_imgs.device
            dtype = history_imgs.dtype

            B, To, C, H, W = history_imgs.shape
            assert To == self.n_obs_steps, f"Expected n_obs_steps={self.n_obs_steps}, got {To}."

            # Start from random noise
            # If you want the model to be deterministic, consider using zero noise
            noisy_image = torch.zeros((B, self.n_future_steps * C, H, W), device=device, dtype=dtype)

            # set up timesteps
            self.noise_scheduler.set_timesteps(self.num_inference_steps, device=device)

            # reshape
            history_imgs = history_imgs.view(B, self.n_obs_steps * C, H, W)
            action_seq = action_seq.view(B, -1)

            for t in self.noise_scheduler.timesteps:
                # forward the model
                if torch.is_tensor(t) and len(t.shape) == 0:
                    timesteps = t[None].to(device)
                timesteps = timesteps.expand(noisy_image.shape[0])

                cond = self.cond_proj(self.diffusion_step_encoder(timesteps) + self.act_proj(action_seq))
                x = self.conv_in(torch.cat((history_imgs, noisy_image), dim=1))
                x, _, _ = self.unet(x, cond)
                x = self.conv_out(F.silu(self.norm_out(x)))
                # diffusion update
                noisy_image = self.noise_scheduler.step(
                    x, t, noisy_image
                ).prev_sample

            predicted = noisy_image.view(B, self.n_future_steps, C, H, W)
            unnormalized_predicted = self.normalizer['image'].unnormalize(predicted)

        return {
            "predicted_future": unnormalized_predicted
        }

    def set_normalizer(self, normalizer: LinearNormalizer):
        """
        If you want to load a normalizer state or override it for training,
        e.g. self.normalizer.load_state_dict(normalizer.state_dict()).
        """
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Standard diffusion objective:
          1) flatten GT future frames
          2) sample random timesteps
          3) add noise
          4) model predicts noise (or sample)
          5) MSE loss
        """
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])

        imgs = nobs['image']
        history_imgs = imgs[:, :self.n_obs_steps]
        future_imgs = imgs[:, self.n_obs_steps:self.n_obs_steps+self.n_future_steps]
        action_seq = nactions[:, self.n_obs_steps-1:self.n_obs_steps+self.n_future_steps-1]

        B, T, C, H, W = future_imgs.shape
        assert T == self.n_future_steps, f"Expected n_future_steps={self.n_future_steps}, got {T}."

        # flatten images
        x_0 = future_imgs.view(B, self.n_future_steps * C, H, W)
        history_imgs = history_imgs.view(B, self.n_obs_steps * C, H, W)

        # sample random timesteps for each sample
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=x_0.device
        ).long()

        # noise
        noise = torch.randn_like(x_0, device=x_0.device)

        # forward diffusion
        noisy_x = self.noise_scheduler.add_noise(x_0, noise, timesteps)

        # condition
        action_seq = action_seq.reshape(B, -1)
        cond = self.cond_proj(self.diffusion_step_encoder(timesteps) + self.act_proj(action_seq))
        x = self.conv_in(torch.cat((history_imgs, noisy_x), dim=1))
        x, _, _ = self.unet(x, cond)
        x = self.conv_out(F.silu(self.norm_out(x)))

        # compute target
        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = x_0
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(x, target, reduction='mean')
        return loss
