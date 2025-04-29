from typing import Dict
import copy
import torch
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
import escnn.nn as nn
from escnn import gspaces

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.world_model.base_world_model import BaseWorldModel
from diffusion_policy.common.pytorch_util import dict_apply

# Example diffusion model architecture
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.equi.equi_conditional_unet2d import EquiConditionalUNet2D
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb

class DiffusionWorldModelImageEquiUnet(BaseWorldModel):
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
                 action_embedding_channels=[8, 8, 8, 8],
                 l1_loss_weight=0.0,
                 N=4,
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
        
        self.group = gspaces.rot2dOnR2(N)

        # get feature dim
        self.n_future_steps = n_future_steps

        # parse action dimension
        action_dim = shape_meta['action']['shape'][0]

        # get img channels
        img_channels = shape_meta['obs']['image']['shape'][0]

        self.diffusion_step_encoder = torch.nn.Sequential(
            SinusoidalPosEmb(cond_channels),
            torch.nn.Linear(cond_channels, cond_channels * 4),
            torch.nn.Mish(),
            torch.nn.Linear(cond_channels * 4, cond_channels),
        )
        # self.act_proj = torch.nn.Sequential(
        #     torch.nn.Linear(action_dim, cond_channels),
        #     torch.nn.SiLU(),
        #     torch.nn.Linear(cond_channels, cond_channels),
        # )
        # self.cond_proj = torch.nn.Sequential(
        #     torch.nn.Linear(cond_channels, cond_channels),
        #     torch.nn.SiLU(),
        #     torch.nn.Linear(cond_channels, cond_channels),
        # )
        self.conv_in = nn.SequentialModule(
            nn.R2Conv(
                nn.FieldType(self.group, (n_obs_steps + 1) * img_channels * [self.group.trivial_repr]),
                nn.FieldType(self.group, channels[0] * [self.group.trivial_repr]),
                kernel_size=3,
                stride=1,
                padding=1,
                initialize=True
            ),
            # nn.InnerBatchNorm(nn.FieldType(self.group, channels[0] * [self.group.trivial_repr]))
        )
        # self.conv_in = Conv3x3((n_obs_steps + 1) * img_channels, channels[0])

        self.unet = EquiConditionalUNet2D(
            group=self.group,
            cond_features=cond_channels,
            depths=depths, 
            channels=channels, 
            action_embedding_channels=action_embedding_channels,
            N=N
        )

        # self.norm_out = GroupNorm(channels[0])
        self.conv_out = nn.SequentialModule(
            nn.R2Conv(
                nn.FieldType(self.group, channels[-1] * [self.group.trivial_repr]),
                nn.FieldType(self.group, img_channels * [self.group.trivial_repr]),
                kernel_size=3, 
                stride=1,
                padding=1,
                initialize=True,
            ),
            nn.PointwiseNonLinearity(nn.FieldType(self.group, img_channels * [self.group.trivial_repr]), 'p_tanh')
        )
        # self.conv_out = Conv3x3(channels[0], img_channels)
        # torch.nn.init.zeros_(self.conv_out.weight)

        trainable_params = sum(p.numel() for p in self.unet.parameters() if p.requires_grad)
        print(f'Trainable parameters in unet: {trainable_params}')

        # self.obs_encoder = obs_encoder

        # trainable_params = sum(p.numel() for p in self.obs_encoder.parameters() if p.requires_grad)
        # print(f'Trainable parameters in resnet: {trainable_params}')

        self.noise_scheduler = noise_scheduler
        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        ddim_scheduler_config = dict(noise_scheduler.config)
        ddim_scheduler_config.pop('variance_type', None)
        self.ddim_scheduler = DDIMScheduler(
            **ddim_scheduler_config,
        )

        # normalizer for images if needed
        self.normalizer = LinearNormalizer()
        self.n_obs_steps = n_obs_steps
        self.l1_loss_weight = l1_loss_weight

    def predict_future(self, obs_dict: Dict[str, torch.Tensor], action) -> Dict[str, torch.Tensor]:
        """
        Inference method:
          - 'obs_dict' should contain "obs" => (B, n_obs_steps, C, H, W)
            and "action" => (B, n_future_steps, action_dim)
          - returns a dict with "predicted_future" => (B, n_future_steps, C, H, W)
        """
        with torch.inference_mode():
            nobs = self.normalizer.normalize(obs_dict)
            # nactions = self.normalizer['action'].normalize(action)
            history_imgs = nobs['image']  # shape (B, n_obs_steps, C, H, W)
            action_seq = action  # shape (B, n_future_steps, Da)
            device = history_imgs.device
            dtype = history_imgs.dtype

            B, To, C, H, W = history_imgs.shape
            assert To == self.n_obs_steps, f"Expected n_obs_steps={self.n_obs_steps}, got {To}."

            # Start from random noise
            # If you want the model to be deterministic, consider using zero noise
            noisy_image = torch.zeros((B, C, H, W), device=device, dtype=dtype)

            # set up timesteps
            self.ddim_scheduler.set_timesteps(self.num_inference_steps, device=device)

            # reshape
            history_imgs = history_imgs.view(B, self.n_obs_steps * C, H, W)
            action_seq = action_seq[:, :1].view(B, -1)
            # action_seq = self.normalize_action(action_seq)

            for t in self.ddim_scheduler.timesteps:
                # forward the model
                if torch.is_tensor(t) and len(t.shape) == 0:
                    timesteps = t[None].to(device)
                timesteps = timesteps.expand(noisy_image.shape[0])

                cond = self.diffusion_step_encoder(timesteps)
                x = torch.cat((history_imgs, noisy_image), dim=1)
                x = nn.GeometricTensor(x, nn.FieldType(self.group, x.shape[1] * [self.group.trivial_repr]))
                x = self.conv_in(x)
                x = self.unet(x, cond, action_seq)
                x = self.conv_out(x).tensor
                # x = self.conv_out(F.silu(self.norm_out(x)))
                # diffusion update
                noisy_image = self.ddim_scheduler.step(
                    x, t, noisy_image
                ).prev_sample

            predicted = noisy_image.view(B, 1, C, H, W)
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
        nactions = batch['action']

        imgs = nobs['image']
        history_imgs = imgs[:, :self.n_obs_steps]

        # in the early stage only use 1 step prediction
        future_imgs = imgs[:, self.n_obs_steps:self.n_obs_steps+1]
        action_seq = nactions[:, self.n_obs_steps-1:self.n_obs_steps]

        B, T, C, H, W = future_imgs.shape
        # assert T == self.n_future_steps, f"Expected n_future_steps={self.n_future_steps}, got {T}."

        # flatten images
        x_0 = future_imgs.view(B, C, H, W)
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
        action_seq = self.normalize_action(action_seq)
        cond = self.diffusion_step_encoder(timesteps)
        x = torch.cat((history_imgs, noisy_x), dim=1)       
        x = nn.GeometricTensor(x, nn.FieldType(self.group, x.shape[1] * [self.group.trivial_repr]))
        x = self.conv_in(x)
        x = self.unet(x, cond, action_seq)
        x = self.conv_out(x).tensor

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

    def compute_autoregressive_loss(self, batch: Dict[str, torch.Tensor], depth=1) -> torch.Tensor:
        """
        Standard diffusion objective:
          1) flatten GT future frames
          2) sample random timesteps
          3) add noise
          4) model predicts noise (or sample)
          5) MSE loss
        """
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = batch['action']

        imgs = nobs['image']
        history_imgs = imgs[:, :self.n_obs_steps]
        future_imgs = imgs[:, self.n_obs_steps:self.n_obs_steps+self.n_future_steps]
        action_seq = nactions[:, self.n_obs_steps-1:self.n_obs_steps+self.n_future_steps-1]

        B, T, C, H, W = future_imgs.shape
        assert T == self.n_future_steps, f"Expected n_future_steps={self.n_future_steps}, got {T}."

        # flatten images
        history_imgs = history_imgs.view(B, self.n_obs_steps * C, H, W)

        total_loss = 0.0

        # for each future time-step
        for t in range(depth):
            x_true = future_imgs[:, t].view(B, C, H, W)

            noise_t = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (B,), device=x_true.device
            ).long()
            noise = torch.randn_like(x_true).view(B, C, H, W)
            noisy = self.noise_scheduler.add_noise(x_true, noise, noise_t)

            a = action_seq[:, t].view(B, -1)
            a = self.normalize_action(a)
            cond = self.diffusion_step_encoder(noise_t)
            inp = torch.cat([history_imgs, noisy], dim=1)
            inp = nn.GeometricTensor(inp, nn.FieldType(self.group, inp.shape[1] * [self.group.trivial_repr]))
            x = self.conv_in(inp)
            x = self.unet(x, cond, a)
            x = self.conv_out(x).tensor

            # 5) form prediction of clean x
            self.ddim_scheduler.set_timesteps(1, device=x_true.device)
            x_pred = self.ddim_scheduler.step(
                x, t, noisy
            ).pred_original_sample

            target = noise if self.noise_scheduler.config.prediction_type == 'epsilon' else x_true
            total_loss = total_loss + F.mse_loss(x, target, reduction='mean') + self.l1_loss_weight*F.l1_loss(x, target, reduction='mean')

            # 8) update history: drop oldest and append next_frame
            #    history has shape [B, n_obs*C, H, W]
            x_pred_flat = x_pred.view(B, C, H, W)
            history_imgs = torch.cat([
                history_imgs[:, C:],                # drop first C channels
                x_pred_flat
            ], dim=1)

        return total_loss / self.n_future_steps
    
    def normalize_action(self, action):
        action = (action - 256.0)/256.0
        action = torch.concat([action[:, 1:], action[:, :1]], dim=1)
        return action