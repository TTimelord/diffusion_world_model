from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.world_model.base_world_model import BaseWorldModel
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply

# Example diffusion model architecture
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb

class DiffusionWorldModelKeypointUnet(BaseWorldModel):
    """
    Diffusion-based keypoint world model. Predicts future keypoints conditioned on
    (past keypoint, action sequence).

    This is analogous to DiffusionWorldModelImageUnet, but for modeling the environment
    in keypoints rather than producing images.
    """
    def __init__(self,
                 shape_meta: dict,
                 noise_scheduler: DDPMScheduler,
                 obs_encoder: MultiImageObsEncoder,
                 n_obs_steps: int,
                 n_future_steps : int = 1,
                 num_inference_steps=None,
                 cond_channels=256,
                 depths = [2,2,2,2],
                 unet_dims = [256, 512, 1024],
                 attn_depths= [0,0,0,0],
                 **kwargs):
        """
        shape_meta:
            includes info about 'obs' shape: e.g. (3, H, W)
            includes info about 'action' shape: e.g. (action_dim,)
        noise_scheduler:
            A diffusion scheduler instance (DDPMScheduler, DDIM, etc.)
        n_obs_steps:
            How many past keypoint frames are used as context
        n_future_steps:
            How many future keypoint frames we want to predict
        """
        super().__init__(shape_meta, **kwargs)

        # get feature dim
        self.n_future_steps = n_future_steps

        # parse action dimension
        action_dim = shape_meta['action']['shape'][0]

        # get keypoint shape
        keypoint_size = shape_meta['obs']['keypoint']['shape'][-1]

        # self.diffusion_step_encoder = nn.Sequential(
        #     SinusoidalPosEmb(cond_channels),
        #     nn.Linear(cond_channels, cond_channels * 4),
        #     nn.Mish(),
        #     nn.Linear(cond_channels * 4, cond_channels),
        # )
        # self.act_proj = nn.Sequential(
        #     nn.Linear(action_dim * (n_obs_steps+n_future_steps-1), cond_channels),
        #     nn.SiLU(),
        #     nn.Linear(cond_channels, cond_channels),
        # )
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_channels, cond_channels),
            nn.SiLU(),
            nn.Linear(cond_channels, cond_channels),
        )
        # self.dense_in = nn.Linear(keypoint_size, 64)

        in_dim = n_obs_steps + n_future_steps
        self.unet = ConditionalUnet1D(
            input_dim=in_dim,
            global_cond_dim=action_dim*n_obs_steps,
            diffusion_step_embed_dim=cond_channels
        )

        self.conv_out = nn.Conv1d(in_dim, n_future_steps, kernel_size=3, padding=1)
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
        self.TT = n_obs_steps + n_future_steps

    def predict_future(self, obs_dict: Dict[str, torch.Tensor], action) -> Dict[str, torch.Tensor]:
        """
        Inference method:
          - 'obs_dict' should contain "obs" => (B, n_obs_steps, S)
            and "action" => (B, n_future_steps, action_dim)
          - returns a dict with "predicted_future" => (B, n_future_steps, S)
        """
        with torch.inference_mode():
            nobs = self.normalizer.normalize(obs_dict)
            nactions = self.normalizer['action'].normalize(action)
            history_keyp = nobs['obs']
            action_seq = nactions  # shape (B, n_future_steps, Da)
            device = history_keyp.device
            dtype = history_keyp.dtype
            # print('history_keyp', history_keyp.shape)
            B, To, S = history_keyp.shape
            assert To == self.n_obs_steps, f"Expected n_obs_steps={self.n_obs_steps}, got {To}."

            # Start from random noise
            # If you want the model to be deterministic, consider using zero noise
            noisy_keyp = torch.randn((B, self.n_future_steps, S), device=device, dtype=dtype)

            # set up timesteps
            self.noise_scheduler.set_timesteps(self.num_inference_steps, device=device)

            # reshape
            # history_imgs = history_imgs.view(B, self.n_obs_steps * C, H, W)
            action_seq = action_seq.view(B, -1)

            for t in self.noise_scheduler.timesteps:
                # print('noisy_keyp', noisy_keyp.shape)
                # forward the model
                # if torch.is_tensor(t) and len(t.shape) == 0:
                assert torch.is_tensor(t) and len(t.shape) == 0
                timesteps = t[None].to(device)
                timesteps = timesteps.expand(noisy_keyp.shape[0])

                x = torch.cat((history_keyp, noisy_keyp), dim=1)
                x = self.unet(x.transpose(1, 2), timesteps, global_cond=action_seq).transpose(1, 2)
                x = self.conv_out(x)

                noisy_keyp = self.noise_scheduler.step(
                    x, t, noisy_keyp
                ).prev_sample

            predicted = self.normalizer['obs'].unnormalize(noisy_keyp.view(B, self.n_future_steps, S))

        return {
            "predicted_future": predicted
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
        nobs = self.normalizer.normalize({'obs': batch['obs']})
        nactions = self.normalizer['action'].normalize(batch['action'])

        keyps = nobs['obs']
        history_keyps = keyps[:, :self.n_obs_steps]
        future_keyps = keyps[:, self.n_obs_steps:self.n_obs_steps+self.n_future_steps]
        action_seq = nactions[:, :self.n_obs_steps+self.n_future_steps-1]

        B, T, S = future_keyps.shape
        assert T == self.n_future_steps, f"Expected n_future_steps={self.n_future_steps}, got {T}."

        # flatten images
        x_0 = future_keyps.view(B, self.n_future_steps, S)
        # history_keyps = history_keyps.view(B, self.n_obs_steps, S)

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
        x = torch.cat((history_keyps, noisy_x), dim=1)
        x = self.unet(x.transpose(1, 2), timesteps, global_cond=action_seq).transpose(1, 2)
        x = self.conv_out(x)

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