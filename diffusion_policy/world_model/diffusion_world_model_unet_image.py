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
from diffusers import UNet2DConditionModel

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
                 obs_encoder: MultiImageObsEncoder,
                 n_obs_steps: int,
                 n_future_steps : int = 1,
                 down_dims=(256,512,1024),
                 num_inference_steps=None,
                 cond_image_feature_dim=128,
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
        obs_feature_dim = obs_encoder.output_shape()[0]

        self.n_future_steps = n_future_steps

        # parse action dimension
        action_dim = shape_meta['action']['shape'][0]
        # We'll condition globally on the entire action chunk

        global_cond_dim = cond_image_feature_dim + action_dim * self.n_future_steps

        self.model = UNet2DConditionModel(
            sample_size=shape_meta['obs']['image']['shape'][1:],
            in_channels=self.n_future_steps*3,                    # For RGB images (default is 4)
            out_channels=self.n_future_steps*3,                   # RGB output (default is 4)
            cross_attention_dim=global_cond_dim,          # Match your encoder's feature dimension (default is 1280)
            attention_head_dim = 1,
            block_out_channels=(8, 16, 32, 32), 
            norm_num_groups=4,
            down_block_types=(
                "CrossAttnDownBlock2D",       # Custom downsampling blocks
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",                 # Custom upsampling blocks
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
            ),
        )

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'Trainable parameters in unet: {trainable_params}')

        self.obs_encoder = obs_encoder

        trainable_params = sum(p.numel() for p in self.obs_encoder.parameters() if p.requires_grad)
        print(f'Trainable parameters in resnet: {trainable_params}')

        self.cond_obs_feature_mlp = nn.Sequential(
            nn.Linear(obs_feature_dim * n_obs_steps, cond_image_feature_dim),
            nn.ReLU()
        )

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

            # condition
            condition_obs = {
                'image': history_imgs,
            }
            this_nobs = dict_apply(condition_obs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.cond_obs_feature_mlp(self.obs_encoder(this_nobs).reshape(B, -1))
            global_cond = torch.concat((nobs_features, action_seq.reshape(B, -1)), dim=-1)
            global_cond = global_cond.view(B, 1, -1)

            # Start from random noise
            # If you want the model to be deterministic, consider using zero noise
            noisy_image = torch.randn((B, self.n_future_steps * C, H, W), device=device, dtype=dtype)

            # set up timesteps
            self.noise_scheduler.set_timesteps(self.num_inference_steps)

            for t in self.noise_scheduler.timesteps:
                # forward the model
                model_out = self.model(
                    sample=noisy_image,
                    timestep=t,
                    encoder_hidden_states=global_cond
                ).sample
                # diffusion update
                noisy_image = self.noise_scheduler.step(
                    model_out, t, noisy_image
                ).prev_sample

            # now 'trajectory' is our final predicted future in flattened form
            predicted = noisy_image.view(B, self.n_future_steps, C, H, W)

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
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])

        print(f"After normalizer: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")

        imgs = nobs['image']
        history_imgs = imgs[:, :self.n_obs_steps]
        future_imgs = imgs[:, self.n_obs_steps:self.n_obs_steps+self.n_future_steps]
        action_seq = batch['action'][:, self.n_obs_steps:self.n_obs_steps+self.n_future_steps]

        B, T, C, H, W = future_imgs.shape
        assert T == self.n_future_steps, f"Expected n_future_steps={self.n_future_steps}, got {T}."

        x_0 = future_imgs.view(B, self.n_future_steps * C, H, W)

        # sample random timesteps for each sample
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=x_0.device
        ).long()

        # noise
        noise = torch.randn_like(x_0)

        # forward diffusion
        noisy_x = self.noise_scheduler.add_noise(x_0, noise, timesteps)

        # condition
        condition_obs = {
            'image': history_imgs,
        }
        this_nobs = dict_apply(condition_obs, lambda x: x.reshape(-1, *x.shape[2:]))
        nobs_features = self.cond_obs_feature_mlp(self.obs_encoder(this_nobs).reshape(B, -1))
        global_cond = torch.concat((nobs_features, action_seq.reshape(B, -1)), dim=-1)
        global_cond = global_cond.view(B, 1, -1)

        print(f"before model: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")

        print(f"noisy_x: {noisy_x.shape}, global_cond: {global_cond.shape}")

        # predict with the model
        model_out = self.model(
            sample=noisy_x,
            timestep=timesteps,
            encoder_hidden_states=global_cond
        ).sample

        # compute target
        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = x_0
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(model_out, target, reduction='mean')
        return loss
