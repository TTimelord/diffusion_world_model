from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.world_model.base_world_model import BaseWorldModel

# Example diffusion model architecture
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusers import UNet2DConditionModel

class DiffusionImageUnetWorldModel(BaseWorldModel):
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
                 down_dims=(256,512,1024),
                 cond_predict_scale=True,
                 num_inference_steps=None,
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

        # parse shapes
        obs_img_shape = shape_meta['obs_image']['shape']  # e.g. (3, H, W)
        c, h, w = obs_img_shape
        # flatten entire future horizon
        self.n_future_steps = n_future_steps

        # parse action dimension
        action_dim = shape_meta['action']['shape'][0]
        self.horizon = kwargs.get('horizon', 8)  # or however you define
        # We'll condition globally on the entire action chunk
        global_cond_dim = self.horizon * action_dim

        self.model = UNet2DConditionModel(
            in_channels=self.n_future_steps*3,                    # For RGB images (default is 4)
            out_channels=self.n_future_steps*3,                   # RGB output (default is 4)
            cross_attention_dim=global_cond_dim,          # Match your encoder's feature dimension (default is 1280)
            block_out_channels=(320, 640, 1280, 1280), # Channels for each block (default is same)
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

        self.noise_scheduler = noise_scheduler
        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        # normalizer for images if needed
        self.normalizer = LinearNormalizer()
        self.n_obs_steps = n_obs_steps

        # possibly store other info
        self.kwargs = kwargs

    def predict_future(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Inference method:
          - 'obs_dict' should contain "obs" => (B, n_obs_steps, C, H, W)
            and "action" => (B, horizon, action_dim)
          - returns a dict with "predicted_future" => (B, n_future_steps, C, H, W)
        """
        # for safety:
        self.model.eval()

        nobs = self.normalizer.normalize(obs_dict)
        obs_imgs = nobs['obs']  # shape (B, n_obs_steps, C, H, W)
        action_seq = nobs['action']  # shape (B, horizon, Da)
        device = obs_imgs.device
        dtype = obs_imgs.dtype

        B, To, C, H, W = obs_imgs.shape
        assert To == self.n_obs_steps, f"Expected n_obs_steps={self.n_obs_steps}, got {To}."

        # Flatten shape for the future frames we want to generate
        future_dim = self.future_dim

        # Global condition from action sequence
        global_cond = action_seq.reshape(B, -1)  # (B, horizon*Da)

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
            )
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
        # from the batch, we typically have:
        #   batch['obs']: (B, n_obs_steps, C, H, W)
        #   batch['action']: (B, horizon, Da)
        #   batch['future']: (B, n_future_steps, C, H, W)
        obs_imgs = batch['obs']
        action_seq = batch['action']
        future_imgs = batch['future']

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
        global_cond = action_seq.reshape(B, -1)

        # predict with the model
        model_out = self.model(
            noisy_x, 
            timesteps,
            local_cond=None,
            global_cond=global_cond
        )

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
