# Diffusion world model
This repo is built on top of [Diffusion Policy](https://github.com/real-stanford/diffusion_policy)

Link to the components used in this project:
- [World Model](https://github.com/TTimelord/diffusion_world_model/tree/diffusion_world_model/diffusion_policy/world_model)
- [Equivariant Architectures](https://github.com/TTimelord/diffusion_world_model/tree/diffusion_world_model/diffusion_policy/model/equi)
- [Training Loop](https://github.com/TTimelord/diffusion_world_model/tree/diffusion_world_model/diffusion_policy/workspace)
- [Configs](https://github.com/TTimelord/diffusion_world_model/tree/diffusion_world_model/diffusion_policy/config)

## Train
```bash
python train.py --config-dir=diffusion_policy/config --config-name=train_diffusion_world_model_unet_image_workspace.yaml training.device=cuda:0
python train.py --config-dir=diffusion_policy/config --config-name=train_autoencoder_workspace.yaml training.device=cuda:0
```

To run a quick debug train:
```bash
python train.py --config-dir=diffusion_policy/config --config-name=train_diffusion_world_model_unet_image_workspace.yaml training.device=cuda:0 training.debug=True
python train.py --config-dir=diffusion_policy/config --config-name=train_diffusion_world_model_latent_unet_image_workspace.yaml training.device=cuda:0 training.debug=True
```

# Evaluation
```bash
python train.py --config-dir=diffusion_policy/config --config-name=eval_unet_image.yaml eval.device=cuda:0
```

