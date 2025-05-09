if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)


import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import copy
import random
import wandb
import wandb.sdk.data_types.video as wv
import tqdm
import numpy as np
import shutil
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.dataset.pusht_image_dataset import PushTImageDataset
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.model.equi.equi_obs_autoencoder import Autoencoder


OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainAutoencoderWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        # self.model = Autoencoder()
        self.model: Autoencoder = hydra.utils.instantiate(cfg.autoencoder)
        if cfg.finetune:
            checkpoint = torch.load(self.cfg.pretrained_auto_encoder_path, map_location=self.model.device)
            missing, unexpected = self.model.load_state_dict(checkpoint["state_dicts"]["model"], strict=True)

        self.ema_model: Autoencoder = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0
        
    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # resume training
        if cfg.training.resume:
            tag = cfg.training.ckpt_tag if 'ckpt_tag' in cfg.training else 'latest'
            lastest_ckpt_path = self.get_checkpoint_path(tag)
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: PushTImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, PushTImageDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure env
        # env_runner: BaseImageRunner
        # env_runner = hydra.utils.instantiate(
        #     cfg.task.env_runner,
        #     output_dir=self.output_dir)
        # assert isinstance(env_runner, BaseImageRunner)
        
        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # create media directory
        media_dir = os.path.join(self.output_dir, 'media')
        os.makedirs(media_dir, exist_ok=True)

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None


        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                # if cfg.training.freeze_encoder:
                #     self.model.obs_encoder.eval()
                #     self.model.obs_encoder.requires_grad_(False)

                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # compute loss
                        if cfg.finetune:
                            raw_loss = self.model.compute_finetune_loss(batch)
                        else:
                            raw_loss = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # # profiling
                        # with profile(
                        #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                        #     profile_memory=True,  # <<-- important!
                        #     record_shapes=True
                        # ) as prof:
                        #     with record_function("model_inference"):
                        #         # run your forward pass

                        #         raw_loss = self.model.compute_loss(batch)
                        #         loss = raw_loss / cfg.training.gradient_accumulate_every
                        #         loss.backward()

                        # # Then print or sort the profiling info
                        # print(prof.key_averages().table(
                        #     sort_by="self_cuda_memory_usage", row_limit=200
                        # ))

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        
                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break
                    
                    # run rollout on training
                    if (self.epoch % cfg.training.rollout_every) == 0:
                        with torch.no_grad():
                            indices = random.sample(range(len(dataset)), cfg.training.num_rollouts)
                            sampled_images = []
                            for idx in indices:
                                image = dataset[idx]['obs']['image'][0].unsqueeze(0)
                                sampled_images.append(image)
                            images = torch.cat(sampled_images, dim=0)
                            batch = {'image': images}
                            nobs = self.model.normalizer.normalize(batch)
                            gts = nobs['image']
                            latent = self.model.encode(gts)
                            if isinstance(latent, tuple):
                                latent = latent[0]
                            reconstructions = self.model.decode(latent)
                            for idx, gt in enumerate(list(gts)):
                                save_image(gt, pathlib.Path(self.output_dir).joinpath(
                                'media', f"{self.epoch}_train_gt_{wv.util.generate_id()}.jpg"))
                                save_image(reconstructions[idx], pathlib.Path(self.output_dir).joinpath(
                                'media', f"{self.epoch}_train_reconstruct_{wv.util.generate_id()}.jpg"))
                                reconstruct_wandb_img = wandb.Image(reconstructions[idx])
                                gt_wandb_img = wandb.Image(gt)
                                step_log[f'train gt {idx}'] = gt_wandb_img
                                step_log[f'train reconstruction {idx}'] = reconstruct_wandb_img
                                
                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                world_model = self.model
                if cfg.training.use_ema:
                    world_model = self.ema_model
                world_model.eval()

                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                loss = self.model.compute_loss(batch)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break

                            indices = random.sample(range(len(val_dataset)), cfg.training.num_rollouts)
                            sampled_images = []
                            for idx in indices:
                                image = val_dataset[idx]['obs']['image'][0].unsqueeze(0)
                                sampled_images.append(image)
                            images = torch.cat(sampled_images, dim=0)
                            batch = {'image': images}
                            nobs = self.model.normalizer.normalize(batch)
                            gts = nobs['image']
                            latent = self.model.encode(gts)
                            if isinstance(latent, tuple):
                                latent = latent[0]
                            reconstructions = self.model.decode(latent)
                            for idx, gt in enumerate(list(gts)):
                                save_image(gt, pathlib.Path(self.output_dir).joinpath(
                                'media', f"{self.epoch}_val_gt_{wv.util.generate_id()}.jpg"))
                                save_image(reconstructions[idx], pathlib.Path(self.output_dir).joinpath(
                                'media', f"{self.epoch}_val_reconstruct_{wv.util.generate_id()}.jpg"))
                                
                                reconstruct_wandb_img = wandb.Image(reconstructions[idx])
                                gt_wandb_img = wandb.Image(gt)
                                step_log[f'val gt {idx}'] = gt_wandb_img
                                step_log[f'val reconstruction {idx}'] = reconstruct_wandb_img
                                    
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss
                            # step_log['test_mse'] = val_loss
                            
                
                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                # ========= eval end for this epoch ==========
                world_model.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionWorldModelUnetImageWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
