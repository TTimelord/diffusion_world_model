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
from torch.profiler import profile, record_function, ProfilerActivity
import copy
import random
import wandb
import wandb.sdk.data_types.video as wv
import tqdm
import numpy as np
import shutil
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.world_model.diffusion_world_model_unet_image import DiffusionWorldModelImageUnet
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.dataset.pusht_image_dataset import PushTImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainDiffusionWorldModelUnetImageWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionWorldModelImageUnet = hydra.utils.instantiate(cfg.world_model)

        self.ema_model: DiffusionWorldModelImageUnet = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0

        # videos
        self.video_recoder: VideoRecorder = VideoRecorder.create_h264(
                        fps=cfg.task.env_runner.fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=22,
                        thread_type='FRAME',
                        thread_count=1
                    )
        
    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
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

        if not hasattr(cfg.training, 'autoregressive_training_begin_epoch') or cfg.training.autoregressive_training_begin_epoch is None:
            cfg.training.autoregressive_training_begin_epoch = cfg.training.num_epochs

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.autoregressive_training_begin_epoch = 1
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.num_rollouts = 1
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1


        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                if cfg.training.freeze_encoder:
                    self.model.obs_encoder.eval()
                    self.model.obs_encoder.requires_grad_(False)

                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # compute loss
                        if self.epoch < cfg.training.autoregressive_training_begin_epoch:
                            raw_loss = self.model.compute_loss(batch)
                        else:
                            raw_loss = self.model.compute_autoregressive_loss(batch, depth=self.model.n_future_steps)
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

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                world_model = self.model
                if cfg.training.use_ema:
                    world_model = self.ema_model
                world_model.eval()

                # run rollout
                if self.epoch>0 and (self.epoch % cfg.training.rollout_every) == 0:
                    gt_video_path_list = []
                    predict_video_path_list = []
                    with torch.inference_mode():
                        selected_indices = np.random.choice(len(dataset.replay_buffer.episode_ends), cfg.training.num_rollouts, replace=False)
                        mse_list = []
                        for idx in selected_indices:
                            start_idx = 0 if idx == 0 else dataset.replay_buffer.episode_ends[idx-1]
                            end_idx = dataset.replay_buffer.episode_ends[idx]
                            episode_length = end_idx - start_idx
                            if episode_length < world_model.n_obs_steps + world_model.n_future_steps:
                                continue
                            episode_length = min(episode_length, 200) # truncate episode

                            # video_writer
                            gt_filename = pathlib.Path(self.output_dir).joinpath(
                            'media', wv.util.generate_id() + ".mp4")
                            gt_filename.parent.mkdir(parents=False, exist_ok=True)
                            gt_filename = str(gt_filename)
                            gt_video_path_list.append(gt_filename)
                            predict_filename = pathlib.Path(self.output_dir).joinpath(
                            'media', wv.util.generate_id() + ".mp4")
                            predict_filename.parent.mkdir(parents=False, exist_ok=True)
                            predict_filename = str(predict_filename) 
                            predict_video_path_list.append(predict_filename)

                            # image trajectory for calcualting mse
                            gt_image_trajectory = torch.tensor(dataset.replay_buffer['img'][start_idx:start_idx + episode_length], dtype=torch.float32)/255
                            predicted_image_trajectory = torch.zeros_like(gt_image_trajectory)
                            predicted_image_trajectory[:world_model.n_obs_steps] = gt_image_trajectory[:world_model.n_obs_steps]

                            # ground truth video
                            self.video_recoder.start(gt_filename)
                            for i in range(episode_length - world_model.n_obs_steps):
                                image = dataset.replay_buffer['img'][start_idx + i]
                                self.video_recoder.write_frame(image.astype(np.uint8))
                            self.video_recoder.stop()

                            # predicted video
                            image_history = dataset.replay_buffer['img'][start_idx:start_idx + world_model.n_obs_steps]
                            image_history = np.moveaxis(image_history,-1,1)/255
                            predicted_image_history = {
                                "image": torch.tensor(image_history, dtype=torch.float32).to(device).unsqueeze(0)
                            }
                            self.video_recoder.start(predict_filename)
                            for i in range(world_model.n_obs_steps):
                                image = dataset.replay_buffer['img'][start_idx + i]
                                self.video_recoder.write_frame(image.astype(np.uint8))
                            for i in tqdm.trange(episode_length - world_model.n_obs_steps):
                                action = dataset.replay_buffer['action'][start_idx + i + world_model.n_obs_steps - 1:start_idx + i + world_model.n_obs_steps + world_model.n_future_steps - 1]
                                action = torch.tensor(action, dtype=torch.float32).to(device)
                                action = action.unsqueeze(0)
                                predicted_images = world_model.predict_future(predicted_image_history, action)["predicted_future"] # B, T, C, H, W
                                # append the first predicted image to update predicted_image_history
                                predicted_image_history['image'] = torch.cat([predicted_image_history['image'][:, 1:], predicted_images[:, :1]], dim=1)
                                unnormalized_images = predicted_images[0]
                                unnormalized_images = torch.moveaxis(unnormalized_images, 1, -1)
                                predicted_image_trajectory[i + world_model.n_obs_steps] = unnormalized_images[0]
                                unnormalized_images = (unnormalized_images.detach().cpu().numpy() * 255).astype(np.uint8)
                                self.video_recoder.write_frame(unnormalized_images[0]) # only use the first frame
                            self.video_recoder.stop()
                        mse_list.append(torch.nn.functional.mse_loss(predicted_image_trajectory, gt_image_trajectory).item())
                    rollout_log = dict()
                    rollout_log['test_mse'] = np.mean(mse_list)
                    for i, video_path in enumerate(gt_video_path_list):
                        sim_video = wandb.Video(video_path)
                        rollout_log[f'gt_video_{i}'] = sim_video
                    for i, video_path in enumerate(predict_video_path_list):
                        sim_video = wandb.Video(video_path)
                        rollout_log[f'predict_video_{i}'] = sim_video
                    step_log.update(rollout_log)

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
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        obs_dict = batch['obs']
                        target_future_images = obs_dict['image'][:,self.model.n_obs_steps:self.model.n_obs_steps+1,...]
                        history_obs_dict = dict_apply(obs_dict, lambda x: x[:,:self.model.n_obs_steps,...])
                        gt_action = batch['action']
                        future_action = gt_action[:,self.model.n_obs_steps-1:self.model.n_obs_steps+self.model.n_future_steps-1,...]
                        
                        result = world_model.predict_future(history_obs_dict, future_action)
                        pred_future_images = result["predicted_future"]
                        mse = torch.nn.functional.mse_loss(pred_future_images, target_future_images)
                        step_log['train_pred_image_mse_error'] = mse.item()
                        # log predicted and ground truth future images
                        pred_images = (pred_future_images[0].cpu().numpy() * 255).astype(np.uint8)
                        pred_images = np.moveaxis(pred_images, 1, -1)
                        gt_images = (target_future_images[0].cpu().numpy() * 255).astype(np.uint8)
                        gt_images = np.moveaxis(gt_images, 1, -1)
                        pred_wandb_img = wandb.Image(pred_images[0], caption="Predicted future image")
                        gt_wandb_img = wandb.Image(gt_images[0], caption="Ground truth future image")
                        step_log['train_pred_future_image'] = pred_wandb_img
                        step_log['train_gt_future_image'] = gt_wandb_img
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del future_action
                        del mse
                        del target_future_images
                        del history_obs_dict
                        del pred_future_images
                
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
