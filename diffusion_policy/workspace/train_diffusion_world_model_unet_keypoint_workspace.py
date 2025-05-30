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
from tqdm import trange
import numpy as np
import shutil
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.world_model.diffusion_world_model_unet_keypoint import DiffusionWorldModelKeypointUnet
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.dataset.pusht_dataset import PushTLowdimDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder

import matplotlib.pyplot as plt
from PIL import Image
import io
import torchvision.transforms.functional as TF

# Make an image from keyopints using matplotlib
def keypoint_to_image(keypoints):
    keypoints = np.array(keypoints).reshape(-1, 2)
    point_x, point_y = keypoints[:, 0], keypoints[:, 1]
    point_y = 512 - point_y
    plt.figure(figsize=(5, 5))
    plt.scatter(point_x[:-1], point_y[:-1], c='b')
    plt.scatter(point_x[-1], point_y[-1], c='r')
    plt.xlim(0, 512)
    plt.ylim(0, 512)
    plt.gca().set_aspect('equal', adjustable='box')
    buf = io.BytesIO()
    plt.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)  
    img_np = np.array(img)
    buf.close()
    plt.close()
    return img_np

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainDiffusionWorldModelUnetKeypointWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionWorldModelKeypointUnet = hydra.utils.instantiate(cfg.world_model)

        self.ema_model: DiffusionWorldModelKeypointUnet = None
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
        dataset: PushTLowdimDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, PushTLowdimDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()
        print('Got normalizer', normalizer)

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

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
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
                        raw_loss = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()
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
                if (self.epoch % cfg.training.rollout_every) == 0:
                    gt_keyp_video_path_list = []
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

                            # video_writer
                            gt_keyp_filename = pathlib.Path(self.output_dir).joinpath(
                            'media', wv.util.generate_id() + "_keyp_gt.mp4")
                            gt_keyp_filename.parent.mkdir(parents=False, exist_ok=True)
                            gt_keyp_filename = str(gt_keyp_filename)
                            gt_keyp_video_path_list.append(gt_keyp_filename)

                            gt_img_filename = pathlib.Path(self.output_dir).joinpath(
                            'media', wv.util.generate_id() + "_img_gt.mp4")
                            gt_img_filename.parent.mkdir(parents=False, exist_ok=True)
                            gt_img_filename = str(gt_img_filename)
                            gt_video_path_list.append(gt_img_filename)

                            predict_filename = pathlib.Path(self.output_dir).joinpath(
                            'media', wv.util.generate_id() + "_keyp_pred.mp4")
                            predict_filename.parent.mkdir(parents=False, exist_ok=True)
                            predict_filename = str(predict_filename) 
                            predict_video_path_list.append(predict_filename)

                            # image trajectory for calcualting mse
                            gt_block_keyp_trajectory = torch.tensor(dataset.replay_buffer['keypoint'][start_idx:start_idx + episode_length], dtype=torch.float32)
                            print('gt_block_keyp_trajectory', gt_block_keyp_trajectory.shape)
                            gt_agent_keyp_trajectory = torch.tensor(dataset.replay_buffer['state'][start_idx:start_idx + episode_length, :2], dtype=torch.float32).unsqueeze(1)
                            print('gt_agent_keyp_trajectory', gt_agent_keyp_trajectory.shape)
                            gt_keyp_trajectory = torch.cat([gt_block_keyp_trajectory, gt_agent_keyp_trajectory], dim=-2)
                            gt_keyp_trajectory_np = gt_keyp_trajectory.cpu().numpy()
                            
                            predicted_keyp_trajectory = torch.zeros_like(gt_keyp_trajectory)
                            predicted_keyp_trajectory[:world_model.n_obs_steps] = gt_keyp_trajectory[:world_model.n_obs_steps]

                            # ground truth key point video
                            self.video_recoder.start(gt_keyp_filename)
                            for i in range(episode_length):
                                image = keypoint_to_image(gt_keyp_trajectory_np[i])[:, :, :3]
                                self.video_recoder.write_frame(image)
                            self.video_recoder.stop()

                            # ground truth image video
                            self.video_recoder.start(gt_img_filename)
                            for i in range(episode_length):
                                image = dataset.replay_buffer['img'][start_idx + i]
                                self.video_recoder.write_frame(image.astype(np.uint8))
                            self.video_recoder.stop()

                            # predicted video
                            print('predicted_keyp_trajectory', predicted_keyp_trajectory.shape)
                            keyp_history = predicted_keyp_trajectory[:world_model.n_obs_steps]
                            print('keyp_history', keyp_history.shape)
                            # dataset._sample_to_data(dataset.replay_buffer)['obs'][start_idx + i:start_idx + i + world_model.n_obs_steps]
                            predicted_keyp_history = {
                                "obs": torch.tensor(keyp_history, dtype=torch.float32).to(device).view(keyp_history.shape[0], -1).unsqueeze(0)# 1, 4, 20
                            }
                            self.video_recoder.start(predict_filename)
                            for i in range(self.model.n_obs_steps):
                                image = keypoint_to_image(gt_keyp_trajectory_np[i])[:, :, :3]
                                self.video_recoder.write_frame(image)
                            for i in trange(episode_length - world_model.n_obs_steps):
                                action = dataset.replay_buffer['action'][start_idx + i:start_idx + i + world_model.n_obs_steps + world_model.n_future_steps - 1]
                                action = torch.tensor(action, dtype=torch.float32).to(device)
                                predicted_keyp = world_model.predict_future(predicted_keyp_history, action)["predicted_future"] # B, T, C, H, W
                                # append the first predicted image to update predicted_image_history
                                predicted_keyp_history['obs'] = torch.cat([predicted_keyp_history['obs'][:, 1:], predicted_keyp], dim=1)
                                predicted_keyp_trajectory[i + world_model.n_obs_steps] = predicted_keyp[0].reshape(-1, 2)
                                unnormalized_keyps = (predicted_keyp[0].detach().cpu().numpy())
                                self.video_recoder.write_frame((keypoint_to_image(unnormalized_keyps[0]))[:,:,:3])# only use the first frame
                            self.video_recoder.stop()
                        mse_list.append(torch.nn.functional.mse_loss(predicted_keyp_trajectory, gt_keyp_trajectory).item())
                    rollout_log = dict()
                    rollout_log['test_mse'] = np.mean(mse_list)
                    for i, video_path in enumerate(gt_video_path_list):
                        sim_video = wandb.Video(video_path)
                        rollout_log[f'gt_video_{i}'] = sim_video
                    for i, video_path in enumerate(gt_keyp_video_path_list):
                        sim_video = wandb.Video(video_path)
                        rollout_log[f'gt_keypoint_video_{i}'] = sim_video
                    for i, video_path in enumerate(predict_video_path_list):
                        sim_video = wandb.Video(video_path)
                        rollout_log[f'predict_keyp_video_{i}'] = sim_video
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
                # if (self.epoch % cfg.training.sample_every) == 0:
                #     with torch.no_grad():
                #         # sample trajectory from training set, and evaluate difference
                #         batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                #         obs_dict = batch['obs']
                #         print('keypoint', obs_dict.shape)
                #         target_future_keyps = obs_dict[:,self.model.n_obs_steps:self.model.n_obs_steps+self.model.n_future_steps,...]
                #         history_obs_dict = dict_apply(obs_dict, lambda x: x[:,:self.model.n_obs_steps,...])
                #         gt_action = batch['action']
                #         future_action = gt_action[:,:self.model.n_obs_steps+self.model.n_future_steps-1,...]
                        
                #         result = world_model.predict_future(history_obs_dict, future_action)
                #         pred_future_keyps = result["predicted_future"]
                #         mse = torch.nn.functional.mse_loss(pred_future_keyps, target_future_keyps)
                #         step_log['train_pred_image_mse_error'] = mse.item()
                #         del batch
                #         del obs_dict
                #         del gt_action
                #         del result
                #         del future_action
                #         del mse
                #         del target_future_keyps
                #         del history_obs_dict
                #         del pred_future_keyps
                
                # checkpoint
                # if (self.epoch % cfg.training.checkpoint_every) == 0:
                #     # checkpointing
                #     if cfg.checkpoint.save_last_ckpt:
                #         self.save_checkpoint()
                #     if cfg.checkpoint.save_last_snapshot:
                #         self.save_snapshot()

                #     # sanitize metric names
                #     metric_dict = dict()
                #     for key, value in step_log.items():
                #         new_key = key.replace('/', '_')
                #         metric_dict[new_key] = value
                    
                #     # We can't copy the last checkpoint here
                #     # since save_checkpoint uses threads.
                #     # therefore at this point the file might have been empty!
                #     topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                #     if topk_ckpt_path is not None:
                #         self.save_checkpoint(path=topk_ckpt_path)
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
    workspace = TrainDiffusionWorldModelUnetKeypointWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
