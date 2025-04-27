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
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.model.equi.ssim import ms_ssim

OmegaConf.register_new_resolver("eval", eval, replace=True)

class EvalDiffusionWorldModelUnetImageWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.eval.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionWorldModelImageUnet = hydra.utils.instantiate(cfg.world_model)
        checkpoint = torch.load(cfg.pretrained_world_model_path)
        missing, unexpected = self.model.load_state_dict(checkpoint["state_dicts"]["model"], strict=True)

        # videos
        self.video_recoder = None
        if cfg.eval.record_video:
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

        # configure dataset
        dataset: PushTImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, PushTImageDataset)
        normalizer = dataset.get_normalizer()

        self.model.set_normalizer(normalizer)

        # create media directory
        media_dir = os.path.join(self.output_dir, 'media')
        os.makedirs(media_dir, exist_ok=True)

        # device transfer
        device = torch.device(cfg.eval.device)
        self.model.to(device)

        world_model = self.model
        world_model.eval()

        log_file = os.path.join(self.output_dir, "eval_log.txt")

        # run rollout
        gt_video_path_list = []
        predict_video_path_list = []
        assert cfg.eval.num_rollouts < len(dataset.replay_buffer.episode_ends), "Number of rollouts should be less than the number of episodes in the dataset"
        with torch.inference_mode():
            selected_indices = np.random.choice(len(dataset.replay_buffer.episode_ends), cfg.eval.num_rollouts, replace=False)
            mse_list = []
            ssim_list = []
            for index, idx in enumerate(selected_indices):
                print(f"Rollout {index+1}/{cfg.eval.num_rollouts}. Episode {idx}")
                start_idx = 0 if idx == 0 else dataset.replay_buffer.episode_ends[idx-1]
                end_idx = dataset.replay_buffer.episode_ends[idx]
                episode_length = end_idx - start_idx
                if episode_length < world_model.n_obs_steps + world_model.n_future_steps:
                    continue

                # video_writer
                if cfg.eval.record_video:
                    gt_filename = pathlib.Path(self.output_dir).joinpath(
                    'media', f"{idx}_GT.mp4")
                    gt_filename.parent.mkdir(parents=False, exist_ok=True)
                    gt_filename = str(gt_filename)
                    gt_video_path_list.append(gt_filename)
                    predict_filename = pathlib.Path(self.output_dir).joinpath(
                    'media', f"{idx}_prediction.mp4")
                    predict_filename.parent.mkdir(parents=False, exist_ok=True)
                    predict_filename = str(predict_filename) 
                    predict_video_path_list.append(predict_filename)

                # image trajectory for calcualting mse
                gt_image_trajectory = torch.tensor(dataset.replay_buffer['img'][start_idx:start_idx + episode_length], dtype=torch.float32)/255
                gt_image_trajectory = torch.moveaxis(gt_image_trajectory, -1, 1).to(device)  # B, C, H, W
                predicted_image_trajectory = torch.zeros_like(gt_image_trajectory)
                predicted_image_trajectory[:world_model.n_obs_steps] = gt_image_trajectory[:world_model.n_obs_steps]

                # ground truth video
                if cfg.eval.record_video:
                    self.video_recoder.start(gt_filename)
                    for i in range(episode_length - world_model.n_obs_steps):
                        image = dataset.replay_buffer['img'][start_idx + i]
                        self.video_recoder.write_frame(image.astype(np.uint8))
                    self.video_recoder.stop()

                # predicted video
                predicted_image_history = {
                    "image": gt_image_trajectory[:world_model.n_obs_steps].unsqueeze(0)
                }

                if cfg.eval.record_video:
                    self.video_recoder.start(predict_filename)
                if cfg.eval.auto_regressive:
                    depth = 1
                else:
                    depth = cfg.eval.teacher_forcing_depth
                    assert depth <= world_model.n_future_steps
                for i in range(world_model.n_obs_steps + depth):
                    image = dataset.replay_buffer['img'][start_idx + i]
                    if cfg.eval.record_video:
                        self.video_recoder.write_frame(image.astype(np.uint8))
                for i in tqdm.trange(episode_length - world_model.n_obs_steps - depth):
                    action = dataset.replay_buffer['action'][start_idx + i + world_model.n_obs_steps - 1:start_idx + i + world_model.n_obs_steps + world_model.n_future_steps - 1]
                    action = torch.tensor(action, dtype=torch.float32).to(device)
                    action = action.unsqueeze(0)
                    if not cfg.eval.auto_regressive:
                        predicted_image_history['image'] = gt_image_trajectory[i:i + world_model.n_obs_steps].unsqueeze(0)
                    for k in range(depth):
                        predicted_images = world_model.predict_future(predicted_image_history, action[:, k:k+1])["predicted_future"] # B, T, C, H, W
                        # append the first predicted image to update predicted_image_history
                        predicted_image_history['image'] = torch.cat([predicted_image_history['image'][:, 1:], predicted_images[:, :1]], dim=1)
                    unnormalized_images = predicted_images[0]
                    predicted_image_trajectory[i + world_model.n_obs_steps] = unnormalized_images[0]
                    if cfg.eval.record_video:
                        unnormalized_images = (unnormalized_images.detach().cpu().numpy() * 255).astype(np.uint8)
                        unnormalized_images = np.moveaxis(unnormalized_images, 1, -1)  # B, H, W, C
                        self.video_recoder.write_frame(unnormalized_images[0]) # only use the first frame
                if cfg.eval.record_video:
                    self.video_recoder.stop()
                mse_loss = torch.nn.functional.mse_loss(predicted_image_trajectory, gt_image_trajectory).item()
                mse_list.append(mse_loss)
                print(f"Mean Squared Error: {mse_loss}")

                log_str = f"Rollout {index+1}/{cfg.eval.num_rollouts}. Episode {idx} - Mean Squared Error: {mse_loss}. "

                if cfg.eval.calculate_ssim:
                    ssim_loss = 1 - ms_ssim(gt_image_trajectory, predicted_image_trajectory, data_range=1.0, size_average=True, win_size=11, weights=[0.6, 0.2, 0.2])
                    ssim_list.append(ssim_loss.item())
                    print(f"SSIM: {ssim_loss.item()}")
                    log_str += f"SSIM: {ssim_loss.item()}. "
                with open(log_file, "a") as f:
                    f.write(log_str + "\n")
            
            # calculate average mse and ssim
            avg_mse = np.mean(mse_list)
            print(f"Average Mean Squared Error: {avg_mse}")
            log_str = f"Average Mean Squared Error: {avg_mse}. "
            if cfg.eval.calculate_ssim:
                avg_ssim = np.mean(ssim_list)
                print(f"Average SSIM: {avg_ssim}")
                log_str += f"Average SSIM: {avg_ssim}. "
            with open(log_file, "a") as f:
                f.write(log_str + "\n")

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = EvalDiffusionWorldModelUnetImageWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
