from diffusion_policy.common.replay_buffer import ReplayBuffer
import tqdm

def merge_replay_buffers(dest: ReplayBuffer, src: ReplayBuffer):
    """
    Append all of `src` into `dest` episode-by-episode.
    After this, dest.n_episodes == old_dest.n_episodes + src.n_episodes.
    """
    for i in tqdm.trange(src.n_episodes):
        ep = src.get_episode(i, copy=True)
        dest.add_episode(ep)

dest = ReplayBuffer.create_from_path("/home/mlq/dl_project/diffusion_world_model/data/pusht_cchi_v7_with_random_play.zarr", mode='a')
src = ReplayBuffer.create_from_path("/home/mlq/dl_project/diffusion_world_model/data/pusht_random_play.zarr", mode='r')

merge_replay_buffers(dest, src)