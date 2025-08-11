import torch
import numpy as np
import gymnasium as gym
from harl.envs.env_wrappers import IsaacLabWrapper, IsaacLabAdversarialWrapper, IsaacVideoWrapper
import os


def _t2n(x):
    return x.detach().cpu().numpy()

def flatten_dict_to_numpy(d, parent_key="", sep="."):
    """
    Recursively flattens a nested dictionary `d`.
    
    - Nested keys are joined by `sep` to form a single string key.
    - Values that are integers/floats/tensors are converted to NumPy arrays.
      For tensors (CPU/CUDA), uses `tensor.detach().cpu().numpy()`.
    
    Example:
        nested_dict = {
            "a": 1,
            "b": {"x": 2.5, "y": torch.tensor([3, 4], device='cuda')},
        }
        flattened = flatten_dict_to_numpy(nested_dict)
        
        # flattened -> {
        #   "a": np.array(1),
        #   "b.x": np.array(2.5),
        #   "b.y": np.array([3, 4])  # from CUDA to CPU
        # }
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            # Recursively flatten
            items.extend(flatten_dict_to_numpy(v, new_key, sep=sep).items())
        else:
            # Convert to numpy array
            if isinstance(v, (int, float)):
                value_np = np.array(v)
            elif isinstance(v, torch.Tensor):
                value_np = v.detach().cpu().numpy()
            else:
                raise TypeError(
                    f"Unsupported type '{type(v).__name__}' "
                    f"for key '{new_key}'. "
                    "Expected int, float, or torch.Tensor."
                )
            items.append((new_key, value_np))
    return dict(items)



class IsaacLabEnv:
    def __init__(self, env_args):

        self.env = gym.make(env_args['task'], cfg=env_args['config'], render_mode="rgb_array", **env_args)
        
        if env_args['video_settings']['video']:
            video_kwargs = {
                "video_folder": os.path.join(env_args['video_settings']['log_dir'], "train"),
                "step_trigger": lambda step: step % env_args['video_settings']['video_interval'] == 0,
                "video_length": env_args['video_settings']['video_length'],
                "disable_logger": True,
            }
            self.env = gym.wrappers.RecordVideo(self.env, **video_kwargs)

        if hasattr(self.env.env.cfg, "teams"):
            self.env =  IsaacLabAdversarialWrapper(self.env)
        else:
            self.env = IsaacLabWrapper(self.env)

        self.unwrapped = self.env.unwrapped

        self.log_info = {}

        self.env_args = env_args
        self.n_envs = env_args["n_threads"]
        self.n_agents = self.env.num_agents
        self.share_observation_space = self.env.share_observation_space
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def step(self, actions: torch.Tensor):
        actions = actions.permute(1, 0, 2)
        obs_all, state_all, reward_all, done_all, info_all, _ = self.env.step(actions)
        self.log_info = flatten_dict_to_numpy(info_all["log"])
        return (
            obs_all,
            state_all,
            reward_all,
            done_all,
            [[{}, {}]] * self.n_envs,
            [None] * self.env_args["n_threads"],
        )

    def reset(self):
        obs, s_obs, _ = self.env.reset()
        return obs, s_obs, [None] * self.env_args["n_threads"]

    def close(self):
        self.unwrapped.sim._disable_app_control_on_stop_handle = True
