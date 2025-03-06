import torch
import gymnasium as gym
from harl.envs.env_wrappers import IsaacLabWrapper, IsaacVideoWrapper
import os


def _t2n(x):
    return x.detach().cpu().numpy()


class IsaacLabEnv:
    def __init__(self, env_args):

        self.env = gym.make(env_args['task'], cfg=env_args['config'], render_mode="rgb_array")
        
        if env_args['video_settings']['video']:
            video_kwargs = {
                "video_folder": os.path.join(env_args['video_settings']['log_dir'], "train"),
                "step_trigger": lambda step: step % env_args['video_settings']['video_interval'] == 0,
                "video_length": env_args['video_settings']['video_length'],
                "disable_logger": True,
            }
            self.env = IsaacVideoWrapper(self.env, **video_kwargs)

        self.env = IsaacLabWrapper(self.env)

        self.unwrapped = self.env.unwrapped
        

        self.env_args = env_args
        self.n_envs = env_args["n_threads"]
        self.n_agents = self.env.num_agents
        self.share_observation_space = self.env.share_observation_space
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def step(self, actions):
        actions = torch.tensor(actions.transpose(1, 0, 2), dtype=torch.float32)
        obs_all, state_all, reward_all, done_all, info_all, _ = self.env.step(actions)
        return (
            _t2n(obs_all),
            _t2n(state_all),
            _t2n(reward_all),
            _t2n(done_all),
            [[{}, {}]] * self.n_envs,
            [None] * self.env_args["n_threads"],
        )

    def reset(self):
        obs, s_obs, _ = self.env.reset()
        return _t2n(obs), _t2n(s_obs), [None] * self.env_args["n_threads"]

    def close(self):
        pass
