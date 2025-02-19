"""
Modified from OpenAI Baselines code to work with multi-agent envs
"""
import numpy as np
import torch
import gym
import gymnasium
from multiprocessing import Process, Pipe
from abc import ABC, abstractmethod
import copy
from typing import Any, Mapping, Sequence, Tuple, Union



def tile_images(img_nhwc):
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.
    input: img_nhwc, list or array of images, ndim=4 once turned into array
        n = batch index, h = height, w = width, c = channel
    returns:
        bigim_HWc, ndarray with ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N) / H))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0] * 0 for _ in range(N, H * W)])
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H * h, W * w, c)
    return img_Hh_Ww_c


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle

        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle

        self.x = pickle.loads(ob)


class ShareVecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    Used to batch data from multiple copies of an environment, so that
    each observation becomes an batch of observations, and expected action is a batch of actions to
    be applied per-environment.
    """

    closed = False
    viewer = None

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(
        self, num_envs, observation_space, share_observation_space, action_space
    ):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.share_observation_space = share_observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a dict of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().

        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a dict of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close_extras(self):
        """
        Clean up the  extra resources, beyond what's in this base class.
        Only runs when not self.closed.
        """
        pass

    def close(self):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras()
        self.closed = True

    def step(self, actions):
        """
        Step the environments synchronously.

        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()

    def render(self, mode="human"):
        imgs = self.get_images()
        bigimg = tile_images(imgs)
        if mode == "human":
            self.get_viewer().imshow(bigimg)
            return self.get_viewer().isopen
        elif mode == "rgb_array":
            return bigimg
        else:
            raise NotImplementedError

    def get_images(self):
        """
        Return RGB images from each environment
        """
        raise NotImplementedError

    @property
    def unwrapped(self):
        if isinstance(self, VecEnvWrapper):
            return self.venv.unwrapped
        else:
            return self

    def get_viewer(self):
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.SimpleImageViewer()
        return self.viewer


def shareworker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            ob, s_ob, reward, done, info, available_actions = env.step(data)
            if "bool" in done.__class__.__name__:  # done is a bool
                if (
                    done
                ):  # if done, save the original obs, state, and available actions in info, and then reset
                    info[0]["original_obs"] = copy.deepcopy(ob)
                    info[0]["original_state"] = copy.deepcopy(s_ob)
                    info[0]["original_avail_actions"] = copy.deepcopy(available_actions)
                    ob, s_ob, available_actions = env.reset()
            else:
                if np.all(
                    done
                ):  # if done, save the original obs, state, and available actions in info, and then reset
                    info[0]["original_obs"] = copy.deepcopy(ob)
                    info[0]["original_state"] = copy.deepcopy(s_ob)
                    info[0]["original_avail_actions"] = copy.deepcopy(available_actions)
                    ob, s_ob, available_actions = env.reset()

            remote.send((ob, s_ob, reward, done, info, available_actions))
        elif cmd == "reset":
            ob, s_ob, available_actions = env.reset()
            remote.send((ob, s_ob, available_actions))
        elif cmd == "reset_task":
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == "render":
            if data == "rgb_array":
                fr = env.render(mode=data)
                remote.send(fr)
            elif data == "human":
                env.render(mode=data)
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_spaces":
            remote.send(
                (env.observation_space, env.share_observation_space, env.action_space)
            )
        elif cmd == "render_vulnerability":
            fr = env.render_vulnerability(data)
            remote.send((fr))
        elif cmd == "get_num_agents":
            remote.send((env.n_agents))
        else:
            raise NotImplementedError


class ShareSubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [
            Process(
                target=shareworker,
                args=(work_remote, remote, CloudpickleWrapper(env_fn)),
            )
            for (work_remote, remote, env_fn) in zip(
                self.work_remotes, self.remotes, env_fns
            )
        ]
        for p in self.ps:
            p.daemon = (
                True  # if the main process crashes, we should not cause things to hang
            )
            p.start()
        for remote in self.work_remotes:
            remote.close()
        self.remotes[0].send(("get_num_agents", None))
        self.n_agents = self.remotes[0].recv()
        self.remotes[0].send(("get_spaces", None))
        observation_space, share_observation_space, action_space = self.remotes[
            0
        ].recv()
        ShareVecEnv.__init__(
            self, len(env_fns), observation_space, share_observation_space, action_space
        )

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, share_obs, rews, dones, infos, available_actions = zip(*results)
        return (
            np.stack(obs),
            np.stack(share_obs),
            np.stack(rews),
            np.stack(dones),
            infos,
            np.stack(available_actions),
        )

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        results = [remote.recv() for remote in self.remotes]
        obs, share_obs, available_actions = zip(*results)
        return np.stack(obs), np.stack(share_obs), np.stack(available_actions)

    def reset_task(self):
        for remote in self.remotes:
            remote.send(("reset_task", None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.ps:
            p.join()
        self.closed = True


# single env
class ShareDummyVecEnv(ShareVecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        ShareVecEnv.__init__(
            self,
            len(env_fns),
            env.observation_space,
            env.share_observation_space,
            env.action_space,
        )
        self.actions = None
        try:
            self.n_agents = env.n_agents
        except:
            pass

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obs, share_obs, rews, dones, infos, available_actions = map(
            np.array, zip(*results)
        )

        for i, done in enumerate(dones):
            if "bool" in done.__class__.__name__:  # done is a bool
                if (
                    done
                ):  # if done, save the original obs, state, and available actions in info, and then reset
                    infos[i][0]["original_obs"] = copy.deepcopy(obs[i])
                    infos[i][0]["original_state"] = copy.deepcopy(share_obs[i])
                    infos[i][0]["original_avail_actions"] = copy.deepcopy(
                        available_actions[i]
                    )
                    obs[i], share_obs[i], available_actions[i] = self.envs[i].reset()
            else:
                if np.all(
                    done
                ):  # if done, save the original obs, state, and available actions in info, and then reset
                    infos[i][0]["original_obs"] = copy.deepcopy(obs[i])
                    infos[i][0]["original_state"] = copy.deepcopy(share_obs[i])
                    infos[i][0]["original_avail_actions"] = copy.deepcopy(
                        available_actions[i]
                    )
                    obs[i], share_obs[i], available_actions[i] = self.envs[i].reset()
        self.actions = None

        return obs, share_obs, rews, dones, infos, available_actions

    def reset(self):
        results = [env.reset() for env in self.envs]
        obs, share_obs, available_actions = map(np.array, zip(*results))
        return obs, share_obs, available_actions

    def close(self):
        for env in self.envs:
            env.close()

    def render(self, mode="human"):
        if mode == "rgb_array":
            return np.array([env.render(mode=mode) for env in self.envs])
        elif mode == "human":
            for env in self.envs:
                env.render(mode=mode)
        else:
            raise NotImplementedError

class SingleAgentIsaacLabWrapper(object):
    def __init__(self, env: Any) -> None:
        self._env = env

    def __getattr__(self, key: str) -> Any:
        if hasattr(self._env, key):
            return getattr(self._env, key)
        raise AttributeError(f"Wrapped environment ({self._env.__class__.__name__}) does not have attribute '{key}'")
    
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:

        _obs, reward, terminated, truncated, info = self._env.step(actions[self.agents[0]])

        _obs = {self.agents[0]:_obs['policy']}
        reward = {self.agents[0]:reward}
        terminated = {self.agents[0]:terminated}
        truncated = {self.agents[0]:truncated}
        info = {self.agents[0]:info}

        return _obs, reward, terminated, truncated, info

    def state(self) -> Any:
        return None

    @property
    def num_agents(self) -> int:
        return 1
    
    @property
    def agents(self) -> Sequence[str]:
        return ["single_agent"]
    
    @property
    def share_observation_space(self) -> Sequence[gym.Space]:
        return [self._env.single_observation_space['policy']]
    
    @property
    def observation_space(self) -> Sequence[gym.Space]:
        return [self._env.single_observation_space['policy']]
    
    @property
    def action_space(self) -> Sequence[gym.Space]:
        return [self._env.single_action_space]
        

class IsaacLabWrapper(object):
    def __init__(self, env: Any) -> None:
        """Base wrapper class for multi-agent environments

        :param env: The multi-agent environment to wrap
        :type env: Any supported multi-agent environment
        """
        if not hasattr(env, "agents"):
            self._env = SingleAgentIsaacLabWrapper(env)
            self.unwrapped = self._env
        else:
            self._env = env
            try:
                self.unwrapped = self._env.unwrapped
            except:
                self.unwrapped = env

        self._agent_map = {agent: i for i, agent in enumerate(self.unwrapped.agents)}
        self._agent_map_inv = {i: agent for i, agent in enumerate(self.unwrapped.agents)}

        # device
        if hasattr(self.unwrapped, "device"):
            self._device = torch.device(self.unwrapped.device)
        else:
            self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __getattr__(self, key: str) -> Any:
        """Get an attribute from the wrapped environment

        :param key: The attribute name
        :type key: str

        :raises AttributeError: If the attribute does not exist

        :return: The attribute value
        :rtype: Any
        """
        if hasattr(self._env, key):
            return getattr(self._env, key)
        if hasattr(self.unwrapped, key):
            return getattr(self.unwrapped, key)
        raise AttributeError(f"Wrapped environment ({self.unwrapped.__class__.__name__}) does not have attribute '{key}'")

    def reset(self) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        _obs, _ = self._env.reset()

        # turn obs into array with dims [batch, agent, *obs_shape]
        _obs = [o for o in _obs.values()]
        obs = torch.stack(_obs, axis=1)

        s_obs = [self.unwrapped.state() for _ in range(self.unwrapped.num_agents)]
        if s_obs[0] != None:
            s_obs = torch.stack(s_obs, axis=1)
        else:
            s_obs = torch.stack(_obs, axis=1)
        
        return obs, s_obs, None

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: dictionary of torch.Tensor

        :raises NotImplementedError: Not implemented

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of dictionaries of torch.Tensor and any other info
        """

        actions = {self._agent_map_inv[i]:actions[i] for i in range(self.unwrapped.num_agents)}

        _obs, reward, terminated, truncated, info = self._env.step(actions)

        obs = torch.stack([_obs[agent] for agent in self.unwrapped.agents], axis=1)
        s_obs = [self.unwrapped.state() for _ in range(self.unwrapped.num_agents)]
        if s_obs[0] != None:
            s_obs = torch.stack(s_obs, axis=1)
        else:
            s_obs = torch.stack([_obs[agent] for agent in self.unwrapped.agents], axis=1)
        reward = torch.stack([reward[agent] for agent in self.unwrapped.agents], axis=1)
        reward = reward.unsqueeze(-1)
        terminated = torch.stack([terminated[agent] for agent in self.unwrapped.agents], axis=1)
        truncated = torch.stack([truncated[agent] for agent in self.unwrapped.agents], axis=1)

        dones = torch.logical_or(terminated, truncated)

        return obs, s_obs, reward, dones, None, None

    def state(self) -> torch.Tensor:
        """Get the environment state

        :raises NotImplementedError: Not implemented

        :return: State
        :rtype: torch.Tensor
        """
        raise NotImplementedError 

    def render(self, *args, **kwargs) -> Any:
        """Render the environment

        :raises NotImplementedError: Not implemented

        :return: Any value from the wrapped environment
        :rtype: any
        """
        raise NotImplementedError

    def close(self) -> None:
        """Close the environment

        :raises NotImplementedError: Not implemented
        """
        raise NotImplementedError

    @property
    def device(self) -> torch.device:
        """The device used by the environment

        If the wrapped environment does not have the ``device`` property, the value of this property
        will be ``"cuda"`` or ``"cpu"`` depending on the device availability
        """
        return self._device

    @property
    def num_envs(self) -> int:
        """Number of environments

        If the wrapped environment does not have the ``num_envs`` property, it will be set to 1
        """
        return self.unwrapped.num_envs if hasattr(self.unwrapped, "num_envs") else 1

    @property
    def num_agents(self) -> int:
        """Number of current agents

        Read from the length of the ``agents`` property if the wrapped environment doesn't define it
        """
        try:
            return self.unwrapped.num_agents
        except:
            return len(self.agents)

    @property
    def max_num_agents(self) -> int:
        """Number of possible agents the environment could generate

        Read from the length of the ``possible_agents`` property if the wrapped environment doesn't define it
        """
        try:
            return self.unwrapped.max_num_agents
        except:
            return len(self.possible_agents)

    @property
    def agents(self) -> Sequence[str]:
        """Names of all current agents

        These may be changed as an environment progresses (i.e. agents can be added or removed)
        """
        return self.unwrapped.agents

    @property
    def possible_agents(self) -> Sequence[str]:
        """Names of all possible agents the environment could generate

        These can not be changed as an environment progresses
        """
        return self.unwrapped.possible_agents

    # @property
    # def state_spaces(self) -> Mapping[str, gym.Space]:
    #     """State spaces

    #     Since the state space is a global view of the environment (and therefore the same for all the agents),
    #     this property returns a dictionary (for consistency with the other space-related properties) with the same
    #     space for all the agents
    #     """
    #     space = self._unwrapped.state_space
    #     return {agent: space for agent in self.possible_agents}

    @property
    def observation_space(self) -> Mapping[int, gym.Space]:
        """Observation spaces
        """
        return {self._agent_map[k]: v for k, v in self.unwrapped.observation_spaces.items()}

    @property
    def action_space(self) -> Mapping[int, gym.Space]:
        """Action spaces
        """
        return {self._agent_map[k]: v for k, v in self.unwrapped.action_spaces.items()}
    
    @property
    def share_observation_space(self) -> Mapping[int, gym.Space]:
        """Share observation space
        """
        if self.unwrapped.state_space.shape[0] == 0:
            return {self._agent_map[k]: self.unwrapped.observation_spaces[k] for k in self.unwrapped.agents}
        else:
            return {self._agent_map[k]: self.unwrapped.state_space for k in self.unwrapped.agents}
    

class IsaacVideoWrapper(gymnasium.wrappers.RecordVideo):

    def step(self, action):
        """Steps through the environment using action, recording observations if :attr:`self.recording`."""
        (
            observations,
            rewards,
            terminateds,
            truncateds,
            infos,
        ) = self.env.step(action)

        self.step_id += 1
        self.episode_id += 1


        if self.recording:
            assert self.video_recorder is not None
            self.video_recorder.capture_frame()
            self.recorded_frames += 1
            if self.video_length > 0:
                if self.recorded_frames > self.video_length:
                    self.close_video_recorder()
                    print("end recording")

        elif self._video_enabled():
            self.start_video_recorder()
            print("start recording")

        return observations, rewards, terminateds, truncateds, infos