"""
Modified from OpenAI Baselines code to work with multi-agent envs
"""
import numpy as np
import torch
import gym
import gymnasium
from gym import spaces
from multiprocessing import Process, Pipe
from abc import ABC, abstractmethod
import copy
from typing import Any, Mapping, Sequence, Tuple, Union

#TODO get neuron states to update
#TODO multiagent cases
#TODO create single agent test env
#TODO single agent case

#TODO add in position and connection information

class neuronWrapper(object):
    """
    This will transform a single agent or multi agent enviroment into a neuron multi agent enviroment.
    Where each agent in the unwapped enviroment is represented by many neuron agents.
    The neuron agents will choose connections to observation, action output, and other neurons.
    The neurons will be orginized in a ND grid and choose connections based on position in the grid.
    Neurons will be able to move through the grid, change connections, and process observations as their actions.
    Each agent will be trained individually using RL and will be unaware of the other agents state.
    """

    def __init__(self, env: Any, neuron_args: Any) -> None:
        self._env = env

        try:
            self._unwrapped = self._env.unwrapped
        except:
            self._unwrapped = env

        self.num_unwrapped_agents = self._unwrapped.n_agents
        self.position_dims = neuron_args["position_dims"]
        self.neuron_bandwidth = neuron_args["model"]["neuron_bandwidth"]
        self.neuron_input = self.neuron_bandwidth + self.position_dims
        self.neuron_output = self.neuron_bandwidth + self.position_dims + 2
        self.speed_factor = neuron_args["speed_factor"]
        self.step_count = 0
        self.num_hidden_neurons_per_agent = neuron_args["num_hidden_neurons_per_agent"]

        self.enviroment_observation_space = self._env.observation_space
        self.enviroment_action_space = self._env.action_space

        self.num_visible_input_neurons = self.enviroment_observation_space[0].shape[0]//self.neuron_bandwidth + (1 if self.enviroment_observation_space[0].shape[0]%self.neuron_bandwidth != 0 else 0)
        self.num_visible_output_neurons = self.enviroment_action_space[0].shape[0]//self.neuron_bandwidth + (1 if self.enviroment_action_space[0].shape[0]%self.neuron_bandwidth != 0 else 0)

        self._observation_space = [spaces.Box(low=-1.0,high=1.0,shape=(self.neuron_input,),dtype=np.float32) for _ in range(self.n_agents)]
        self._action_space = [spaces.Box(low=-1.0,high=1.0,shape=(self.neuron_output,),dtype=np.float32) for _ in range(self.n_agents)]
        self._shared_observation_space = [spaces.Box(low=-1.0,high=1.0,shape=(self.neuron_input,),dtype=np.float32) for _ in range(self.n_agents)]

    def _reshape_obs(self, obs: np.ndarray) -> torch.Tensor:

        size = self.enviroment_observation_space[0].shape[0]

        batch = obs.shape[0]
        
        out = np.zeros((batch, self.n_agents, self.neuron_input), dtype=obs.dtype)

        for j in range(self.num_unwrapped_agents):

            out_slice = out[:, j * self.total_neurons_per_agent : j * self.total_neurons_per_agent + self.num_visible_input_neurons, :]
            out_slice_flat = out_slice.reshape(batch, self.num_visible_input_neurons * self.neuron_input)

            out_slice_flat[:, :size] = obs[:, j, :]
        
        return out
    
    def _reshape_actions(self, actions: np.ndarray) -> torch.Tensor:

        size = self.enviroment_action_space[0].shape[0]

        batch = actions.shape[0]
        
        out = np.zeros((batch, self.num_unwrapped_agents, size + (self.neuron_input-size%self.neuron_input)), dtype=actions.dtype)
        
        for j in range(self.num_unwrapped_agents):
            out_slice = out[:, j : j+1, :]
            
            out_slice_flat = out_slice.reshape(batch, self.num_visible_output_neurons, self.neuron_input)

            act_pos = j*self.total_neurons_per_agent+self.num_visible_input_neurons

            out_slice_flat[:,:,:] = actions[:, act_pos:act_pos+self.num_visible_output_neurons,:]
        
        return out[:,:, :size]

    def _convert_observation(self, obs) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        """
        Converts the observation from the enviroment to the observation for the neurons
        """
        #TODO add in prev position
        return self._reshape_obs(np.array(obs))
    
    def _convert_actions(self, actions) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        """
        Converts the action from the enviroment to the actions for the neurons
        """

        env_actions = actions[:,:,:self.neuron_input]
        neron_actions = actions[:,:,self.neuron_bandwidth:self.neuron_output]

        env_actions = self._reshape_actions(np.array(env_actions))

        return env_actions, neron_actions

    def reset(self) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        """
        Will reset/setup the neurons states and the enviroment states. The enviroment is also reset in the step function
        """

        self._global_positions = torch.rand((self.num_unwrapped_agents,self.num_hidden_neurons_per_agent,self.position_dims))
        self._global_connections = torch.rand((self.num_unwrapped_agents,self.num_hidden_neurons_per_agent,self.num_hidden_neurons_per_agent))

        obs, _, self.other = self._env.reset()

        self.out_obs = self._convert_observation(obs)
        self.out_shared = self._convert_observation(obs)

        return self.out_obs, self.out_shared, self.other

    def step(self, actions: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """
        Takes a step in the enviroment and neron states. The actions are the output of the neurons and they are converted for the enviroment.
        """

        env_actions, neron_actions = self._convert_actions(actions)

        #TODO update neuron states
        #TODO find good way to 

        if neron_actions.shape[0] != self._global_connections.shape[0] or self._global_connections.shape[0] != self._global_connections.shape[1]:
            raise ValueError("Shapes of input arrays are not compatible.")

        weights_sum = self._global_connections.sum(axis=1, keepdims=True)
        if np.any(weights_sum == 0):
            raise ValueError("Weights cannot sum to zero for any row.")
        normalized_weights = self._global_connections / weights_sum

        new_neuron_observations = np.dot(normalized_weights, neron_actions)

        self.step_count += 1

        if self.step_count == self.speed_factor:
            obs, self.share_obs, rewards,dones, self.infos, self.available_actions = self._env.step(env_actions)
            self.step_count = 0
        
            self.out_obs = self._convert_observation(obs)
            self.out_shared = self._convert_observation(obs)

            self.rewards = np.repeat(rewards, repeats=self.total_neurons_per_agent, axis=1)
            self.dones = np.repeat(dones, repeats=self.total_neurons_per_agent, axis=1)

        return self.out_obs, self.out_shared, self.rewards, self.dones, self.infos, self.available_actions

    def state(self) -> torch.Tensor:
        pass

    def render(self, *args, **kwargs) -> Any:
        pass

    def close(self) -> None:
        pass
    
    @property
    def total_neurons_per_agent(self) -> int:
        return self.total_visible_neurons + self.num_hidden_neurons_per_agent
    
    @property
    def total_visible_neurons(self) -> int:
        return self.num_visible_input_neurons + self.num_visible_output_neurons

    @property
    def device(self) -> torch.device:
        pass

    @property
    def num_envs(self) -> int:
        pass

    @property
    def n_agents(self) -> int:
        return self.num_unwrapped_agents*self.total_neurons_per_agent

    @property
    def max_num_agents(self) -> int:
        pass

    @property
    def agents(self) -> Sequence[str]:
        pass

    @property
    def possible_agents(self) -> Sequence[str]:
        pass

    @property
    def observation_space(self) -> Mapping[int, gym.Space]:
        return self._observation_space

    @property
    def action_space(self) -> Mapping[int, gym.Space]:
        return self._action_space

    @property
    def share_observation_space(self) -> Mapping[int, gym.Space]:
        return self._shared_observation_space

    