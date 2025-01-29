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

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

#TODO create single agent test env
#TODO single agent case


class neuronWrapper(object):
    """
    This will transform a single agent or multi agent enviroment into a neuron multi agent enviroment.
    Where each agent in the unwapped enviroment is represented by many neuron agents.
    The neuron agents will choose connections to observation, action output, and other neurons.
    The neurons will be orginized in a ND grid and choose connections based on position in the grid.
    Neurons will be able to move through the grid, change connections, and process observations as their actions.
    Each agent will be trained individually using RL and will be unaware of the other agents state.

    (n,n) 0 is the neuron and 1 is the connection to other neurons
    """

    def __init__(self, env: Any, neuron_args: Any) -> None:
        self._env = env

        try:
            self._unwrapped = self._env.unwrapped
        except:
            self._unwrapped = env

        self.num_unwrapped_agents = self._unwrapped.n_agents
        self.position_dims = neuron_args["position_dims"]
        self.neuron_bandwidth = neuron_args["neuron_bandwidth"]
        self.neuron_input = self.neuron_bandwidth + self.position_dims
        self.neuron_output = self.neuron_bandwidth + 2*self.position_dims + 2
        self.speed_factor = neuron_args["speed_factor"]
        self.exp_decay_scale = neuron_args["exp_decay_scale"]
        self.max_connection_change = neuron_args["max_connection_change"]
        self.step_count = 0
        self.num_hidden_neurons_per_agent = neuron_args["num_hidden_neurons_per_agent"]
        self.reset_env_factor = neuron_args["reset_env_factor"]
        self.reset_env_factor_count = 0
        self.noise_scale = neuron_args["noise_scale"]
        self.max_velocity = neuron_args["max_velocity"]
        self.brain_size = neuron_args["brain_size"]

        self.enviroment_observation_space = self._env.observation_space
        self.enviroment_action_space = self._env.action_space

        self.num_visible_input_neurons = self.enviroment_observation_space[0].shape[0]//self.neuron_bandwidth + (1 if self.enviroment_observation_space[0].shape[0]%self.neuron_bandwidth != 0 else 0)
        self.num_visible_output_neurons = self.enviroment_action_space[0].shape[0]//self.neuron_bandwidth + (1 if self.enviroment_action_space[0].shape[0]%self.neuron_bandwidth != 0 else 0)

        self._observation_space = [spaces.Box(low=-1.0,high=1.0,shape=(self.neuron_input,),dtype=np.float32) for _ in range(self.n_agents)]
        self._action_space = [spaces.Box(low=-1.0,high=1.0,shape=(self.neuron_output,),dtype=np.float32) for _ in range(self.n_agents)]
        self._shared_observation_space = [spaces.Box(low=-1.0,high=1.0,shape=(self.neuron_input,),dtype=np.float32) for _ in range(self.n_agents)]

    def _reshape_obs(self, obs: np.ndarray, prev_state: np.ndarray) -> np.ndarray:

        size = self.enviroment_observation_space[0].shape[0]

        batch = obs.shape[0]
        
        out = prev_state.reshape(batch, self.n_agents, self.neuron_input)

        for j in range(self.num_unwrapped_agents):

            out_slice = out[:, j * self.total_neurons_per_agent : j * self.total_neurons_per_agent + self.num_visible_input_neurons, :self.neuron_bandwidth]
            out_slice_flat = out_slice.reshape(batch, self.num_visible_input_neurons * self.neuron_bandwidth)

            out_slice_flat[:, :size] = obs[:, j, :]
        
        return out
    
    def _reshape_actions(self, actions: np.ndarray) -> np.ndarray:

        size = self.enviroment_action_space[0].shape[0]

        batch = actions.shape[0]
        
        out = np.zeros((batch, self.num_unwrapped_agents, size + (self.neuron_input-size%self.neuron_input)), dtype=actions.dtype)
        
        for j in range(self.num_unwrapped_agents):
            out_slice = out[:, j : j+1, :]
            
            out_slice_flat = out_slice.reshape(batch, self.num_visible_output_neurons, self.neuron_input)

            act_pos = j*self.total_neurons_per_agent+self.num_visible_input_neurons

            out_slice_flat[:,:,:] = actions[:, act_pos:act_pos+self.num_visible_output_neurons,:]
        
        return out[:,:, :size]

    def _convert_observation(self, obs, prev_state) -> Tuple[np.ndarray, np.ndarray, Any]:
        """
        Converts the observation from the enviroment to the observation for the neurons
        """
        return self._reshape_obs(np.array(obs), prev_state)
    
    def _convert_actions(self, actions) -> Tuple[np.ndarray, np.ndarray, Any]:
        """
        Converts the action from the enviroment to the actions for the neurons
        """

        env_actions = actions[:,:,:self.neuron_input]
        neron_actions = actions[:,:,:self.neuron_output]

        env_actions = self._reshape_actions(np.array(env_actions))

        return env_actions, neron_actions

    def reset(self) -> Tuple[np.ndarray, np.ndarray, Any]:
        """
        Will reset/setup the neurons states and the enviroment states. The enviroment is also reset in the step function
        """

        obs, _, other = self._env.reset()

        self._global_positions = np.random.random((obs.shape[0],self.num_unwrapped_agents,self.total_neurons_per_agent,self.position_dims))
        self._global_connections = np.random.random((obs.shape[0],self.num_unwrapped_agents,self.total_neurons_per_agent,self.total_neurons_per_agent))*self.noise_scale

        inital_state = np.random.random((obs.shape[0],self.num_unwrapped_agents,self.total_neurons_per_agent,self.neuron_input))

        self.out_obs = self._convert_observation(obs, inital_state)
        self.out_shared = self._convert_observation(obs, inital_state)

        self.rewards = np.zeros((obs.shape[0],self.n_agents,1))
        self.infos = [[{} for __ in range(self.num_unwrapped_agents)] for _ in range(obs.shape[0])]
        self.available_actions = [None for _ in range(obs.shape[0])]

        return self.out_obs, self.out_shared, other

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any]:
        """
        Takes a step in the enviroment and neron states. The actions are the output of the neurons and they are converted for the enviroment.
        """

        env_actions, neron_actions = self._convert_actions(actions)

        neron_actions = np.stack(np.split(neron_actions, self.num_unwrapped_agents, axis=1))
        neron_actions = neron_actions.reshape(neron_actions.shape[1],neron_actions.shape[0], neron_actions.shape[2], neron_actions.shape[3])

        neron_signal = neron_actions[...,:self.neuron_bandwidth]
        neron_connect_positions = neron_actions[...,self.neuron_bandwidth:self.neuron_bandwidth+self.position_dims]
        neron_weight_scale = neron_actions[...,self.neuron_bandwidth+self.position_dims:self.neuron_bandwidth+self.position_dims+1]
        neron_velocity_direction = neron_actions[...,self.neuron_bandwidth+self.position_dims+1:self.neuron_bandwidth+2*self.position_dims+1]
        neron_velocity_scale = neron_actions[...,self.neuron_bandwidth+2*self.position_dims+1:self.neuron_bandwidth+2*self.position_dims+2]

        neron_velocity_direction = neron_velocity_direction/np.linalg.norm(neron_velocity_direction, axis=-1, keepdims=True)
        neron_velocity = neron_velocity_direction*neron_velocity_scale*  self.max_velocity

        neuron_weight_scale = np.repeat(neron_weight_scale, self.total_neurons_per_agent, axis=3)

        weights_sum = np.repeat(self._global_connections.sum(axis=3, keepdims=True), self.total_neurons_per_agent, axis=3)
        weights_sum[weights_sum == 0] = 1
        normalized_weights = self._global_connections / weights_sum

        new_neuron_signals = np.matmul(normalized_weights, neron_signal)

        pairwise_distances = self._pairwise_distances(neron_connect_positions,self._global_positions)

        pairwise_weights = self._remap_distances(pairwise_distances)

        weights_delta = pairwise_weights * neuron_weight_scale * self.max_connection_change

        self._global_positions += neron_velocity

        self._global_positions = np.clip(self._global_positions, -self.brain_size, self.brain_size)

        self._global_connections += weights_delta

        new_neuron_input = np.concatenate((new_neuron_signals, self._global_positions), axis=-1)

        self.step_count += 1

        if self.step_count == self.speed_factor:
            self.step_count = 0

            obs, self.share_obs, rewards, dones, self.infos, self.available_actions = self._env.step(env_actions)

            self.rewards = np.repeat(rewards, repeats=self.total_neurons_per_agent, axis=1)

            if dones.any():
                obs, _, other = self._env.reset()
                self.reset_env_factor_count += 1

                if self.reset_env_factor_count == self.reset_env_factor:
                    dones = np.ones_like(self.rewards).squeeze(axis=-1)
                    self.reset_env_factor_count = 0
                else:
                    dones = np.zeros_like(self.rewards).squeeze(axis=-1)
            else:
                dones = np.zeros_like(self.rewards).squeeze(axis=-1)

            self.out_obs = self._convert_observation(obs,new_neuron_input)
            self.out_shared = self._convert_observation(obs,new_neuron_input)

            return self.out_obs, self.out_shared, self.rewards, dones, self.infos, self.available_actions
        
        else:

            out_obs = new_neuron_input.reshape(new_neuron_input.shape[0], self.n_agents, self.neuron_input)
            out_obs_shared = new_neuron_input.reshape(new_neuron_input.shape[0], self.n_agents, self.neuron_input)

            return out_obs, out_obs_shared, np.zeros_like(self.rewards), np.zeros_like(self.rewards).squeeze(axis=-1), self.infos, self.available_actions
        
    
    def _pairwise_distances(self, array1, array2):

        diff = array1[..., :, np.newaxis, :] - array2[..., np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=-1))
        return distances

    def _remap_distances(self, distances, scale=1.0):

        remapped = np.exp(-distances / scale)
        return remapped

    def state(self) -> np.ndarray:
        pass

    def render(self, mode='human'):
        """
        Visualizes the neurons, their positions, and connections in a 2D grid.
        """

        # Prepare the data for visualization
        positions = self._global_positions[0]  # Using the first environment in the batch
        connections = self._global_connections[0]  # Using the first environment in the batch

        # Create a 2D scatter plot for neuron positions
        fig, ax = plt.subplots(figsize=(8, 8))
        for agent_idx, agent_positions in enumerate(positions):
            # Extract neuron positions for this agent
            neuron_positions = agent_positions[:, :2]  # Assuming 2D positions

            # Plot neuron positions
            ax.scatter(
                neuron_positions[:, 0],
                neuron_positions[:, 1],
                label=f"Agent {agent_idx + 1}",
                alpha=0.7,
            )

            # Create lines to represent connections
            line_segments = []
            for i, start_pos in enumerate(neuron_positions):
                for j, weight in enumerate(connections[agent_idx, i]):
                    if weight > 0.1:  # Visualize only significant connections
                        end_pos = neuron_positions[j]
                        line_segments.append([start_pos, end_pos])

            # Add the connection lines to the plot
            lc = LineCollection(
                line_segments,
                linewidths=0.5,
                alpha=0.3,
                colors="gray",
            )
            ax.add_collection(lc)

        ax.set_title("Neuron Network Visualization")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.legend()
        ax.grid(True)
        ax.set_xlim(-self.brain_size, self.brain_size)
        ax.set_ylim(-self.brain_size, self.brain_size)

        if mode == "human":
            plt.show()
        elif mode == "rgb_array":
            # Return an RGB array for rendering
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            return image
        else:
            raise NotImplementedError(f"Render mode {mode} is not supported.")

    def close(self) -> None:
        self._env.close()
    
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



