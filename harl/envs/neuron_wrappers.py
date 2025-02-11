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

#TODO Each agent gets own critic


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
        self.movment_change_penalty = neuron_args["movment_change_penalty"] if "movment_change_penalty" in neuron_args else 0
        self.connection_change_penalty = neuron_args["connection_change_penalty"] if "connection_change_penalty" in neuron_args else 0

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
        
        out = np.zeros((batch, self.num_unwrapped_agents, self.num_visible_output_neurons*self.neuron_bandwidth), dtype=actions.dtype)
        
        for j in range(self.num_unwrapped_agents):
            out_slice = out[:, j : j+1, :]
            
            out_slice_flat = out_slice.reshape(batch, self.num_visible_output_neurons, self.neuron_bandwidth)

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

        env_actions = actions[:,:,:self.neuron_bandwidth]
        neron_actions = actions[:,:,:self.neuron_output]

        env_actions = self._reshape_actions(np.array(env_actions))

        return env_actions, neron_actions

    def reset(self) -> Tuple[np.ndarray, np.ndarray, Any]:
        """
        Will reset/setup the neurons states and the enviroment states. The enviroment is also reset in the step function
        """

        obs, _, other = self._env.reset()

        obs = np.array(obs)

        if len(obs.shape) == 2:
            obs = obs.reshape(1,obs.shape[0],obs.shape[1])

        self._global_positions = (2*np.random.random((obs.shape[0],self.num_unwrapped_agents,self.total_neurons_per_agent,self.position_dims))-1)
        self._global_connections = (2*np.random.random((obs.shape[0],self.num_unwrapped_agents,self.total_neurons_per_agent,self.total_neurons_per_agent))-1)*self.noise_scale

        inital_state = np.random.random((obs.shape[0],self.num_unwrapped_agents,self.total_neurons_per_agent,self.neuron_input))

        self.out_obs = self._convert_observation(obs, inital_state)
        self.out_shared = self._convert_observation(obs, inital_state)

        if self.out_obs.shape[0] == 1:
            self.rewards = np.zeros((self.n_agents,1))
            self.infos = [{} for _ in range(self.n_agents)]
            self.available_actions = None
            return self.out_obs[0], self.out_shared[0], other
        else:
            self.rewards = np.zeros((obs.shape[0],self.n_agents,1))
            self.infos = [[{} for __ in range(self.n_agents)] for _ in range(obs.shape[0])]
            self.available_actions = [None for _ in range(obs.shape[0])]

            return self.out_obs, self.out_shared, other

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any]:
        """
        Takes a step in the enviroment and neron states. The actions are the output of the neurons and they are converted for the enviroment.
        """

        if len(actions.shape) == 2:
            actions = actions.reshape(1,actions.shape[0],actions.shape[1])

        env_actions, neron_actions = self._convert_actions(actions)

        neron_actions = np.stack(np.split(neron_actions, self.num_unwrapped_agents, axis=1))
        neron_actions = neron_actions.reshape(neron_actions.shape[1],neron_actions.shape[0], neron_actions.shape[2], neron_actions.shape[3])

        neron_signal = neron_actions[...,:self.neuron_bandwidth]
        neron_connect_positions = neron_actions[...,self.neuron_bandwidth:self.neuron_bandwidth+self.position_dims]
        neron_weight_scale = neron_actions[...,self.neuron_bandwidth+self.position_dims:self.neuron_bandwidth+self.position_dims+1]
        neron_velocity_direction = neron_actions[...,self.neuron_bandwidth+self.position_dims+1:self.neuron_bandwidth+2*self.position_dims+1]
        neron_velocity_scale = neron_actions[...,self.neuron_bandwidth+2*self.position_dims+1:self.neuron_bandwidth+2*self.position_dims+2]

        neron_velocity_direction = neron_velocity_direction/np.linalg.norm(neron_velocity_direction, axis=-1, keepdims=True)
        neron_velocity = neron_velocity_direction*neron_velocity_scale * self.max_velocity

        neron_velocity_penalty = np.linalg.norm(neron_velocity, axis=-1, keepdims=True) / self.max_velocity * self.movment_change_penalty

        neuron_weight_scale = np.repeat(neron_weight_scale, self.total_neurons_per_agent, axis=3)

        weights_sum = np.repeat(self._global_connections.sum(axis=3, keepdims=True), self.total_neurons_per_agent, axis=3)
        weights_sum[weights_sum == 0] = 1
        normalized_weights = self._global_connections / weights_sum

        new_neuron_signals = np.matmul(normalized_weights, neron_signal)

        pairwise_distances = self._pairwise_distances(neron_connect_positions,self._global_positions)

        pairwise_weights = self._remap_distances(pairwise_distances, scale=self.exp_decay_scale)

        weights_delta = pairwise_weights * neuron_weight_scale * self.max_connection_change

        weights_delta_penalty = np.abs(np.mean(weights_delta,-1)) / self.max_connection_change * self.connection_change_penalty

        self._global_positions += neron_velocity

        self._global_positions = np.clip(self._global_positions, -self.brain_size, self.brain_size)

        self._global_connections += weights_delta

        new_neuron_input = np.concatenate((new_neuron_signals, self._global_positions), axis=-1)

        self.step_count += 1

        rewards_modifier = np.zeros_like(self.rewards)#self.rewards*neron_velocity_penalty.reshape(self.rewards.shape) + self.rewards*weights_delta_penalty.reshape(self.rewards.shape)

        if self.step_count == self.speed_factor:
            self.step_count = 0

            if env_actions.shape[0] == 1:
                env_actions = env_actions[0]

            obs, self.share_obs, rewards, dones, _, self.available_actions = self._env.step(env_actions)

            obs = np.array(obs)
            rewards = np.array(rewards)
            dones = np.array(dones)

            self.rewards = np.repeat(rewards, repeats=self.total_neurons_per_agent, axis=-2)

            self.rewards -= rewards_modifier

            if dones.any():
                obs, _, other = self._env.reset()
                obs = np.array(obs)
                self.reset_env_factor_count += 1

                if self.reset_env_factor_count == self.reset_env_factor:
                    dones = np.ones_like(self.rewards).squeeze(axis=-1)
                    self.reset_env_factor_count = 0
                else:
                    dones = np.zeros_like(self.rewards).squeeze(axis=-1)
            else:
                dones = np.zeros_like(self.rewards).squeeze(axis=-1)

            if len(obs.shape) == 2:
                obs = obs.reshape(1,obs.shape[0],obs.shape[1])

            self.out_obs = self._convert_observation(obs,new_neuron_input)
            self.out_shared = self._convert_observation(obs,new_neuron_input)

            if self.out_obs.shape[0] == 1:
                return self.out_obs[0], self.out_shared[0], self.rewards, dones, self.infos, self.available_actions

            return self.out_obs, self.out_shared, self.rewards, dones, self.infos, self.available_actions
        
        else:

            out_obs = new_neuron_input.reshape(new_neuron_input.shape[0], self.n_agents, self.neuron_input)
            out_obs_shared = new_neuron_input.reshape(new_neuron_input.shape[0], self.n_agents, self.neuron_input)

            if out_obs.shape[0] == 1:
                return out_obs[0], out_obs_shared[0], np.zeros_like(self.rewards), np.zeros_like(self.rewards).squeeze(axis=-1), self.infos, self.available_actions

            return out_obs, out_obs_shared, -rewards_modifier, np.zeros_like(self.rewards).squeeze(axis=-1), self.infos, self.available_actions
    
    def _pairwise_distances(self, array1, array2):

        diff = array1[..., :, np.newaxis, :] - array2[..., np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=-1))
        return distances

    def _remap_distances(self, distances, scale=.1):

        remapped = np.exp(-distances / (scale*self.brain_size*2))
        return remapped

    def state(self) -> np.ndarray:
        pass

    def render(self, mode='human'):
        self._env.render()
        if not hasattr(self, "_fig") or self._fig is None:
            # Create one row of subplots, one column per agent
            self._fig, self._axes = plt.subplots(
                1,
                self.num_unwrapped_agents,
                figsize=(6 * self.num_unwrapped_agents, 6),  # Adjust size if you like
            )

            self._colors = plt.cm.get_cmap("tab10", self.num_unwrapped_agents)
            # If there's only one agent, self._axes is not an iterable array by default
            if self.num_unwrapped_agents == 1:
                self._axes = [self._axes]

            # Turn interactive mode on so that we can update the figure without blocking
            plt.ion()

        # Clear axes from any previous drawing
        for ax in self._axes:
            ax.clear()

        # -------------------------------------------------------------------------
        # 2) PREPARE NEURON DATA
        #    We show data from the first environment in the batch (index=0).
        # -------------------------------------------------------------------------
        positions = self._global_positions[0]     # Shape: [num_agents, total_neurons_per_agent, 2 or 3...]
        connections = self._global_connections[0] # Shape: [num_agents, total_neurons_per_agent, total_neurons_per_agent]

        # -------------------------------------------------------------------------
        # 3) PLOT EACH AGENT IN ITS OWN SUBPLOT
        # -------------------------------------------------------------------------
        for agent_idx in range(self.num_unwrapped_agents):
            ax = self._axes[agent_idx]
            # Extract 2D (x, y) positions of this agent's neurons
            neuron_positions = positions[agent_idx, :, :2]  # shape: [num_neurons, 2]
            agent_connections = connections[agent_idx]      # shape: [num_neurons, num_neurons]

            
            # Scatter plot of hidden neuron positions
            ax.scatter(
                neuron_positions[self.num_visible_input_neurons+self.num_visible_output_neurons:, 0],
                neuron_positions[self.num_visible_input_neurons+self.num_visible_output_neurons:, 1],
                label="Hidden Neurons",
                alpha=0.7,
                color=self._colors(agent_idx),
                marker="o",
            )

             # Scatter plot of input neuron positions
            ax.scatter(
                neuron_positions[:self.num_visible_input_neurons, 0],
                neuron_positions[:self.num_visible_input_neurons, 1],
                label="Input Neurons",
                alpha=0.7,
                color=self._colors(agent_idx),
                marker="+",
                s=100
            )

            # Scatter plot of output neuron positions
            ax.scatter(
                neuron_positions[self.num_visible_input_neurons:self.num_visible_input_neurons+self.num_visible_output_neurons, 0],
                neuron_positions[self.num_visible_input_neurons:self.num_visible_input_neurons+self.num_visible_output_neurons, 1],
                label="Output Neurons",
                alpha=0.7,
                color=self._colors(agent_idx),
                marker="x",
                s=80
            )


            # Create line segments for connections above a certain weight threshold
            line_segments = []
            colors = []
            alpha = []
            linewidth = []
            weight_color_map = plt.cm.get_cmap("coolwarm")
            for i, start_pos in enumerate(neuron_positions):
                for j, weight in enumerate(agent_connections[i]):
                    end_pos = neuron_positions[j]
                    line_segments.append([start_pos, end_pos])
                    alpha.append(np.clip(abs(weight)**2,0,1))
                    linewidth.append((abs(weight)+.1)**2)
                    colors.append(weight_color_map(weight))


            # Add these connection segments to the subplot
            lc = LineCollection(
                line_segments,
                linewidths=linewidth,
                alpha=alpha,
                color=colors,
            )
            ax.add_collection(lc)

            # Set subplot titles, labels, limits
            ax.set_title(f"Neuron Network (Agent {agent_idx+1})")
            ax.set_xlabel("X Position")
            ax.set_ylabel("Y Position")
            ax.grid(True)
            ax.set_xlim(-self.brain_size, self.brain_size)
            ax.set_ylim(-self.brain_size, self.brain_size)
            ax.legend()

        # -------------------------------------------------------------------------
        # 4) RENDER MODE LOGIC
        # -------------------------------------------------------------------------
        if mode == "human":
            # Draw/update the figure without blocking
            self._fig.canvas.draw()
            self._fig.canvas.flush_events()
            plt.pause(0.001)  # A tiny pause so the figure updates
        elif mode == "rgb_array":
            # Return an RGB array for rendering
            self._fig.canvas.draw()
            width, height = self._fig.canvas.get_width_height()
            image = np.frombuffer(self._fig.canvas.tostring_rgb(), dtype=np.uint8)
            image = image.reshape((height, width, 3))
            return image
        else:
            raise NotImplementedError(f"Render mode '{mode}' is not supported.")

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


