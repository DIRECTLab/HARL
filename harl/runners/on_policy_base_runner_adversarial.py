"""Base runner for on-policy algorithms."""

import time
import numpy as np
import torch
import setproctitle
from harl.common.valuenorm import ValueNorm
from harl.common.buffers.on_policy_actor_buffer import OnPolicyActorBuffer
from harl.common.buffers.on_policy_critic_buffer_ep import OnPolicyCriticBufferEP
from harl.common.buffers.on_policy_critic_buffer_fp import OnPolicyCriticBufferFP
from harl.algorithms.actors import ALGO_REGISTRY
from harl.algorithms.critics.v_critic import VCritic
from harl.algorithms.critics.v_critic_adv import VCriticAdv
from harl.utils.trans_tools import _t2n
from harl.utils.envs_tools import set_seed, get_num_agents, make_render_env, make_eval_env, make_train_env 
from harl.utils.configs_tools import init_dir, save_config, get_task_name
from harl.utils.models_tools import init_device
from harl.envs import LOGGER_REGISTRY
import os
from pathlib import Path


class OnPolicyBaseRunnerAdversarial:
    """Base runner for on-policy algorithms."""

    def __init__(self, args, algo_args, env_args):
        """Initialize the OnPolicyBaseRunnerAdversarial class.
        Args:
            args: command-line arguments parsed by argparse. Three keys: algo, env, exp_name.
            algo_args: arguments related to algo, loaded from config file and updated with unparsed command-line arguments.
            env_args: arguments related to env, loaded from config file and updated with unparsed command-line arguments.
        """

        self.args = args
        self.algo_args = algo_args
        self.env_args = env_args
        self.best_avg_reward = -torch.inf
        self.hidden_sizes = algo_args["model"]["hidden_sizes"]
        self.hidden_sizes_critic = algo_args["model"]["hidden_sizes"]
        self.rnn_hidden_size = self.hidden_sizes[-1]
        self.rnn_hidden_size_critic = self.hidden_sizes_critic[-1]
        self.recurrent_n = algo_args["model"]["recurrent_n"]
        self.action_aggregation = algo_args["algo"]["action_aggregation"]
        self.state_type = env_args.get("state_type", "EP")
        self.share_param = algo_args["algo"]["share_param"]
        self.fixed_order = algo_args["algo"]["fixed_order"]
        set_seed(algo_args["seed"])
        self.device = init_device(algo_args["device"])
        if not self.algo_args["render"]["use_render"]:  # train, not render
            self.run_dir, self.log_dir, self.save_dir, self.writter = init_dir(
                args["env"],
                env_args,
                args["algo"],
                args["exp_name"],
                algo_args["seed"]["seed"],
                logger_path=algo_args["logger"]["log_dir"],
            )
            save_config(args, algo_args, env_args, self.run_dir)
        # set the title of the process
        setproctitle.setproctitle(
            str(args["algo"]) + "-" + str(args["env"]) + "-" + str(args["exp_name"])
        )
        # set the config of env
        if self.algo_args["render"]["use_render"]:  # make envs for rendering
            (
                self.env,
                self.manual_render,
                self.manual_expand_dims,
                self.manual_delay,
                self.env_num,
            ) = make_render_env(args["env"], algo_args["seed"]["seed"], env_args)
        else:  # make envs for training and evaluation
            self.env = make_train_env(
                args["env"],
                algo_args["seed"]["seed"],
                algo_args["train"]["n_rollout_threads"],
                env_args,
            )
            self.eval_envs = (
                make_eval_env(
                    args["env"],
                    algo_args["seed"]["seed"],
                    algo_args["eval"]["n_eval_rollout_threads"],
                    env_args,
                )
                if algo_args["eval"]["use_eval"]
                else None
            )
            
        self.num_agents = get_num_agents(args["env"], env_args, self.env)

        print("share_observation_space: ", self.env.share_observation_space)
        print("observation_space: ", self.env.observation_space)
        print("action_space: ", self.env.action_space)

        self.is_heter_action_space = False
        self.max_action_space = 0

        first_act_space = self.env.action_space[0]
        for key, val in self.env.action_space.items():
            if val.shape[0] > self.max_action_space:
                self.max_action_space =  val.shape[0] 
            if val != first_act_space and not self.is_heter_action_space:
                self.is_heter_action_space = True

        # actor
        if self.share_param:
            self.actor = []
            agent = ALGO_REGISTRY[args["algo"]](
                {**algo_args["model"], **algo_args["algo"]},
                self.env.observation_space[0],
                self.env.action_space[0],
                device=self.device,
            )
            self.actor.append(agent)
            for agent_id in range(1, self.num_agents):
                assert (
                    self.env.observation_space[agent_id]
                    == self.env.observation_space[0]
                ), "Agents have heterogeneous observation spaces, parameter sharing is not valid."
                assert (
                    self.env.action_space[agent_id] == self.env.action_space[0]
                ), "Agents have heterogeneous action spaces, parameter sharing is not valid."
                self.actor.append(self.actor[0])
        else:
            self.actor = []
            for agent_id in range(self.num_agents):
                agent = ALGO_REGISTRY[args["algo"]](
                    {**algo_args["model"], **algo_args["algo"]},
                    self.env.observation_space[agent_id],
                    self.env.action_space[agent_id],
                    device=self.device,
                )
                self.actor.append(agent)
        algo_args["model"]["hidden_sizes"] = self.hidden_sizes_critic
        if self.algo_args["render"]["use_render"] is False:  # train, not render
            self.actor_buffer = []
            for agent_id in range(self.num_agents):
                ac_bu = OnPolicyActorBuffer(
                    {**algo_args["train"], **algo_args["model"]},
                    self.env.observation_space[agent_id],
                    self.env.action_space[agent_id],
                )
                self.actor_buffer.append(ac_bu)

            share_observation_space = self.env.share_observation_space
            
            self.critics = {team : VCriticAdv(
                {**algo_args["model"], **algo_args["algo"]},
                share_observation_space[team],
                device=self.device,
            ) for team, _ in self.env.unwrapped.cfg.teams.items()}

            if self.state_type == "EP":
                # EP stands for Environment Provided, as phrased by MAPPO paper.
                # In EP, the global states for all agents are the same.
                self.critic_buffers = {}
                for team, _ in self.env.unwrapped.cfg.teams.items():
                    self.critic_buffers[team] = OnPolicyCriticBufferEP(
                        {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
                        share_observation_space[team],
                    )

            elif self.state_type == "FP":
                # FP stands for Feature Pruned, as phrased by MAPPO paper.
                # In FP, the global states for all agents are different, and thus needs the dimension of the number of agents.
                self.critic_buffer = OnPolicyCriticBufferFP(
                    {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
                    share_observation_space,
                    self.num_agents,
                )
            else:
                raise NotImplementedError

            if self.algo_args["train"]["use_valuenorm"] is True:
                self.value_normalizer = ValueNorm(1, device=self.device)
            else:
                self.value_normalizer = None
            
            self.logger = LOGGER_REGISTRY[args["env"]](
                args, algo_args, env_args, self.num_agents, self.writter, self.run_dir
            )
        self.algo_args["model"]["hidden_sizes"] = self.hidden_sizes
        if self.algo_args["train"]["model_dir"] is not None:  # restore model
            self.restore()

    def run(self):
        """Run the training (or rendering) pipeline."""
        if self.algo_args["render"]["use_render"] is True:
            self.render()
            return
        print("start running")
        
        self.warmup()

        episodes = (
            int(self.algo_args["train"]["num_env_steps"])
            // self.algo_args["train"]["episode_length"]
            // self.algo_args["train"]["n_rollout_threads"]
        )

        self.logger.init(episodes)  # logger callback at the beginning of training

        for episode in range(1, episodes + 1):
            if self.algo_args["train"][
                "use_linear_lr_decay"
            ]:  # linear decay of learning rate
                if self.share_param:
                    self.actor[0].lr_decay(episode, episodes)
                else:
                    for agent_id in range(self.num_agents):
                        self.actor[agent_id].lr_decay(episode, episodes)
                # self.critic.lr_decay(episode, episodes)
                for _, critic in self.critics.items():
                    critic.lr_decay(episode, episodes)

            self.logger.episode_init(
                episode
            )  # logger callback at the beginning of each episode
            with torch.inference_mode():
                self.prep_rollout()  # change to eval mode
                for step in range(self.algo_args["train"]["episode_length"]):
                    # Sample actions from actors and values from critics
                    (
                        values,
                        actions,
                        action_log_probs,
                        rnn_states,
                        rnn_states_critic,
                    ) = self.collect(step)
                    # actions: (n_threads, n_agents, action_dim)
                    if self.not_tensor:
                        (
                            obs,
                            share_obs,
                            rewards,
                            dones,
                            infos,
                            available_actions,
                        ) = self.env.step(actions.cpu().numpy())

                        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
                        share_obs = torch.tensor(share_obs, dtype=torch.float32, device=self.device)
                        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
                        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

                    else:
                        (
                            obs,
                            share_obs,
                            rewards,
                            dones,
                            infos,
                            available_actions,
                        ) = self.env.step(actions)
                    
                    # obs: (n_threads, n_agents, obs_dim)
                    # share_obs: (n_threads, n_agents, share_obs_dim)
                    # rewards: (n_threads, n_agents, 1)
                    # dones: (n_threads, n_agents)
                    # infos: (n_threads)
                    # available_actions: (n_threads, ) of None or (n_threads, n_agents, action_number)
                    data = (
                        obs,
                        share_obs,
                        rewards,
                        dones,
                        infos,
                        available_actions,
                        values,
                        actions,
                        action_log_probs,
                        rnn_states,
                        rnn_states_critic,
                    )
                    if hasattr(self.env, "log_info"):
                        self.logger.per_step(self.env.log_info)  # logger callback at each step
                    else:
                        a_infos = {}
                        for info in infos:
                            for i in range(len(info)):
                                for key, val in info[i].items():
                                    if "reward" in key:
                                        if key not in a_infos:
                                            a_infos[key] = []
                                        a_infos[key].append(val)
                                
                        for key, val in a_infos.items():
                            a_infos[key] = np.mean(val)
                        self.logger.per_step(a_infos)

                    self.insert(data)  # insert data into buffer

            # compute return and update network
            self.compute()
            self.prep_training()  # change to train mode

            actor_train_infos, critic_train_info = self.train()
            
            # log information
            if episode % self.algo_args["train"]["log_interval"] == 0:
                self.logger.episode_log(
                    actor_train_infos,
                    critic_train_info,
                    self.actor_buffer,
                    self.critic_buffer,
                )


            
            if hasattr(self.logger,"total_reward") and self.logger.total_reward > self.best_avg_reward:
                self.best_avg_reward = self.logger.total_reward
                self.save(Path(Path(self.save_dir).parent, 'best_model'))
            
            # eval
            if episode % self.algo_args["train"]["eval_interval"] == 0:
                if self.algo_args["eval"]["use_eval"]:
                    self.prep_rollout()
                    self.eval()
                self.save(self.save_dir)

            self.after_update()

    def warmup(self):
        """Warm up the replay buffer."""
        # reset env
        with torch.inference_mode():
            obs, share_obs, available_actions = self.env.reset()
            #check if obs is already a torch tensor
            if not isinstance(obs, torch.Tensor):
                obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
                share_obs = torch.tensor(share_obs, dtype=torch.float32, device=self.device)
                self.not_tensor = True
            else:
                self.not_tensor = False
            # replay buffer
            for agent_id in range(self.num_agents):
                obs_space = self.env.observation_space[agent_id].shape[0]
                self.actor_buffer[agent_id].obs[0] = obs[:, agent_id, :obs_space].clone()
                if self.actor_buffer[agent_id].available_actions is not None:
                    self.actor_buffer[agent_id].available_actions[0] = available_actions[
                        :, agent_id
                    ].clone()

            if self.state_type == "EP":
                for team, _ in self.critics.items():
                    self.critic_buffers[team].share_obs = share_obs.clone()

                self.critic_buffer.share_obs[0] = share_obs[:, 0].clone()
            elif self.state_type == "FP":
                self.critic_buffer.share_obs[0] = share_obs.clone()

    def collect(self, step):
        """Collect actions and values from actors and critics.
        Args:
            step: step in the episode.
        Returns:
            values, actions, action_log_probs, rnn_states, rnn_states_critic
        """
        # collect actions, action_log_probs, rnn_states from n actors
        with torch.inference_mode():
            action_collector = []
            action_log_prob_collector = []
            rnn_state_collector = []
            for agent_id in range(self.num_agents):
                action, action_log_prob, rnn_state = self.actor[agent_id].get_actions(
                    self.actor_buffer[agent_id].obs[step],
                    self.actor_buffer[agent_id].rnn_states[step],
                    self.actor_buffer[agent_id].masks[step],
                    self.actor_buffer[agent_id].available_actions[step]
                    if self.actor_buffer[agent_id].available_actions is not None
                    else None,
                )
                action_collector.append(action)
                action_log_prob_collector.append(action_log_prob)
                rnn_state_collector.append(rnn_state)
            # (n_agents, n_threads, dim) -> (n_threads, n_agents, dim)

            if self.is_heter_action_space:
                for i in range(len(action_collector)):
                    pad_diff = self.max_action_space - action_collector[i].shape[1]
                    if pad_diff > 0:
                        action_collector[i] = torch.nn.functional.pad(action_collector[i], pad=(0, pad_diff), mode="constant", value=0)
                        action_log_prob_collector[i] = torch.nn.functional.pad(action_log_prob_collector[i], pad=(0, pad_diff), mode="constant", value=0)
            
            actions = torch.stack(action_collector).permute(1, 0, 2).contiguous()
            # actions = torch.tensor(action_collector).permute(1, 0, 2)
            action_log_probs = torch.stack(action_log_prob_collector).permute(1, 0, 2).contiguous()
            rnn_states = torch.stack(rnn_state_collector).permute(1, 0, 2, 3).contiguous()
            values = []
            rnn_states_critic = []
            if self.state_type == "EP":
                for _, critic in self.critics.items():
                    # EP stands for Environment Provided, as phrased by MAPPO paper.
                    # In EP, the global states for all agents are the same.
                    # (n_threads, dim)
                    value, rnn_state_critic = critic.get_values(
                        self.critic_buffer.share_obs[step],
                        self.critic_buffer.rnn_states_critic[step],
                        self.critic_buffer.masks[step],
                    )
                    values.append(value)
                    rnn_states_critic.append(rnn_state_critic)
                
                values = torch.stack(values).permute(1, 0, 2).contiguous()
                rnn_states_critic = torch.stack(rnn_states_critic).permute(1, 0, 2, 3).contiguous()

            elif self.state_type == "FP":
                value, rnn_state_critic = self.critic.get_values(
                    torch.concatenate(self.critic_buffer.share_obs[step]),
                    torch.concatenate(self.critic_buffer.rnn_states_critic[step]),
                    torch.concatenate(self.critic_buffer.masks[step]),
                )  # concatenate (n_threads, n_agents, dim) into (n_threads * n_agents, dim)
                # split (n_threads * n_agents, dim) into (n_threads, n_agents, dim)
                values = torch.tensor(
                    torch.split(value), self.algo_args["train"]["n_rollout_threads"]
                )
                rnn_states_critic = torch.tensor(
                    torch.split(
                        rnn_state_critic), self.algo_args["train"]["n_rollout_threads"]
                    
                )

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        """Insert data into buffer."""
        (
            obs,  # (n_threads, n_agents, obs_dim)
            share_obs,  # (n_threads, n_agents, share_obs_dim)
            rewards,  # (n_threads, n_agents, 1)
            dones,  # (n_threads, n_agents)
            infos,  # type: list, shape: (n_threads, n_agents)
            available_actions,  # (n_threads, ) of None or (n_threads, n_agents, action_number)
            values,  # EP: (n_threads, dim), FP: (n_threads, n_agents, dim)
            actions,  # (n_threads, n_agents, action_dim)
            action_log_probs,  # (n_threads, n_agents, action_dim)
            rnn_states,  # (n_threads, n_agents, dim)
            rnn_states_critic,  # EP: (n_threads, dim), FP: (n_threads, n_agents, dim)
        ) = data

        dones_env = torch.all(dones, axis=1)  # if all agents are done, then env is done
        rnn_states[
            dones_env == True
        ] = torch.zeros(  # if env is done, then reset rnn_state to all zero
            (
                (dones_env == True).sum(),
                self.num_agents,
                self.recurrent_n,
                self.rnn_hidden_size,
            ),
            dtype=torch.float32, device=self.device
        )

        # If env is done, then reset rnn_state_critic to all zero
        if self.state_type == "EP":
            rnn_states_critic[dones_env == True] = torch.zeros(
                ((dones_env == True).sum(), self.recurrent_n, self.rnn_hidden_size_critic),
                dtype=torch.float32, device=self.device
            )
        elif self.state_type == "FP":
            rnn_states_critic[dones_env == True] = torch.zeros(
                (
                    (dones_env == True).sum(),
                    self.num_agents,
                    self.recurrent_n,
                    self.rnn_hidden_size_critic,
                ),
                dtype=torch.float32, device=self.device
            )

        # masks use 0 to mask out threads that just finish.
        # this is used for denoting at which point should rnn state be reset
        masks = torch.ones(
            (self.algo_args["train"]["n_rollout_threads"], self.num_agents, 1),
            dtype=torch.float32, device=self.device
        )
        masks[dones_env == True] = torch.zeros(
            ((dones_env == True).sum(), self.num_agents, 1), dtype=torch.float32, device=self.device
        )

        # active_masks use 0 to mask out agents that have died
        active_masks = torch.ones(
            (self.algo_args["train"]["n_rollout_threads"], self.num_agents, 1),
            dtype=torch.float32, device=self.device
        )
        active_masks[dones == True] = torch.zeros(
            ((dones == True).sum(), 1), dtype=torch.float32, device=self.device
        )
        active_masks[dones_env == True] = torch.ones(
            ((dones_env == True).sum(), self.num_agents, 1), dtype=torch.float32, device=self.device
        )

        # bad_masks use 0 to denote truncation and 1 to denote termination
        if self.state_type == "EP":
            bad_masks = torch.tensor(
                [
                    [0.0]
                    if "bad_transition" in info[0].keys()
                    and info[0]["bad_transition"] == True
                    else [1.0]
                    for info in infos
                ]
            )
        elif self.state_type == "FP":
            bad_masks = torch.tensor(
                [
                    [
                        [0.0]
                        if "bad_transition" in info[agent_id].keys()
                        and info[agent_id]["bad_transition"] == True
                        else [1.0]
                        for agent_id in range(self.num_agents)
                    ]
                    for info in infos
                ]
            )

        for agent_id in range(self.num_agents):
            self.actor_buffer[agent_id].insert(
                obs[:, agent_id],
                rnn_states[:, agent_id],
                actions[:, agent_id],
                action_log_probs[:, agent_id],
                masks[:, agent_id],
                active_masks[:, agent_id],
                available_actions[:, agent_id]
                if available_actions[0] is not None
                else None,
            )

        if self.state_type == "EP":
            self.critic_buffer.insert(
                share_obs[:, 0],
                rnn_states_critic,
                values,
                rewards[:, 0],
                masks[:, 0],
                bad_masks,
            )
        elif self.state_type == "FP":
            self.critic_buffer.insert(
                share_obs, rnn_states_critic, values, rewards, masks, bad_masks
            )

    def compute(self):
        """Compute returns and advantages.
        Compute critic evaluation of the last state,
        and then let buffer compute returns, which will be used during training.
        """
        with torch.inference_mode():
            if self.state_type == "EP":
                next_value, _ = self.critic.get_values(
                    self.critic_buffer.share_obs[-1],
                    self.critic_buffer.rnn_states_critic[-1],
                    self.critic_buffer.masks[-1],
                )
                next_value = next_value
            elif self.state_type == "FP":
                next_value, _ = self.critic.get_values(
                    torch.concatenate(self.critic_buffer.share_obs[-1]),
                    torch.concatenate(self.critic_buffer.rnn_states_critic[-1]),
                    torch.concatenate(self.critic_buffer.masks[-1]),
                )
                next_value = torch.tensor(
                    torch.split(next_value), self.algo_args["train"]["n_rollout_threads"]
                )
            self.critic_buffer.compute_returns(next_value, self.value_normalizer)

    def train(self):
        """Train the model."""
        raise NotImplementedError

    def after_update(self):
        """Do the necessary data operations after an update.
        After an update, copy the data at the last step to the first position of the buffer.
        This will be used for then generating new actions.
        """
        for agent_id in range(self.num_agents):
            self.actor_buffer[agent_id].after_update()
        self.critic_buffer.after_update()

    @torch.no_grad()
    def eval(self):
        """Evaluate the model."""
        self.logger.eval_init()  # logger callback at the beginning of evaluation
        eval_episode = 0

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        eval_rnn_states = np.zeros(
            (
                self.algo_args["eval"]["n_eval_rollout_threads"],
                self.num_agents,
                self.recurrent_n,
                self.rnn_hidden_size,
            ),
            dtype=np.float32,
        )
        eval_masks = np.ones(
            (self.algo_args["eval"]["n_eval_rollout_threads"], self.num_agents, 1),
            dtype=np.float32,
        )

        while True:
            eval_actions_collector = []
            for agent_id in range(self.num_agents):
                eval_actions, temp_rnn_state = self.actor[agent_id].act(
                    eval_obs[:, agent_id],
                    eval_rnn_states[:, agent_id],
                    eval_masks[:, agent_id],
                    eval_available_actions[:, agent_id]
                    if eval_available_actions[0] is not None
                    else None,
                    deterministic=True,
                )
                eval_rnn_states[:, agent_id] = temp_rnn_state.cpu().numpy()
                eval_actions_collector.append(eval_actions.cpu().numpy())

            eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)

            (
                eval_obs,
                eval_share_obs,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            ) = self.eval_envs.step(eval_actions)
            eval_data = (
                eval_obs,
                eval_share_obs,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            )
            self.logger.eval_per_step(
                eval_data
            )  # logger callback at each step of evaluation

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[
                eval_dones_env == True
            ] = np.zeros(  # if env is done, then reset rnn_state to all zero
                (
                    (eval_dones_env == True).sum(),
                    self.num_agents,
                    self.recurrent_n,
                    self.rnn_hidden_size,
                ),
                dtype=np.float32,
            )

            eval_masks = np.ones(
                (self.algo_args["eval"]["n_eval_rollout_threads"], self.num_agents, 1),
                dtype=np.float32,
            )
            eval_masks[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32
            )

            for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    self.logger.eval_thread_done(
                        eval_i
                    )  # logger callback when an episode is done

            if eval_episode >= self.algo_args["eval"]["eval_episodes"]:
                self.logger.eval_log(
                    eval_episode
                )  # logger callback at the end of evaluation
                break

    
    def render(self):
        """Render the model."""
        with torch.inference_mode():
            print("start rendering")
            if self.manual_expand_dims:
                log_data = []
                # this env needs manual expansion of the num_of_parallel_envs dimension
                for epi in range(self.algo_args["render"]["render_episodes"]):
                    eval_obs, _, eval_available_actions = self.env.reset()
                    eval_obs = np.expand_dims(np.array(eval_obs), axis=0)
                    start_pos_x = self.env.env.sim.data.qpos[0]
                    start_pos_y = self.env.env.sim.data.qpos[1]
                    
                    step_count = 0
                    prev_pos_x = start_pos_x
                    prev_pos_y = start_pos_y
                    eval_available_actions = (
                        np.expand_dims(np.array(eval_available_actions), axis=0)
                        if eval_available_actions is not None
                        else None
                    )
                    eval_rnn_states = np.zeros(
                        (
                            self.env_num,
                            self.num_agents,
                            self.recurrent_n,
                            self.rnn_hidden_size,
                        ),
                        dtype=np.float32,
                    )
                    eval_masks = np.ones(
                        (self.env_num, self.num_agents, 1), dtype=np.float32
                    )
                    rewards = 0
                    while step_count < 200:
                        eval_actions_collector = []
                        for agent_id in range(self.num_agents):
                            eval_actions, temp_rnn_state = self.actor[agent_id].act(
                                eval_obs[:, agent_id],
                                eval_rnn_states[:, agent_id],
                                eval_masks[:, agent_id],
                                eval_available_actions[:, agent_id]
                                if eval_available_actions is not None
                                else None,
                                deterministic=True,
                            )
                            eval_rnn_states[:, agent_id] = temp_rnn_state.cpu().numpy()
                            eval_actions_collector.append(eval_actions.cpu().numpy())
                        eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)
                        (
                            eval_obs,
                            _,
                            eval_rewards,
                            eval_dones,
                            _,
                            eval_available_actions,
                        ) = self.env.step(eval_actions[0])
                        curr_pos_x = self.env.env.sim.data.qpos[0]
                        curr_pos_y = self.env.env.sim.data.qpos[1]
                        distance = np.sqrt((curr_pos_x - prev_pos_x) ** 2 + (curr_pos_y - prev_pos_y) ** 2)
                        log_data.append([epi,step_count, step_count * self.env.env.dt, curr_pos_x, curr_pos_y, distance])
                        prev_pos_x = curr_pos_x
                        prev_pos_y = curr_pos_y
                        step_count += 1
                        rewards += eval_rewards[0][0]
                        eval_obs = np.expand_dims(np.array(eval_obs), axis=0)
                        eval_available_actions = (
                            np.expand_dims(np.array(eval_available_actions), axis=0)
                            if eval_available_actions is not None
                            else None
                        )
                        if self.manual_render:
                            self.env.render()
                        if self.manual_delay:
                            time.sleep(0.1)
                        if eval_dones[0]:
                            print(f"total reward of this episode: {rewards}")
                            break
                    import csv
                with open(self.algo_args["train"]["model_dir"]+"/render_distance_log_seed.csv", "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["ep","step", "time", "x_pos", "y_pos", "distance"])
                    writer.writerows(log_data)
                print("Saved stepwise position log to render_distance_log.csv")
            else:
                # this env does not need manual expansion of the num_of_parallel_envs dimension
                # such as dexhands, which instantiates a parallel env of 64 pair of hands
                for _ in range(self.algo_args["render"]["render_episodes"]):
                    eval_obs, _, eval_available_actions = self.env.reset()
                    eval_rnn_states = torch.zeros(
                        (
                            self.env_num,
                            self.num_agents,
                            self.recurrent_n,
                            self.rnn_hidden_size,
                        ),
                        dtype=torch.float32,
                    )
                    eval_masks = torch.ones(
                        (self.env_num, self.num_agents, 1), dtype=torch.float32, device=self.device
                    )
                    rewards = 0
                    while True:
                        eval_actions_collector = []
                        for agent_id in range(self.num_agents):
                            eval_actions, temp_rnn_state = self.actor[agent_id].act(
                                eval_obs[:, agent_id],
                                eval_rnn_states[:, agent_id],
                                eval_masks[:, agent_id],
                                eval_available_actions[:, agent_id]
                                if eval_available_actions[0] is not None
                                else None,
                                deterministic=True,
                            )
                            eval_rnn_states[:, agent_id] = temp_rnn_state
                            eval_actions_collector.append(eval_actions)
                        eval_actions = torch.tensor(eval_actions_collector).permute(1, 0, 2).contiguous()
                        (
                            eval_obs,
                            _,
                            eval_rewards,
                            eval_dones,
                            _,
                            eval_available_actions,
                        ) = self.env.step(eval_actions)
                        rewards += eval_rewards[0][0][0]
                        if self.manual_render:
                            self.env.render()
                        if self.manual_delay:
                            time.sleep(0.1)
                        if eval_dones[0][0]:
                            print(f"total reward of this episode: {rewards}")
                            break
            if "smac" in self.args["env"]:  # replay for smac, no rendering
                if "v2" in self.args["env"]:
                    self.env.env.save_replay()
                else:
                    self.env.save_replay()

    def prep_rollout(self):
        """Prepare for rollout."""
        for agent_id in range(self.num_agents):
            self.actor[agent_id].prep_rollout()
        for _, critic in self.critics.items():
            critic.prep_rollout()

    def prep_training(self):
        """Prepare for training."""
        for agent_id in range(self.num_agents):
            self.actor[agent_id].prep_training()
        for _, critic in self.critics.items():
            critic.prep_training()

    def save(self, directory):
        """Save model parameters."""
        if not os.path.exists(directory):
            os.mkdir(directory)

        for agent_id in range(self.num_agents):
            policy_actor = self.actor[agent_id].actor
            torch.save(
                policy_actor.state_dict(),
                str(directory) + "/actor_agent" + str(agent_id) + ".pt",
            )
        
        for team, critic in self.critics.items():
            policy_critic = critic.critic
            torch.save(
                policy_critic.state_dict(), str(directory) + f"/{team}_critic_agent" + ".pt"
            )
            if self.value_normalizer is not None:
                torch.save(
                    self.value_normalizer.state_dict(),
                    str(directory) + f"/{team}_value_normalizer" + ".pt",
                )

    def restore(self):
        """Restore model parameters."""
        for agent_id in range(self.num_agents):
            policy_actor_state_dict = torch.load(
                str(self.algo_args["train"]["model_dir"])
                + "/actor_agent"
                + str(agent_id)
                + ".pt"
            )
            self.actor[agent_id].actor.load_state_dict(policy_actor_state_dict)
        if not self.algo_args["render"]["use_render"]:
            for team, critic in self.critics.items():
                if os.path.exists(str(self.algo_args["train"]["model_dir"]) + f"/{team}_critic_agent" + ".pt"):
                    # restore critic
                    policy_critic_state_dict = torch.load(
                        str(self.algo_args["train"]["model_dir"])
                        + f"/{team}_critic_agent"
                        + ".pt"
                    )
                    critic.critic.load_state_dict(policy_critic_state_dict)
                    if self.value_normalizer is not None:
                        value_normalizer_state_dict = torch.load(
                            str(self.algo_args["train"]["model_dir"])
                            + f"/{team}_value_normalizer"
                            + ".pt"
                        )
                        self.value_normalizer.load_state_dict(value_normalizer_state_dict)

    def close(self):
        """Close environment, writter, and logger."""
        if self.algo_args["render"]["use_render"]:
            self.env.close()
        else:
            self.env.close()
            if self.algo_args["eval"]["use_eval"] and self.eval_envs is not self.env:
                self.eval_envs.close()
            self.writter.export_scalars_to_json(str(self.log_dir + "/summary.json"))
            self.writter.close()
            self.logger.close()
