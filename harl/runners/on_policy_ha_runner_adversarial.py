"""Runner for on-policy HARL algorithms."""
import numpy as np
import torch
import random
from harl.utils.trans_tools import _t2n
from harl.utils.models_tools import torch_nanstd
from harl.runners.on_policy_base_runner_adversarial import OnPolicyBaseRunnerAdversarial


class OnPolicyHARunnerAdversarial(OnPolicyBaseRunnerAdversarial):
    """Runner for on-policy HA algorithms."""

    def train(self):
        """Train the model."""
        actor_train_infos = {}

        # factor is used for considering updates made by previous agents
        factor = torch.ones(
            (
                self.algo_args["train"]["episode_length"],
                self.algo_args["train"]["n_rollout_threads"],
                1,
            ),
            dtype=torch.float32, device=self.device,
        )

        advantages = {}

        for team in self.training_teams:
            # compute advantages
            if self.value_normalizers is not None:
                advantages[team] = self.critic_buffers[team].returns[
                    :-1
                ] - self.value_normalizers[team].denormalize(self.critic_buffers[team].value_preds[:-1])
            else:
                advantages[team] = (
                    self.critic_buffers[team].returns[:-1] - self.critic_buffers[team].value_preds[:-1]
                )

        # normalize advantages for FP
        if self.state_type == "FP":
            #TODO: get to work with adversarial
            pass
            # active_masks_collector = [
            #     self.actor_buffer[i].active_masks for i in range(self.num_agents)
            # ]
            # active_masks_array = torch.stack(active_masks_collector, axis=2)
            # advantages_copy = advantages.clone()
            # advantages_copy[active_masks_array[:-1] == 0.0] = torch.nan
            # mean_advantages = torch.nanmean(advantages_copy)
            # std_advantages = torch_nanstd(advantages_copy)
            # advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        for team in self.training_teams:
            agents = self.teams[team]
            if not self.fixed_order:
                random.shuffle(agents)

            for agent_id in agents:
                self.actor_buffers[team][agent_id].update_factor(
                    factor
                )  # current actor save factor

                # the following reshaping combines the first two dimensions (i.e. episode_length and n_rollout_threads) to form a batch
                available_actions = (
                    None
                    if self.actor_buffers[team][agent_id].available_actions is None
                    else self.actor_buffers[team][agent_id]
                    .available_actions[:-1]
                    .reshape(-1, *self.actor_buffers[team][agent_id].available_actions.shape[2:])
                )

                # compute action log probs for the actor before update.
                old_actions_logprob, _, _ = self.actors[team][agent_id].evaluate_actions(
                    self.actor_buffers[team][agent_id]
                    .obs[:-1]
                    .reshape(-1, *self.actor_buffers[team][agent_id].obs.shape[2:]),
                    self.actor_buffers[team][agent_id]
                    .rnn_states[0:1]
                    .reshape(-1, *self.actor_buffers[team][agent_id].rnn_states.shape[2:]),
                    self.actor_buffers[team][agent_id].actions.reshape(
                        -1, *self.actor_buffers[team][agent_id].actions.shape[2:]
                    ),
                    self.actor_buffers[team][agent_id]
                    .masks[:-1]
                    .reshape(-1, *self.actor_buffers[team][agent_id].masks.shape[2:]),
                    available_actions,
                    self.actor_buffers[team][agent_id]
                    .active_masks[:-1]
                    .reshape(-1, *self.actor_buffers[team][agent_id].active_masks.shape[2:]),
                )

                # update actor
                if self.state_type == "EP":
                    actor_train_info = self.actors[team][agent_id].train(
                        self.actor_buffers[team][agent_id], advantages[team].clone(), "EP"
                    )
                elif self.state_type == "FP":
                    actor_train_info = self.actors[team][agent_id].train(
                        self.actor_buffers[team][agent_id], advantages[team][:, :, agent_id].clone(), "FP"
                    )

                # compute action log probs for updated agent
                new_actions_logprob, _, _ = self.actors[team][agent_id].evaluate_actions(
                    self.actor_buffers[team][agent_id]
                    .obs[:-1]
                    .reshape(-1, *self.actor_buffers[team][agent_id].obs.shape[2:]),
                    self.actor_buffers[team][agent_id]
                    .rnn_states[0:1]
                    .reshape(-1, *self.actor_buffers[team][agent_id].rnn_states.shape[2:]),
                    self.actor_buffers[team][agent_id].actions.reshape(
                        -1, *self.actor_buffers[team][agent_id].actions.shape[2:]
                    ),
                    self.actor_buffers[team][agent_id]
                    .masks[:-1]
                    .reshape(-1, *self.actor_buffers[team][agent_id].masks.shape[2:]),
                    available_actions,
                    self.actor_buffers[team][agent_id]
                    .active_masks[:-1]
                    .reshape(-1, *self.actor_buffers[team][agent_id].active_masks.shape[2:]),
                )

                # update factor for next agent
                factor = factor * (
                    getattr(torch, self.action_aggregation)(
                        torch.exp(new_actions_logprob - old_actions_logprob), dim=-1
                    ).reshape(
                        self.algo_args["train"]["episode_length"],
                        self.algo_args["train"]["n_rollout_threads"],
                        1,
                    )
                )

                factor.detach_()
                
                actor_train_infos[agent_id] = actor_train_info

        critic_train_infos = {}
        # update critic
        for team in self.team_names:
            critic_train_infos[team] = self.critics[team].train(self.critic_buffers[team], self.value_normalizers[team])

        return actor_train_infos, critic_train_infos
