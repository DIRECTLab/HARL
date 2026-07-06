"""Runner for on-policy MA algorithms."""
import numpy as np
import torch
from harl.utils.models_tools import torch_nanstd
from harl.runners.on_policy_base_runner import OnPolicyBaseRunner


class OnPolicyMARunner(OnPolicyBaseRunner):
    """Runner for on-policy MA algorithms."""

    def train(self):
        """Training procedure for MAPPO."""
        actor_train_infos = []

        # compute advantages
        if self.value_normalizer is not None:
            advantages = self.critic_buffer.returns[
                :-1
            ] - self.value_normalizer.denormalize(self.critic_buffer.value_preds[:-1])
        else:
            advantages = (
                self.critic_buffer.returns[:-1] - self.critic_buffer.value_preds[:-1]
            )

        # normalize advantages for FP
        if self.state_type == "FP":
            active_masks_collector = [
                self.actor_buffer[i].active_masks for i in range(self.num_agents)
            ]
            active_masks_array = torch.stack(active_masks_collector, axis=2)
            advantages_copy = advantages.clone()
            advantages_copy[active_masks_array[:-1] == 0.0] = torch.nan
            mean_advantages = torch.nanmean(advantages_copy)
            std_advantages = torch_nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        # update actors
        if self.share_param:
            actor_train_info = self.actor[0].share_param_train(
                self.actor_buffer, advantages.clone(), self.num_agents, self.state_type
            )
            for _ in torch.randperm(self.num_agents):
                actor_train_infos.append(actor_train_info)
        else:
            for agent_id in range(self.num_agents):
                if self.state_type == "EP":
                    actor_train_info = self.actor[agent_id].train(
                        self.actor_buffer[agent_id], advantages.clone(), "EP"
                    )
                elif self.state_type == "FP":
                    actor_train_info = self.actor[agent_id].train(
                        self.actor_buffer[agent_id],
                        advantages[:, :, agent_id].clone(),
                        "FP",
                    )
                actor_train_infos.append(actor_train_info)

        # update V-critic
        critic_train_info = self.critic.train(self.critic_buffer, self.value_normalizer)

        # update ROSA reward-conditioned Q-critic (if enabled)
        if self.use_rosa:
            rosa_train_info = self.rosa_q_critic.train(self.rosa_critic_buffer)
            critic_train_info.update(rosa_train_info)

        return actor_train_infos, critic_train_info
