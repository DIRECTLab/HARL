"""On-policy ROSA critic buffer — EP variant.

Extends OnPolicyCriticBufferEP with the extra tensors required by the
reward-conditioned Q-critic:

  reward_vecs    — the reward-function parameter vector sampled from the
                   environment at each timestep, shape
                   (episode_length + 1, n_rollout_threads, reward_dim).

  joint_actions  — all agents' actions concatenated into a single vector,
                   shape (episode_length, n_rollout_threads, joint_act_dim).

  td_targets     — 1-step Bellman targets for Q-critic training,
                   shape (episode_length, n_rollout_threads, 1).
                   Computed by the runner in compute() as:
                       R_t + γ · V(s_{t+1}, r_t) · (1 − done_t)
                   where V is the reward-conditioned V-critic already stored
                   in the companion critic_buffer.value_preds.
"""
import torch
from harl.common.buffers.on_policy_critic_buffer_ep import OnPolicyCriticBufferEP


class OnPolicyRosaCriticBufferEP(OnPolicyCriticBufferEP):
    """On-policy critic buffer for the ROSA Q-critic (EP state type)."""

    def __init__(self, args, share_obs_space, reward_dim, joint_act_dim, device="cuda:0"):
        """
        Args:
            args: dict — same merged args dict passed to the base buffer.
            share_obs_space: centralized observation space (plain, not augmented;
                             the Q-critic concatenates obs and reward_vec itself).
            reward_dim: dimensionality of the reward-function parameter vector
                        returned by env.state().
            joint_act_dim: total action dimension of all agents concatenated
                           (n_agents × max_action_dim after any padding).
            device: torch device string or object.
        """
        super().__init__(args, share_obs_space, device)

        self.reward_dim = reward_dim
        self.joint_act_dim = joint_act_dim

        # Reward-function parameter vectors (one per step per env thread).
        # Index 0 is filled during warmup; indices 1…episode_length during insert.
        self.reward_vecs = torch.zeros(
            (self.episode_length + 1, self.n_rollout_threads, reward_dim),
            dtype=torch.float32,
            device=self.device,
        )

        # Joint actions taken at each step (all agents' actions flattened).
        # episode_length entries only — no t+1 sentinel needed.
        self.joint_actions = torch.zeros(
            (self.episode_length, self.n_rollout_threads, joint_act_dim),
            dtype=torch.float32,
            device=self.device,
        )

        # 1-step TD targets: R_t + γ · V(s_{t+1}, r_t) · (1 − done_t).
        # Populated by the runner after each rollout via set_td_targets().
        self.td_targets = torch.zeros(
            (self.episode_length, self.n_rollout_threads, 1),
            dtype=torch.float32,
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------

    def insert(
        self,
        share_obs,
        rnn_states_critic,
        value_preds,
        rewards,
        masks,
        bad_masks,
        reward_vec,
        joint_actions,
    ):
        """Insert one timestep of data.

        Args:
            share_obs, rnn_states_critic, value_preds, rewards, masks,
            bad_masks: forwarded unchanged to the base class.
            reward_vec:    (n_rollout_threads, reward_dim) Tensor — the reward-
                           function parameter vector active when this action was
                           chosen (fetched from env.state() before env.step()).
            joint_actions: (n_rollout_threads, joint_act_dim) Tensor — all
                           agents' actions concatenated.
        """
        super().insert(share_obs, rnn_states_critic, value_preds, rewards, masks, bad_masks)
        # super().insert advances self.step, so the step we just filled is
        # (self.step - 1) % episode_length.
        filled_step = (self.step - 1) % self.episode_length
        self.reward_vecs[filled_step + 1] = reward_vec.clone()
        self.joint_actions[filled_step] = joint_actions.clone()

    # ------------------------------------------------------------------
    # TD-target setter (called from runner.compute)
    # ------------------------------------------------------------------

    def set_td_targets(self, td_targets):
        """Store pre-computed 1-step TD targets.

        Args:
            td_targets: Tensor of shape (episode_length, n_rollout_threads, 1)
                        already on the correct device and in the raw reward scale.
        """
        self.td_targets[:] = td_targets

    # ------------------------------------------------------------------
    # After update
    # ------------------------------------------------------------------

    def after_update(self):
        """Carry the last timestep forward to position 0."""
        super().after_update()
        self.reward_vecs[0] = self.reward_vecs[-1].clone()
        # joint_actions and td_targets are episode-length only; nothing to carry.

    # ------------------------------------------------------------------
    # Mini-batch generator for the ROSA Q-critic
    # ------------------------------------------------------------------

    def feed_forward_generator_rosa(self, critic_num_mini_batch, mini_batch_size=None):
        """Yield mini-batches of (share_obs, joint_actions, reward_vecs, td_targets).

        Targets are the 1-step TD values R + γ·V(s')·(1−done) computed by
        the runner, *not* the GAE returns.  They are already de-normalised and
        on the raw reward scale so the Q-critic update applies plain MSE.

        Args:
            critic_num_mini_batch: number of mini-batches per epoch.
            mini_batch_size: if provided, overrides critic_num_mini_batch.
        Yields:
            4-tuple:
              share_obs_batch     — (mini_batch_size, obs_dim)
              joint_actions_batch — (mini_batch_size, joint_act_dim)
              reward_vecs_batch   — (mini_batch_size, reward_dim)
              td_targets_batch    — (mini_batch_size, 1)
        """
        episode_length, n_rollout_threads = self.rewards.shape[0:2]
        batch_size = n_rollout_threads * episode_length

        if mini_batch_size is None:
            assert batch_size >= critic_num_mini_batch, (
                f"Batch size ({batch_size}) must be >= critic_num_mini_batch "
                f"({critic_num_mini_batch})."
            )
            mini_batch_size = batch_size // critic_num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [
            rand[i * mini_batch_size : (i + 1) * mini_batch_size]
            for i in range(critic_num_mini_batch)
        ]

        # Flatten (episode_length, n_rollout_threads, *dim) → (batch, *dim).
        # share_obs[:-1]: episode_length steps (drop the sentinel at -1).
        share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[2:])
        joint_actions = self.joint_actions.reshape(-1, self.joint_act_dim)
        # reward_vecs[:-1]: r_t for each step t (drop terminal sentinel).
        reward_vecs = self.reward_vecs[:-1].reshape(-1, self.reward_dim)
        td_targets = self.td_targets.reshape(-1, 1)

        for indices in sampler:
            yield (
                share_obs[indices],
                joint_actions[indices],
                reward_vecs[indices],
                td_targets[indices],
            )
