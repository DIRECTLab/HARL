"""ROSA Reward-Conditioned Q-Critic."""
import os
import torch
import torch.nn as nn
from harl.utils.models_tools import (
    get_grad_norm,
    huber_loss,
    mse_loss,
    update_linear_schedule,
)
from harl.utils.envs_tools import check
from harl.models.value_function_models.rosa_q_net import RosaQNet


class RosaQCritic:
    """Reward-conditioned Q-critic for ROSA.

    Learns Q(s, a, r) where r is the sampled reward-function parameter vector.
    During rollout, the runner samples N candidate joint actions per environment
    thread, scores each with this critic, and executes the highest-scoring action.
    During training, the critic is updated with the GAE returns already computed
    by the V-critic, treating them as regression targets for Q(s, a, r).
    """

    def __init__(self, args, cent_obs_space, act_spaces, reward_dim, device=torch.device("cpu")):
        """
        Args:
            args: dict — merged model + algo args (same dict passed to VCritic).
            cent_obs_space: centralized observation space.
            act_spaces: list of per-agent action spaces (same order as agent IDs).
            reward_dim: dimensionality of the reward parameter vector from env.state().
            device: torch device.
        """
        self.args = args
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.critic_epoch = args["critic_epoch"]
        self.critic_num_mini_batch = args["critic_num_mini_batch"]
        self.value_loss_coef = args["value_loss_coef"]
        self.max_grad_norm = args["max_grad_norm"]
        self.huber_delta = args["huber_delta"]
        self.use_max_grad_norm = args["use_max_grad_norm"]
        self.use_huber_loss = args["use_huber_loss"]
        self.critic_lr = args["critic_lr"]
        self.opti_eps = args["opti_eps"]
        self.weight_decay = args["weight_decay"]

        self.critic = RosaQNet(args, cent_obs_space, act_spaces, reward_dim, device)

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )

    # ------------------------------------------------------------------
    # Learning-rate schedule
    # ------------------------------------------------------------------

    def lr_decay(self, episode, episodes):
        """Linearly decay the learning rate."""
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def get_q_values(self, cent_obs, actions, reward_vec):
        """Score a batch of (state, joint-action, reward-vector) triples.

        Args:
            cent_obs:   (np.ndarray | Tensor) shape (batch, obs_dim).
            actions:    (np.ndarray | Tensor) shape (batch, joint_act_dim).
                        Joint action = all agents' actions concatenated along the
                        last dimension.
            reward_vec: (np.ndarray | Tensor) shape (batch, reward_dim).
        Returns:
            q_values: Tensor shape (batch, 1).
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv)
        reward_vec = check(reward_vec).to(**self.tpdv)
        return self.critic(cent_obs, actions, reward_vec)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def cal_q_loss(self, q_preds, q_targets):
        """MSE or Huber regression loss between Q predictions and targets.

        Args:
            q_preds:   (Tensor) shape (batch, 1).
            q_targets: (Tensor) shape (batch, 1), already de-normalised if
                       value normalisation is in use.
        Returns:
            scalar loss Tensor.
        """
        error = q_targets - q_preds
        if self.use_huber_loss:
            loss = huber_loss(error, self.huber_delta)
        else:
            loss = mse_loss(error)
        return loss.mean()

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, sample):
        """Single mini-batch gradient step.

        Args:
            sample: 4-tuple yielded by
                    OnPolicyRosaCriticBufferEP.feed_forward_generator_rosa —
                    (share_obs_batch, joint_actions_batch,
                     reward_vecs_batch, td_targets_batch).
                    td_targets = R + γ·V(s')·(1−done) already in raw reward
                    scale; no value-normalizer de-normalisation is required.
        Returns:
            q_loss (scalar Tensor), grad_norm (float).
        """
        share_obs_batch, joint_actions_batch, reward_vecs_batch, td_targets_batch = sample

        td_targets_batch = check(td_targets_batch).to(**self.tpdv)

        q_preds = self.get_q_values(share_obs_batch, joint_actions_batch, reward_vecs_batch)

        q_loss = self.cal_q_loss(q_preds, td_targets_batch)

        self.critic_optimizer.zero_grad()
        (q_loss * self.value_loss_coef).backward()

        if self.use_max_grad_norm:
            grad_norm = nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.max_grad_norm
            )
        else:
            grad_norm = get_grad_norm(self.critic.parameters())

        self.critic_optimizer.step()
        return q_loss, grad_norm

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, rosa_buffer):
        """Full training loop over the ROSA replay buffer.

        TD targets stored in ``rosa_buffer.td_targets`` are already in the
        raw reward scale (computed by the runner as R + γ·V(s')·(1−done)),
        so no value-normalizer is needed here.

        Args:
            rosa_buffer: OnPolicyRosaCriticBufferEP instance.
        Returns:
            train_info dict with keys ``rosa_q_loss`` and ``rosa_q_grad_norm``.
        """
        train_info = {"rosa_q_loss": 0.0, "rosa_q_grad_norm": 0.0}

        for _ in range(self.critic_epoch):
            data_generator = rosa_buffer.feed_forward_generator_rosa(
                self.critic_num_mini_batch
            )
            for sample in data_generator:
                q_loss, grad_norm = self.update(sample)
                train_info["rosa_q_loss"] += q_loss.item()
                train_info["rosa_q_grad_norm"] += float(grad_norm)

        num_updates = self.critic_epoch * self.critic_num_mini_batch
        for k in train_info:
            train_info[k] /= num_updates

        return train_info

    # ------------------------------------------------------------------
    # Mode switching
    # ------------------------------------------------------------------

    def prep_training(self):
        """Switch to training mode."""
        self.critic.train()

    def prep_rollout(self):
        """Switch to evaluation mode."""
        self.critic.eval()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory):
        """Save critic weights to *directory*."""
        torch.save(
            self.critic.state_dict(),
            os.path.join(directory, "rosa_q_critic.pt"),
        )

    def restore(self, directory):
        """Load critic weights from *directory* (silently skips if absent)."""
        path = os.path.join(directory, "rosa_q_critic.pt")
        if os.path.exists(path):
            state_dict = torch.load(path, map_location="cpu")
            self.critic.load_state_dict(state_dict)
