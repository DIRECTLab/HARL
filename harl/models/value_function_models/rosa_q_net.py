import torch
import torch.nn as nn
from harl.models.base.plain_cnn import PlainCNN
from harl.models.base.plain_mlp import PlainMLP
from harl.utils.envs_tools import get_shape_from_obs_space


def get_combined_dim(cent_obs_feature_dim, act_spaces, reward_dim):
    """Get the combined input dimension: obs features + actions + reward vector."""
    combined_dim = cent_obs_feature_dim + reward_dim
    for space in act_spaces:
        if space.__class__.__name__ == "Box":
            combined_dim += space.shape[0]
        elif space.__class__.__name__ == "Discrete":
            combined_dim += space.n
        else:
            for action_dim in space.nvec:
                combined_dim += action_dim
    return combined_dim


class RosaQNet(nn.Module):
    """Reward-conditioned Q-network for ROSA.

    Evaluates Q(s, a, r) where r is a sampled reward-function parameter vector.
    Used to rank a set of candidate actions under a given reward function so the
    agent can select the action that scores highest without committing to a single
    fixed reward signal during training.

    Input:  centralized observation  (obs_dim)
          + concatenated agent actions (sum of act_dims)
          + reward parameter vector   (reward_dim)
    Output: scalar Q-value
    """

    def __init__(self, args, cent_obs_space, act_spaces, reward_dim, device=torch.device("cpu")):
        """
        Args:
            args: dict with keys activation_func, hidden_sizes (and optionally
                  rosa_reward_dim, which overrides the reward_dim argument).
            cent_obs_space: centralized observation space.
            act_spaces: list of per-agent action spaces.
            reward_dim: dimensionality of the reward parameter vector.
            device: torch device.
        """
        super(RosaQNet, self).__init__()

        activation_func = args["activation_func"]
        hidden_sizes = args["hidden_sizes"]
        # Allow args to override reward_dim for convenience
        reward_dim = args.get("rosa_reward_dim", reward_dim)

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        if len(cent_obs_shape) == 3:
            self.feature_extractor = PlainCNN(cent_obs_shape, hidden_sizes[0], activation_func)
            cent_obs_feature_dim = hidden_sizes[0]
        else:
            self.feature_extractor = None
            cent_obs_feature_dim = cent_obs_shape[0]

        input_dim = get_combined_dim(cent_obs_feature_dim, act_spaces, reward_dim)
        sizes = [input_dim] + list(hidden_sizes) + [1]
        self.mlp = PlainMLP(sizes, activation_func)
        self.to(device)

    def forward(self, cent_obs, actions, reward_vec):
        """
        Args:
            cent_obs:   (batch, obs_dim)   centralized observation.
            actions:    (batch, act_dim)   concatenated agent actions.
            reward_vec: (batch, reward_dim) reward function parameter vector.
        Returns:
            q_values: (batch, 1) scalar Q-value for each (s, a, r) triple.
        """
        if self.feature_extractor is not None:
            obs_feat = self.feature_extractor(cent_obs)
        else:
            obs_feat = cent_obs
        concat_x = torch.cat([obs_feat, actions, reward_vec], dim=-1)
        return self.mlp(concat_x)
