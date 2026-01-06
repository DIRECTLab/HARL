import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional
from harl.utils.models_tools import init, get_init_method


class SpikingMLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_sizes, initialization_method, T=10):
        """
        SNN version of MLPLayer using direct encoding and LIF neurons.

        Args:
            input_dim: (int) input dimension.
            hidden_sizes: (list[int]) sizes of each hidden layer.
            initialization_method: (str) init method name (from HARL utils).
            T: (int) number of internal SNN timesteps per forward.
        """
        super().__init__()
        self.T = T
        self.hidden_sizes = hidden_sizes

        init_method = get_init_method(initialization_method)

        def init_(m):
            # mimic MLPLayer: init weights with init_method, bias to 0
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        # -----------------------------
        # Build linear + LIF stacks
        # -----------------------------
        self.linears = nn.ModuleList()
        self.lifs = nn.ModuleList()

        prev_dim = input_dim
        for h_dim in hidden_sizes:
            lin = init_(nn.Linear(prev_dim, h_dim))
            lif = neuron.LIFNode(
                tau=2.0,
                surrogate_function=neuron.surrogate.Sigmoid(),
                detach_reset=True,
            )
            self.linears.append(lin)
            self.lifs.append(lif)
            prev_dim = h_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, input_dim]
        returns: [B, last_hidden_dim] firing-rate representation
        """
        B = x.shape[0]

        # Reset membrane/spike states for all LIFNodes in this module
        functional.reset_net(self)

        # Repeat input across internal timesteps: [T, B, input_dim]
        seq = x.unsqueeze(0).repeat(self.T, 1, 1)

        # Propagate through each (Linear + LIF) block over time
        for lin, lif in zip(self.linears, self.lifs):
            spk_seq = []
            for t in range(self.T):
                # Linear on current representation (initially raw input, later spikes)
                cur = lin(seq[t])             # [B, h_dim]
                spk = lif(cur)                # [B, h_dim], 0/1 spikes
                spk_seq.append(spk)
            # Stack back into sequence for next layer: [T, B, h_dim]
            seq = torch.stack(spk_seq, dim=0)

        # seq now holds spikes from final layer: [T, B, last_hidden_dim]
        # Convert to firing rate by averaging over time
        firing_rate = seq.mean(dim=0)          # [B, last_hidden_dim]

        return firing_rate


class SpikingMLPBase(nn.Module):
    """SNN version of MLPBase using SpikingMLPLayer."""

    def __init__(self, args, obs_shape):
        super(SpikingMLPBase, self).__init__()

        self.use_feature_normalization = args["use_feature_normalization"]
        self.initialization_method = args["initialization_method"]
        self.hidden_sizes = args["hidden_sizes"]
        # new: internal SNN timesteps
        self.T = args.get("snn_T", 10)

        obs_dim = obs_shape[0]

        if self.use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)

        self.mlp = SpikingMLPLayer(
            obs_dim,
            self.hidden_sizes,
            self.initialization_method,
            T=self.T,
        )

    def forward(self, x, num_envs=None):
        # x: [B, obs_dim]
        if self.use_feature_normalization:
            x = self.feature_norm(x)

        x = self.mlp(x)   # [B, last_hidden_dim] firing rate
        return x
