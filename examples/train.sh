# Baseline experiment
python examples/train.py --algo happo --env pettingzoo_mpe --exp_name baseline --use_neuron True --neuron_bandwidth 3 --num_hidden_neurons_per_agent 3 --position_dims 2 --speed_factor 2 --exp_decay_scale 0.1 --max_connection_change 0.1 --max_velocity 0.25 --brain_size 1 --reset_env_factor 3 --noise_scale 1

# Slow reset environment factor
python examples/train.py --algo happo --env pettingzoo_mpe --exp_name slow_reset --use_neuron True --neuron_bandwidth 3 --num_hidden_neurons_per_agent 3 --position_dims 2 --speed_factor 2 --exp_decay_scale 0.1 --max_connection_change 0.1 --max_velocity 0.25 --brain_size 1 --reset_env_factor 9 --noise_scale 1

# Fast reset environment factor
python examples/train.py --algo happo --env pettingzoo_mpe --exp_name fast_reset --use_neuron True --neuron_bandwidth 3 --num_hidden_neurons_per_agent 3 --position_dims 2 --speed_factor 2 --exp_decay_scale 0.1 --max_connection_change 0.1 --max_velocity 0.25 --brain_size 1 --reset_env_factor 1 --noise_scale 1

# Increase neuron bandwidth
python examples/train.py --algo happo --env pettingzoo_mpe --exp_name increase_bandwidth --use_neuron True --neuron_bandwidth 5 --num_hidden_neurons_per_agent 3 --position_dims 2 --speed_factor 2 --exp_decay_scale 0.1 --max_connection_change 0.1 --max_velocity 0.25 --brain_size 1 --reset_env_factor 3 --noise_scale 1

#decrease neuron bandwidth
python examples/train.py --algo happo --env pettingzoo_mpe --exp_name decrease_bandwidth --use_neuron True --neuron_bandwidth 1 --num_hidden_neurons_per_agent 3 --position_dims 2 --speed_factor 2 --exp_decay_scale 0.1 --max_connection_change 0.1 --max_velocity 0.25 --brain_size 1 --reset_env_factor 3 --noise_scale 1

# Increase speed factor
python examples/train.py --algo happo --env pettingzoo_mpe --exp_name high_speed --use_neuron True --neuron_bandwidth 3 --num_hidden_neurons_per_agent 3 --position_dims 2 --speed_factor 4 --exp_decay_scale 0.1 --max_connection_change 0.1 --max_velocity 0.25 --brain_size 1 --reset_env_factor 3 --noise_scale 1

# decrease speed factor
python examples/train.py --algo happo --env pettingzoo_mpe --exp_name low_speed --use_neuron True --neuron_bandwidth 3 --num_hidden_neurons_per_agent 3 --position_dims 2 --speed_factor 1 --exp_decay_scale 0.1 --max_connection_change 0.1 --max_velocity 0.25 --brain_size 1 --reset_env_factor 3 --noise_scale 1
