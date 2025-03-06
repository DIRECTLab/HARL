"""Train an algorithm."""
import argparse
import json
from harl.utils.configs_tools import get_defaults_yaml_args, update_args


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="happo",
        choices=[
            "happo",
            "hatrpo",
            "haa2c",
            "haddpg",
            "hatd3",
            "hasac",
            "had3qn",
            "maddpg",
            "matd3",
            "mappo",
        ],
        help="Algorithm name. Choose from: happo, hatrpo, haa2c, haddpg, hatd3, hasac, had3qn, maddpg, matd3, mappo.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="pettingzoo_mpe",
        choices=[
            "smac",
            "mamujoco",
            "pettingzoo_mpe",
            "gym",
            "football",
            "dexhands",
            "smacv2",
            "lag",
        ],
        help="Environment name. Choose from: smac, mamujoco, pettingzoo_mpe, gym, football, dexhands, smacv2, lag.",
    )
    parser.add_argument(
        "--exp_name", type=str, default="installtest", help="Experiment name."
    )
    parser.add_argument(
        "--load_config",
        type=str,
        default="",
        help="If set, load existing experiment config file instead of reading from yaml config file.",
    )
    parser.add_argument(
        "--use_neuron",
        type=bool,
        default=False,
        help="If set uses neuron model, by converting the environment to neuron environment.",
    )
    parser.add_argument(
    "--neuron_bandwidth",
    type=int,
    default=3,
    help="Bandwidth of the neuron.",
)

    parser.add_argument(
        "--num_hidden_neurons_per_agent",
        type=int,
        default=10,
        help="Number of hidden neurons per agent.",
    )

    parser.add_argument(
        "--position_dims",
        type=int,
        default=2,
        help="Dimensionality of position representation.",
    )

    parser.add_argument(
        "--speed_factor",
        type=int,
        default=2,
        help="Factor influencing agent speed.",
    )

    parser.add_argument(
        "--exp_decay_scale",
        type=float,
        default=0.5,
        help="Scale of exponential decay.",
    )

    parser.add_argument(
        "--max_connection_change",
        type=float,
        default=0.01,
        help="Maximum allowed connection change per step.",
    )

    parser.add_argument(
        "--max_velocity",
        type=float,
        default=0.25,
        help="Maximum velocity of agents.",
    )

    parser.add_argument(
        "--brain_size",
        type=int,
        default=1,
        help="Size of the agent's brain structure.",
    )

    parser.add_argument(
        "--reset_env_factor",
        type=int,
        default=3,
        help="Factor controlling environment reset frequency.",
    )

    parser.add_argument(
        "--noise_scale",
        type=float,
        default=0.1,
        help="Scale of added noise in the environment.",
    )

    args, unparsed_args = parser.parse_known_args()

    def process(arg):
        try:
            return eval(arg)
        except:
            return arg

    keys = [k[2:] for k in unparsed_args[0::2]]  # remove -- from argument
    values = [process(v) for v in unparsed_args[1::2]]
    unparsed_dict = {k: v for k, v in zip(keys, values)}
    args = vars(args)  # convert to dict
    if args["load_config"] != "":  # load config from existing config file
        with open(args["load_config"], encoding="utf-8") as file:
            all_config = json.load(file)
        args["algo"] = all_config["main_args"]["algo"]
        args["env"] = all_config["main_args"]["env"]
        algo_args = all_config["algo_args"]
        env_args = all_config["env_args"]
    else:  # load config from corresponding yaml file
        algo_args, env_args, neuron_args = get_defaults_yaml_args(args["algo"], args["env"])

    update_args(unparsed_dict, algo_args, env_args)  # update args from command line

    if args["env"] == "dexhands":
        import isaacgym  # isaacgym has to be imported before PyTorch

    # note: isaac gym does not support multiple instances, thus cannot eval separately
    algo_args["eval"]["use_eval"] = False
    if args["env"] == "dexhands":
        algo_args["train"]["episode_length"] = env_args["hands_episode_length"]

    if args["use_neuron"] and args["load_config"] == "":
        neuron_args["neuron_bandwidth"] = args["neuron_bandwidth"]
        neuron_args["num_hidden_neurons_per_agent"] = args["num_hidden_neurons_per_agent"]
        neuron_args["position_dims"] = args["position_dims"]
        neuron_args["speed_factor"] = args["speed_factor"]
        neuron_args["exp_decay_scale"] = args["exp_decay_scale"]
        neuron_args["max_connection_change"] = args["max_connection_change"]
        neuron_args["max_velocity"] = args["max_velocity"]
        neuron_args["brain_size"] = args["brain_size"]
        neuron_args["reset_env_factor"] = args["reset_env_factor"]
        neuron_args["noise_scale"] = args["noise_scale"]
        env_args["neuron_args"] = neuron_args

    # start training
    from harl.runners import RUNNER_REGISTRY

    runner = RUNNER_REGISTRY[args["algo"]](args, algo_args, env_args)
    runner.run()
    runner.close()


if __name__ == "__main__":
    main()
