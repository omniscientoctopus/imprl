# Module to make structural environments

import os
import yaml

from imprl.structural_envs.k_out_of_n import KOutOfN


def make(setting=None):
    """
    Make a structural environment.
    """
    # get the environment module
    env_class = KOutOfN

    # get the environment config
    pwd = os.path.dirname(__file__)
    rel_path = f"env_configs/{setting}.yaml"
    abs_file_path = os.path.join(pwd, rel_path)

    with open(abs_file_path) as file:
        env_config = yaml.load(file, Loader=yaml.FullLoader)

    # get baselines
    rel_path = "baselines.yaml"
    abs_file_path = os.path.join(pwd, rel_path)

    with open(abs_file_path) as file:
        all_baselines = yaml.load(file, Loader=yaml.FullLoader)

    baselines = all_baselines["k_out_of_n"][setting]

    # create the environment
    env = env_class(env_config, baselines)

    return env
