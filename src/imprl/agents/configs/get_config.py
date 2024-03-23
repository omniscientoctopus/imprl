import yaml
from importlib_resources import files

def load_config(yaml_file=None, algorithm=None):

    """
    Load configuration file or a default configuration for a given algorithm.

    Parameters
    ----------
    yaml_file : str
        Path to a yaml file containing the configuration parameters.

    algorithm : str
        Algorithm name.

    Returns
    -------
    config : dict
        Dictionary containing the configuration parameters.

    """

    if yaml_file is not None:

        with open(yaml_file, 'r') as f:
            config = yaml.safe_load(f)

    elif algorithm is not None:

        # load default config
        config_file = files('imprl.agents.configs').joinpath(f'{algorithm}.yaml')
        with open(config_file, 'r') as config_file:
            config = yaml.safe_load(config_file)

        print(f"Loaded default configuration for {algorithm}.")

    else:
        raise ValueError("Either yaml_file or algorithm must be provided.")

    return config