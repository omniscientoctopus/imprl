import yaml
import time
import math
from importlib_resources import files

import torch
import wandb
import numpy as np

import imprl.agents
import imprl.structural_envs as structural_envs
from imprl.runners.serial import training_rollout, evaluate_agent
from imprl.agents.configs.get_config import load_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

run = wandb.init(
    project=None,  # change this to your project name
    entity=None,  # change this to your username
)

# logging and checkpointing
logging_frequency = 100
checkpt_frequency = 5_000
inferencing_frequency = 5_000
num_inference_episodes = 1_000
best_cost = math.inf
best_checkpt = 0

is_time_to_checkpoint = (
    lambda ep: ep % checkpt_frequency == 0 or ep == wandb.config.NUM_EPISODES - 1
)
is_time_to_log = (
    lambda ep: ep % logging_frequency == 0 or ep == wandb.config.NUM_EPISODES - 1
)
is_time_to_infer = (
    lambda ep: ep % inferencing_frequency == 0 or ep == wandb.config.NUM_EPISODES - 1
)

checkpt_dir = wandb.run.dir
print("Checkpoint directory: ", checkpt_dir)

training_log = {}  # log for training metrics

# Environment
setting = "hard-5-of-5"
env = structural_envs.make(setting=setting)

# Agent
algorithm = "DDQN"
config = load_config(algorithm=algorithm)  # load default config
agent_class = imprl.agents.get_agent_class(algorithm)
LearningAgent = agent_class(env, config, device)  # initialize agent

wandb.config.update(config)  # log the config to wandb

# baselines
_path = files("imprl.structural_envs").joinpath("baselines.yaml")
with open(_path, "r") as f:
    _baseline = yaml.safe_load(f)

_baseline = _baseline["k_out_of_n"][setting]

time0 = time.time()

# training loop
for ep in range(config["NUM_EPISODES"]):

    episode_return = training_rollout(env, LearningAgent)

    LearningAgent.report()

    # CHECKPOINT
    if is_time_to_checkpoint(ep):
        LearningAgent.save_weights(checkpt_dir, ep)

    # INFERENCE
    if is_time_to_infer(ep):
        TrainedAgent = agent_class(env, config, device)  # create a new agent
        TrainedAgent.load_weights(
            checkpt_dir, ep
        )  # load the weights of the trained agent

        eval_costs = []
        # evaluate the agent
        for _ in range(num_inference_episodes):
            evaluation_cost = evaluate_agent(env, TrainedAgent)
            eval_costs.append(evaluation_cost)

        _mean = np.mean(eval_costs)
        _stderr = np.std(eval_costs) / np.sqrt(len(eval_costs))

        if _mean < best_cost:
            best_cost = _mean
            best_checkpt = ep

        training_log.update(
            {"inference_ep": ep, "inference_mean": _mean, "inference_stderr": _stderr}
        )

    # LOGGING
    if is_time_to_log(ep):
        training_log.update(LearningAgent.logger)  # agent logger
        training_log.update(_baseline)  # baseline logger
        wandb.log(training_log, step=ep)  # log to wandb

print(f"Total time: {time.time()-time0:.2f}")
