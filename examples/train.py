import time
import torch

import imprl.agents
import imprl.structural_envs as structural_envs
from imprl.runners.serial import training_rollout
from imprl.agents.configs.get_config import load_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# Environment
env = structural_envs.make(setting="hard-5-of-5")

# Agent
algorithm = "DDQN"
config = load_config(algorithm=algorithm)  # load default config
agent_class = imprl.agents.get_agent_class(algorithm)
LearningAgent = agent_class(env, config, device)  # initialize agent

time0 = time.time()

# training loop
for ep in range(100):

    episode_return = training_rollout(env, LearningAgent)

    LearningAgent.report()

print(f"Total time: {time.time()-time0:.2f}")