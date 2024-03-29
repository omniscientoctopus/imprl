import os
import torch

import imprl.agents
import imprl.structural_envs as structural_envs
from imprl.agents.configs.get_config import load_config
from imprl.post_process.plotter.rollout_animation import AnimatedRollout
from imprl.post_process.inference import AgentInference

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# Environment
setting = "hard-5-of-5"
env = structural_envs.make(setting=setting)

# Agent
algorithm = "DDQN"
config = load_config(algorithm=algorithm)  # load default config
agent_class = imprl.agents.get_agent_class(algorithm)
TrainedAgent = agent_class(env, config, device)  # initialize agent

local_path = os.path.dirname(os.path.realpath(__file__))
checkpt_dir = f"{local_path}/data/run-j1fm9qvi/model_weights"
ep = 20_000  # episode number
TrainedAgent.load_weights(checkpt_dir, ep)

# Animation
ar = AnimatedRollout(env, TrainedAgent)

ar.run()
