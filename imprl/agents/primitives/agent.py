import random

import torch

torch.set_default_dtype(torch.float64)

from imprl.agents.modules.exploration_schedulers import LinearExplorationScheduler
from imprl.agents.modules.replay_memory import AbstractReplayMemory
from imprl.agents.utils import get_multiagent_obs, get_multiagent_obs_with_idx


class Agent:

    def __init__(self, env, config, device):

        self.device = device  # to send neural network ops to device

        # Environment parameters
        self.env = env
        self.state_space = env.observation_space
        self.action_space = env.action_space
        self.n_components = env.n_components  # number of components
        self.n_damage_states = env.n_damage_states  # damage states per component
        self.n_comp_actions = env.n_comp_actions

        # Initialization
        self.episode = 0
        self.total_time = 0  # total time steps in lifetime
        self.time = 0  # time steps in current episode
        self.episode_return = 0  # return in current episode

        ## Training parameters
        self.discount_factor = config["DISCOUNT_FACTOR"]
        self.batch_size = config["BATCH_SIZE"]
        self.exploration_strategy = config["EXPLORATION_STRATEGY"]
        self.exploration_param = self.exploration_strategy["max_value"]

        # exploration scheduler
        self.exp_scheduler = LinearExplorationScheduler(
            self.exploration_strategy["min_value"],
            num_episodes=self.exploration_strategy["num_episodes"],
        )

        # replay memory stores (a subset of) experience across episodes
        self.replay_memory = AbstractReplayMemory(config["MAX_MEMORY_SIZE"])

        # logger
        self.logger = {"exploration_param": self.exploration_param}

    def reset_episode(self, training=True):

        self.episode_return = 0
        self.episode += 1
        self.time = 0

        if training:

            # update exploration param
            self.exploration_param = self.exp_scheduler.step()

            # logging
            self.logger["exploration_param"] = self.exploration_param

    def epsilon_greedy_strategy(self, observation, training):

        # select random action
        if self.exploration_param > random.random():
            return self.get_random_action()
        else:
            # select greedy action
            return self.get_greedy_action(observation, training)

    def select_action(self, observation, training):

        if training:
            if self.exploration_strategy["name"] == "epsilon_greedy":
                return self.epsilon_greedy_strategy(observation, training)
        else:
            return self.get_greedy_action(observation, training=False)

    def process_rewards(self, reward):

        self.episode_return += reward

        # updating time here so that only this method needs to be called
        # during inference
        self.time += 1
        self.total_time += 1

    def compute_td_target(self, t_next_beliefs, t_rewards, t_dones):

        # bootstrapping
        future_values = self.get_future_values(t_next_beliefs)

        # set future values of done states to 0
        not_dones = 1 - t_dones
        future_values *= not_dones
        td_target = t_rewards + self.discount_factor * future_values

        return td_target.detach()

    def report(self, stats=None):
        """Print stats to console."""

        print(f"Ep:{self.episode:05}| Cost: {-self.episode_return:.2f}", flush=True)

        if stats is not None:
            print(stats)

    def get_multiagent_obs(self, t_obs, batch_size=1):

        return get_multiagent_obs(
            t_obs, self.n_damage_states, self.n_components, batch_size=batch_size
        )

    def get_multiagent_obs_with_idx(self, t_obs, batch_size=1):

        return get_multiagent_obs_with_idx(
            t_obs, self.n_damage_states, self.n_components, batch_size=batch_size
        )

    def process_experience(self):
        NotImplementedError

    def train(self):
        NotImplementedError

    def _preprocess_inputs(self):
        NotImplementedError

    def get_random_action(self):
        NotImplementedError

    def future_values(self):
        NotImplementedError

    def get_greedy_action(self):
        NotImplementedError

    def compute_loss(self):
        NotImplementedError

    def save_weights(self):
        NotImplementedError

    def load_weights(self):
        NotImplementedError
