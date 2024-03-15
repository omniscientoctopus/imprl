import numpy as np

class FailureReplace():

    def __init__(self, env):

        self.n_components = env.n_components
        self.n_damage_states = env.n_damage_states
        self.discount = env.discount_factor

        # Initialization
        self.episode = 0
        self.total_time = 0  # total time steps in lifetime
        self.time = 0  # time steps in current episode
        self.episode_return = 0 # discounted return in current episode

    @staticmethod
    def policy(observation):
        action = 1 if observation == 1 else 0
        return action

    def reset_episode(self, training=False):

        self.episode_return = 0
        self.episode += 1
        self.time = 0

    def select_action(self, observation, training=False):

        _, beliefs = observation

        vpolicy = np.vectorize(self.policy)

        actions = vpolicy(beliefs[-1, :])

        return actions

    def process_rewards(self, reward):

        self.episode_return += self.discount ** self.time * reward

        # updating time here so that only this method needs to be called 
        # during inference
        self.time += 1
        self.total_time += 1