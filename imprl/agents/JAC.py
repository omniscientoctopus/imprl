import itertools
import random

import numpy as np
import torch

from imprl.agents.primitives.PG_agent import PolicyGradientAgent as PGAgent
from imprl.agents.primitives.ActorNetwork import ActorNetwork
from imprl.agents.primitives.MLP import NeuralNetwork
from imprl.agents.utils import _get_from_device, preprocess_inputs


class JointActorCritic(PGAgent):

    def __init__(self, env, config, device):

        super().__init__(env, config, device)

        # compute joint action space
        # enumerate all possible action combinations
        action_space = list(
            itertools.product(np.arange(env.n_comp_actions), repeat=env.n_components)
        )
        self.system_action_space = [list(action) for action in action_space]
        self.n_joint_actions = self.n_comp_actions**self.n_components

        ## Neural Networks
        n_inputs = self.n_damage_states * self.n_components + 1  # shape: system_states + time
        n_outputs_actor = self.n_joint_actions
        n_outputs_critic = 1

        self.actor_config['architecture'] = [n_inputs] + self.actor_config['hidden_layers'] + [n_outputs_actor]
        self.critic_config['architecture'] = [n_inputs] + self.critic_config['hidden_layers'] + [n_outputs_critic]

        # Actors (info centralised: can observe the entire system state/belief)
        self.actor = ActorNetwork(self.actor_config["architecture"], 
                                  initialization='orthogonal',
                                  optimizer=self.actor_config["optimizer"], 
                                  learning_rate=self.actor_config["lr"],
                                  lr_scheduler=self.actor_config["lr_scheduler"]).to(device)

        # Critic (info centralised: can observe entire system state/belief)
        self.critic = NeuralNetwork(self.critic_config["architecture"], 
                                    initialization='orthogonal',
                                    optimizer=self.critic_config["optimizer"], 
                                    learning_rate=self.critic_config["lr"],
                                    lr_scheduler=self.critic_config["lr_scheduler"]).to(device)

    def get_random_action(self):

        idx_action = random.randint(0, self.n_joint_actions-1)
        action = self.system_action_space[idx_action]
        t_idx_action = torch.tensor(idx_action)
        action_prob = 1 / self.n_joint_actions
        
        return action,t_idx_action,action_prob

    def get_greedy_action(self, observation, training):

        # get action from actor network
        t_observation = preprocess_inputs(observation, 1).to(self.device)
        action_dist = self.actor.forward(t_observation, training)
        t_idx_action = action_dist.sample() # index of joint action

        # get action from index
        idx_action = _get_from_device(t_idx_action)
        action = self.system_action_space[idx_action]

        if training:
            log_prob = action_dist.log_prob(t_idx_action)
            action_prob = torch.exp(log_prob)
            return action, t_idx_action, action_prob
        else:
            return action

    def get_future_values(self, t_next_beliefs):

        # bootstrapping
        future_values = self.critic.forward(t_next_beliefs)

        return future_values

    def compute_log_prob(self, t_beliefs, t_idx_actions):

        # shape: (batch_size, n_actions)
        action_dists = self.actor.forward(t_beliefs)
    
        # compute log prob of each action under current policy
        # shape: (batch_size)
        _log_probs = action_dists.log_prob(t_idx_actions)

        return _log_probs

    def compute_sample_weight(self, log_probs, t_action_probs):

        new_probs = torch.exp(log_probs)

        # true dist / proposal dist
        weights = new_probs / t_action_probs

        # truncate weights to reduce variance
        weights = torch.clamp(weights, max=2)

        return weights.detach()

    def compute_loss(self, *args):

        (t_beliefs, t_next_beliefs, t_dones, 
        t_rewards, t_idx_actions, t_action_probs) = self._preprocess_inputs(*args)

        current_values = self.critic.forward(t_beliefs)
        td_targets = self.compute_td_target(t_next_beliefs, t_rewards, t_dones)

        t_log_probs = self.compute_log_prob(t_beliefs, t_idx_actions)

        weights = self.compute_sample_weight(t_log_probs.detach(), t_action_probs)

        # TD_target = r_t + γ V(b')
        # L_V(theta) = E[w (TD_target - V(b))^2]
        critic_loss = torch.mean(weights * torch.square(current_values - td_targets))

        advantage = (td_targets - current_values).detach().flatten()

        # max  (log_prob * advantage)
        # min -(log_prob * advantage)
        actor_loss = torch.mean(-t_log_probs * advantage * weights)

        return actor_loss, critic_loss

    def _preprocess_inputs(self, beliefs, actions, action_probs, next_beliefs, rewards, dones):

        t_beliefs = preprocess_inputs(beliefs, self.batch_size).to(self.device)
        t_next_beliefs = preprocess_inputs(next_beliefs, self.batch_size).to(self.device)
        t_dones = torch.tensor(np.asarray(dones).astype(int)).reshape(-1, 1).to(self.device)
        t_rewards = torch.tensor(rewards).reshape(-1, 1).to(self.device)
        t_action_probs = torch.tensor(action_probs).to(self.device)
        
        # difference from parent class
        # actions combined using torch.tensor instead of torch.stack
        t_actions = torch.tensor(actions).to(self.device)
        
        return t_beliefs,t_next_beliefs,t_dones,t_rewards,t_actions,t_action_probs

    def save_weights(self, path, episode):
        torch.save(self.actor.state_dict(), f"{path}/actor_{episode}.pth")
        torch.save(self.critic.state_dict(), f"{path}/critic_{episode}.pth")

    def load_weights(self, path, episode):
        
        # load actor weights
        full_path = f'{path}/actor_{episode}.pth'
        self.actor.load_state_dict(torch.load(full_path, map_location=torch.device('cpu')))
        
        # load critic weights
        full_path = f'{path}/critic_{episode}.pth'
        self.critic.load_state_dict(torch.load(full_path, map_location=torch.device('cpu')))