import numpy as np
import torch

from imprl.agents.primitives.agent import Agent
from imprl.agents.primitives.ActorNetwork import ActorNetwork
from imprl.agents.primitives.MLP import NeuralNetwork
from imprl.agents.utils import _get_from_device, preprocess_inputs


class AdvantageActorCritic(Agent):

    def __init__(self, env, config, device):

        super().__init__(env, config, device)

        ## Neural networks
        n_inputs = self.n_damage_states * self.n_components + 1  # shape: system_states + time
        n_outputs_actor = self.n_components * self.n_comp_actions 
        n_outputs_critic = 1

        self.actor_config['architecture'] = [n_inputs] + self.actor_config['hidden_layers'] + [n_outputs_actor]
        self.critic_config['architecture'] = [n_inputs] + self.critic_config['hidden_layers'] + [n_outputs_critic] 

        # Actors (centralised: can observe the entire system state/belief)
        self.actor = ActorNetwork(self.actor_config['architecture'],
                                self.n_components,
                                self.n_comp_actions,
                                initialization='orthogonal',
                                optimizer=self.actor_config['optimizer'],
                                learning_rate=self.actor_config['lr'],
                                lr_scheduler=self.actor_config['lr_scheduler']).to(device)

        # Critic (centralised: can observe entire system state)
        self.critic = NeuralNetwork(self.critic_config['architecture'],
                                    initialization='orthogonal',
                                    optimizer=self.critic_config['optimizer'],
                                    learning_rate=self.critic_config['lr'],
                                    lr_scheduler=self.critic_config['lr_scheduler']).to(device) 

    def get_random_action(self):
        
        action = self.env.action_space.sample()
        t_action = torch.tensor(action).to(self.device)
            # alternatively: (1/n_comp_actions)^(n_components)
        action_prob = np.prod(np.ones(self.n_components) / self.n_comp_actions)

        return action, t_action, action_prob

    def compute_td_target(self, next_beliefs, rewards, dones, batch_size):

        # bootstrapping
        t_next_beliefs = preprocess_inputs(next_beliefs, batch_size)
        future_values = self.critic_network.forward(t_next_beliefs)
        # set future values of done states to 0
        not_dones = 1 - np.asarray(dones).astype(int)
        future_values *= not_dones

        td_target = rewards + self.discount * future_values

        return td_target.detach()

    def compute_log_prob(self, beliefs, actions, batch_size):

        t_beliefs = preprocess_inputs(beliefs, batch_size)
        # actor network generates
        # a distribution over actions for each component
        # batch: beliefs
        action_dist = self.actor_network.forward(t_beliefs)
        log_prob = action_dist.log_prob(actions)

        return torch.sum(log_prob, dim=-1)

    def compute_loss(self, beliefs, actions, next_beliefs, rewards, dones, batch_size):

        t_beliefs = preprocess_inputs(beliefs, batch_size)
        current_values = self.critic_network.forward(t_beliefs)

        td_target = self.compute_td_target(next_beliefs, rewards, dones, batch_size)

        advantage = (td_target - current_values).detach()

        critic_loss = self.critic_network.loss_function(current_values, td_target)

        log_prob = self.compute_log_prob(beliefs, actions, batch_size)

        # max  (log_prob * advantage)
        # min -(log_prob * advantage)
        actor_loss = -log_prob * advantage 

        return actor_loss, critic_loss
