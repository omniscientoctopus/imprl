import itertools
import random

import numpy as np
import torch
torch.set_default_dtype(torch.float64)

from imprl.agents.primitives.Value_agent import ValueAgent
from imprl.agents.primitives.MLP import NeuralNetwork
from imprl.agents.utils import preprocess_inputs, _get_from_device

class DDQNAgent(ValueAgent):

    def __init__(self, env, config, device):

        super().__init__(env, config, device)
        
        # Compute joint action space
        # enumerate all possible action combinations
        action_space = list(
            itertools.product(np.arange(env.n_comp_actions), repeat=env.n_components)
        )
        self.system_action_space = [list(action) for action in action_space]

        # Neural network parameters
        n_inputs = self.n_damage_states * self.n_components + 1  # shape: system_states + 1
        self.n_joint_actions = self.n_comp_actions**self.n_components  # output

        self.network_config['architecture'] = [n_inputs] + self.network_config['hidden_layers'] + [self.n_joint_actions]

        # initialise Q network and target network
        self.q_network = NeuralNetwork( self.network_config["architecture"],
                                        optimizer=self.network_config["optimizer"],
                                        learning_rate=self.network_config["lr"],
                                        lr_scheduler=self.network_config["lr_scheduler"]).to(device)

        self.target_network = NeuralNetwork(self.network_config["architecture"]).to(device)
        # set weights equal
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Initialization
        self.target_network_reset = config['TARGET_NETWORK_RESET']

        self.learning_log = {"TD_loss": None,
                            "learning_rate": self.network_config["lr"]}

    def get_random_action(self):
        
        idx_action = random.randint(0, self.n_joint_actions-1)
        action = self.system_action_space[idx_action]

        return action, idx_action

    def get_greedy_action(self, observation, training):

        # compute Q values
        t_states = preprocess_inputs(observation, batch_size=1).to(self.device)
        q_values = self.q_network.forward(t_states, training)

        max_indices = torch.nonzero(q_values == q_values.max()).flatten()
        idx_action = random.choice(_get_from_device(max_indices))
        
        action = self.system_action_space[idx_action]

        if training:
            return action, idx_action
        else:
            return action

    def compute_current_values(self, t_observations, t_idx_actions):

        """
        Compute current Q-values for given observations and actions.

        Parameters
        ----------
        t_observations : torch.Tensor
            (batch_size, num_damage_states, num_components)

        idx_actions : list
            (batch_size)

        Returns
        -------
        value : torch.Tensor
            (batch_size, 1)

        """

        q_values = self.q_network.forward(t_observations)

        return torch.gather(q_values, dim=1, index=t_idx_actions)


    def get_future_values(self, t_next_beliefs):

        """
        Compute future values for Q-learning.

        Parameters
        ----------
        t_next_observation : torch.Tensor
            (batch_size, num_damage_states, num_components)

        Returns
        -------
        future_values : torch.Tensor
            (batch_size, 1)      
        
        """
        
        # compute Q-values using Q-network
        q_values = self.q_network.forward(t_next_beliefs).detach()

        # compute argmax over actions
        t_idx_best_actions = torch.argmax(q_values, axis=1, keepdim=True)

        # compute Q-values using *target* network
        target_q_values = self.target_network.forward(t_next_beliefs).detach()

        # select values correspoding to best actions
        future_values = torch.gather(target_q_values, dim=1, index=t_idx_best_actions)

        return future_values.detach()

    def compute_loss(self, *args):

        """
        Compute loss for Q-learning.

        Parameters
        ----------
        beliefs : list
            list of tuples (t, (num_damage_states, num_components))

        idx_actions : list
            list of ints

        next_beliefs : list
            list of tuples (t+1, (num_damage_states, num_components))

        rewards : list
            list of floats

        dones : list
            list of bools

        Returns
        -------
        loss : torch.Tensor

        """

        # preprocess inputs
        (t_beliefs, t_idx_actions, 
         t_next_beliefs, t_rewards, t_dones) = self._preprocess_inputs(*args)

        td_targets = self.compute_td_target(t_next_beliefs, t_rewards, t_dones)

        current_values = self.compute_current_values(t_beliefs, t_idx_actions)

        loss = self.q_network.loss_function(current_values, td_targets)

        return loss

    def save_weights(self, path, episode):
        torch.save(self.q_network.state_dict(), f"{path}/q_network_{episode}.pth")

    def load_weights(self, path, episode):
        self.q_network.load_state_dict(torch.load(f"{path}/q_network_{episode}.pth", map_location=torch.device('cpu')))