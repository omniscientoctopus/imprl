import numpy as np
import torch

from imprl.agents.primitives.Value_agent import ValueAgent
from imprl.agents.primitives.MLP import NeuralNetwork
from imprl.agents.utils import preprocess_inputs


class ValueDecompositionNetworkParameterSharing(ValueAgent):

    def __init__(self, env, config, device):

        super().__init__(env, config, device)

        # Neural networks
        n_inputs = (
            self.n_components + self.n_damage_states + 1
        )  # shape: n_components (id) + damage_states + time

        self.network_config["architecture"] = (
            [n_inputs] + self.network_config["hidden_layers"] + [self.n_comp_actions]
        )

        self.q_network = NeuralNetwork(
            self.network_config["architecture"],
            initialization="orthogonal",
            optimizer=self.network_config["optimizer"],
            learning_rate=self.network_config["lr"],
            lr_scheduler=self.network_config["lr_scheduler"],
        ).to(device)

        self.target_network = NeuralNetwork(self.network_config["architecture"]).to(
            device
        )

        # set weights equal
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Initialization
        self.target_network_reset = config["TARGET_NETWORK_RESET"]

        self.logger = {
            "TD_loss": None,
            "learning_rate": self.network_config["lr"],
        }

    def get_random_action(self):
        action = self.env.action_space.sample()
        t_action = torch.tensor(action).to(self.device)

        return action, t_action

    def get_greedy_action(self, observation, training):

        # compute Q-values
        t_obs = preprocess_inputs(observation, 1).to(self.device)
        t_ma_obs = self.get_multiagent_obs_with_idx(t_obs).to(self.device)

        # shape: (n_components, n_comp_actions)
        q_values = self.q_network.forward(t_ma_obs, training).squeeze()
        t_action = torch.argmax(q_values, dim=-1)

        action = t_action.cpu().numpy()

        if training:
            return action, t_action
        else:
            return action

    def mixer(self, q_values):
        # input shape: (batch_size, n_components)
        return torch.sum(q_values, dim=1)

    def compute_current_values(self, t_ma_obs, t_actions):

        # shape: (batch_size, n_components, n_comp_actions)
        all_q_values = self.q_network.forward(t_ma_obs)

        q_values = torch.gather(all_q_values, 2, t_actions.unsqueeze(2))

        q_total = self.mixer(q_values)

        return q_total

    def get_future_values(self, t_ma_next_obs):

        # compute Q-values using Q-network
        # shape: (batch_size, n_components, n_comp_actions)
        q_values = self.target_network.forward(t_ma_next_obs).detach()

        # compute argmax_a Q(s', a)
        # shape: (batch_size, n_components)
        t_best_actions = torch.argmax(q_values, dim=2)

        # compute Q-values using *target* network
        # shape: (batch_size, n_components, n_comp_actions)
        target_q_values = self.target_network.forward(t_ma_next_obs).detach()

        # select values correspoding to best actions
        # shape: (batch_size, n_components)
        future_values = torch.gather(target_q_values, 2, t_best_actions.unsqueeze(2))

        q_total_future = self.mixer(future_values)

        return q_total_future.detach()

    def compute_loss(self, *args):

        # preprocess inputs
        t_ma_obs, t_actions, t_ma_next_obs, t_rewards, t_dones = (
            self._preprocess_inputs(*args)
        )

        td_target = self.compute_td_target(t_ma_next_obs, t_rewards, t_dones)

        current_values = self.compute_current_values(t_ma_obs, t_actions)

        loss = self.q_network.loss_function(current_values, td_target)

        return loss

    def _preprocess_inputs(self, beliefs, actions, next_beliefs, rewards, dones):

        t_beliefs = preprocess_inputs(beliefs, self.n_components).to(self.device)
        t_next_beliefs = preprocess_inputs(next_beliefs, self.n_components).to(
            self.device
        )
        t_actions = torch.stack(actions).to(self.device)
        t_dones = (
            torch.tensor(np.asarray(dones).astype(int)).reshape(-1, 1).to(self.device)
        )
        t_rewards = torch.tensor(rewards).reshape(-1, 1).to(self.device)
        t_ma_beliefs = self.get_multiagent_obs_with_idx(t_beliefs, self.batch_size).to(
            self.device
        )
        t_ma_next_beliefs = self.get_multiagent_obs_with_idx(
            t_next_beliefs, self.batch_size
        ).to(self.device)

        return t_ma_beliefs, t_actions, t_ma_next_beliefs, t_rewards, t_dones

    def save_weights(self, path, episode):
        torch.save(self.q_network.state_dict(), f"{path}/q_network_{episode}.pth")

    def load_weights(self, path, episode):
        full_path = f"{path}/q_network_{episode}.pth"
        self.q_network.load_state_dict(
            torch.load(full_path, map_location=torch.device("cpu"))
        )
