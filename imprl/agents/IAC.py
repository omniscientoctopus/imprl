import numpy as np
import torch

from imprl.agents.primitives.PG_agent import PolicyGradientAgent as PGAgent
from imprl.agents.primitives.MultiAgentActors import MultiAgentActors
from imprl.agents.primitives.MultiAgentCritics import MultiAgentCritics
from imprl.agents.utils import preprocess_inputs


class IndependentActorCritic(PGAgent):

    def __init__(self, env, config, device):

        super().__init__(env, config, device)

        ## Neural networks
        n_inputs = self.n_damage_states + 1  # shape: component_states + time
        n_outputs_actor = self.n_comp_actions
        n_outputs_critic = 1

        self.actor_config["architecture"] = (
            [n_inputs] + self.actor_config["hidden_layers"] + [n_outputs_actor]
        )
        self.critic_config["architecture"] = (
            [n_inputs] + self.critic_config["hidden_layers"] + [n_outputs_critic]
        )

        # Actors
        # (decentralised: can only observe component state/belief)
        self.actor = MultiAgentActors(
            self.n_components, self.n_comp_actions, self.actor_config, device
        )

        # Critics (decentralised: can observe only component state/belief)
        self.critic = MultiAgentCritics(self.n_components, self.critic_config, device)

    def get_greedy_action(self, observation, training):

        # convert to tensor
        t_observation = preprocess_inputs(observation, 1).to(self.device)
        t_ma_obs = self.get_multiagent_obs(t_observation).to(self.device)

        return self.actor.forward(t_ma_obs, training=training, ind_obs=True)

    def get_future_values(self, t_ma_next_beliefs):

        # bootstrapping
        # shape: (batch_size, num_components)
        future_values = self.critic.forward(
            t_ma_next_beliefs, training=True
        )  # shape: (batch_size, num_components)

        return future_values

    def compute_log_prob(self, t_ma_beliefs, t_actions):

        _log_probs = torch.ones((self.batch_size, self.n_components)).to(self.device)

        # get actions from each actor network
        for k, actor_network in enumerate(self.actor.networks):
            action_dists = actor_network.forward(t_ma_beliefs[:, k, :])

            # compute log prob of each action under current policy
            # shape: (batch_size)
            _log_probs[:, k] = action_dists.log_prob(t_actions[:, k])

        return _log_probs

    def compute_sample_weight(self, joint_log_probs, joint_action_probs):

        new_probs = torch.exp(joint_log_probs)

        # true dist / proposal dist
        weights = new_probs / joint_action_probs

        # truncate weights to reduce variance
        weights = torch.clamp(weights, max=2)

        # shape: (batch_size, 1)
        return weights.detach().reshape(-1, 1)

    def compute_loss(self, *args):

        (
            t_ma_beliefs,
            t_actions,
            t_action_probs,
            t_ma_next_beliefs,
            t_rewards,
            t_dones,
        ) = self._preprocess_inputs(*args)

        # input shape: (batch_size)
        # output shape: (batch_size, num_damage_states+1, n_components)n_components
        current_values = self.critic.forward(
            t_ma_beliefs, training=True
        )  # shape: (batch_size, num_components)
        td_targets = self.compute_td_target(
            t_ma_next_beliefs, t_rewards, t_dones
        )  # shape: (batch_size, n_components)

        advantage = (
            td_targets - current_values
        ).detach()  # shape: (batch_size, n_components)

        # compute log_prob actions
        # shape: (batch_size, num_components)
        t_log_probs = self.compute_log_prob(t_ma_beliefs, t_actions)

        # compute joint probs
        # shape: (batch_size)
        t_joint_log_probs = torch.sum(t_log_probs, dim=-1)

        # compute importance sampling weights
        # shape: (batch_size, 1)
        weights = self.compute_sample_weight(t_joint_log_probs.detach(), t_action_probs)

        # L_V(theta) = E[w (TD_target - V(b))^2]
        # weights: (B, 1), current_values: (B, M), td_targets: (B, M)
        # compute the BMSE across batches
        # weights * (current_values - td_targets)^2
        # (B, 1)  * ((B, M) - (B, M))^2 => (B, M)
        # we take the mean across batches and add losses of all critics
        critic_loss = torch.mean(
            weights * torch.square(current_values - td_targets), dim=0
        ).sum()

        # t_log_probs @ advantage.T
        # (B, M) * (B, M) => (B, M)
        # torch.sum(B, M, dim=1,keepdim=True) => (B, 1)
        # torch.mean(-(B, 1) * (B, 1)) => scalar
        actor_loss = torch.mean(
            -torch.sum(t_log_probs * advantage, dim=1, keepdim=True) * weights
        )

        return actor_loss, critic_loss

    def _preprocess_inputs(
        self, beliefs, actions, action_probs, next_beliefs, rewards, dones
    ):

        t_beliefs = preprocess_inputs(beliefs, self.batch_size).to(self.device)
        t_next_beliefs = preprocess_inputs(next_beliefs, self.batch_size).to(
            self.device
        )
        t_dones = (
            torch.tensor(np.asarray(dones).astype(int)).reshape(-1, 1).to(self.device)
        )
        t_rewards = torch.tensor(rewards).reshape(-1, 1).to(self.device)
        t_actions = torch.stack(actions).to(self.device)
        t_action_probs = torch.tensor(action_probs).to(self.device)

        # output shape: (batch_size, num_damage_states+1, num_components)
        t_ma_beliefs = self.get_multiagent_obs(t_beliefs, self.batch_size).to(
            self.device
        )
        t_ma_next_beliefs = self.get_multiagent_obs(t_next_beliefs, self.batch_size).to(
            self.device
        )

        return (
            t_ma_beliefs,
            t_actions,
            t_action_probs,
            t_ma_next_beliefs,
            t_rewards,
            t_dones,
        )

    def save_weights(self, path, episode):

        for c in range(self.n_components):
            actor_network = self.actor.networks[c]
            torch.save(actor_network.state_dict(), f"{path}/actor_{c+1}_{episode}.pth")

            critic_network = self.critic.networks[c]
            torch.save(
                critic_network.state_dict(), f"{path}/critic_{c+1}_{episode}.pth"
            )

    def load_weights(self, path, episode):

        for c in range(self.n_components):

            actor_network = self.actor.networks[c]
            actor_network.load_state_dict(
                torch.load(
                    f"{path}/actor_{c+1}_{episode}.pth",
                    map_location=torch.device("cpu"),
                )
            )

            critic_network = self.critic.networks[c]
            critic_network.load_state_dict(
                torch.load(
                    f"{path}/critic_{c+1}_{episode}.pth",
                    map_location=torch.device("cpu"),
                )
            )

    def evaluate_critic(self, beliefs, batch_size=1):

        # preprocess inputs
        t_beliefs = preprocess_inputs(beliefs, batch_size).to(self.device)

        # if critic is decentralized
        t_ma_beliefs = self.get_multiagent_obs(t_beliefs, batch_size).to(self.device)
        values = self.get_future_values(t_ma_beliefs)

        return values.detach()
