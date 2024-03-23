import numpy as np
import torch

from imprl.agents.primitives.PG_agent import PolicyGradientAgent as PGAgent
from imprl.agents.primitives.MLP import NeuralNetwork
from imprl.agents.primitives.MultiAgentActors import MultiAgentActors
from imprl.agents.utils import preprocess_inputs


class IndependentActorCentralisedCritic(PGAgent):

    def __init__(self, env, config, device):

        super().__init__(env, config, device)

        ## Neural networks
        n_inputs_actor = self.n_damage_states + 1 # shape: damage_states + time
        n_inputs_critic = self.n_damage_states * self.n_components + 1  # shape: damage_states * n_components + time
        n_outputs_actor = self.n_comp_actions
        n_outputs_critic = 1

        self.actor_config['architecture'] = [n_inputs_actor] + self.actor_config['hidden_layers'] + [n_outputs_actor]
        self.critic_config['architecture'] = [n_inputs_critic] + self.critic_config['hidden_layers'] + [n_outputs_critic] 

        # Actors
        # (decentralised: can only observe component state/belief)
        # actions for individual component
        self.actor = MultiAgentActors(self.n_components, self.n_comp_actions, self.actor_config, device)

        # Critic (centralised: can observe entire system state)
        self.critic = NeuralNetwork(self.critic_config['architecture'],
                                    initialization='orthogonal',
                                    optimizer=self.critic_config['optimizer'],
                                    learning_rate=self.critic_config['lr'],
                                    lr_scheduler=self.critic_config['lr_scheduler']).to(device)

    def get_greedy_action(self, observation, training):

        # convert to tensor
        t_observation = preprocess_inputs(observation, 1).to(self.device)
        t_ma_obs = self.get_multiagent_obs(t_observation).to(self.device)

        return self.actor.forward(t_ma_obs, training=training, ind_obs=True)

    def get_future_values(self, t_next_beliefs):

        # bootstrapping
        # shape: (batch_size, 1)
        future_values = self.critic.forward(t_next_beliefs)

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

        # preprocess inputs
        (t_beliefs, t_ma_beliefs, t_actions, t_action_probs,
         t_next_beliefs, t_rewards, t_dones) = self._preprocess_inputs(*args)

        # Value function update
        current_values = self.critic.forward(t_beliefs) # shape: (batch_size, 1)
        td_targets = self.compute_td_target(t_next_beliefs, t_rewards, t_dones) # shape: (batch_size, 1)

        advantage = (td_targets - current_values).detach() # shape: (batch_size, 1)

        # compute log_prob actions
        # shape: (batch_size, num_components)
        t_log_probs = self.compute_log_prob(t_ma_beliefs, t_actions)

        # compute joint probs
        # sum over all actions for each component
        # shape: (batch_size)
        t_joint_log_probs = torch.sum(t_log_probs, dim=-1)

        # compute importance sampling weights
        # shape: (batch_size)
        weights = self.compute_sample_weight(t_joint_log_probs.detach(), t_action_probs)

        # TD_target = r_t + γ V(b')
        # L_V(theta) = E[w (TD_target - V(b))^2]
        critic_loss = torch.mean(weights * torch.square(current_values - td_targets))

        # t_log_probs @ advantage
        # (B, M) * (B, 1) => (B, M)
        # torch.sum(B, M, dim=1,keepdim=True) => (B, 1)
        # torch.mean(-(B, 1) * (B, 1)) => scalar
        actor_loss = torch.mean(-torch.sum(t_log_probs * advantage, dim=1, keepdim=True) * weights)

        return actor_loss, critic_loss

    def _preprocess_inputs(self, beliefs, actions, action_probs, next_beliefs, rewards, dones):
        
        t_beliefs = preprocess_inputs(beliefs, self.batch_size).to(self.device)
        t_next_beliefs = preprocess_inputs(next_beliefs, self.batch_size).to(self.device)
        t_dones = torch.tensor(np.asarray(dones).astype(int)).reshape(-1, 1).to(self.device)
        t_rewards = torch.tensor(rewards).reshape(-1, 1).to(self.device)
        t_actions = torch.stack(actions).to(self.device)
        t_action_probs = torch.tensor(action_probs).to(self.device)

        # output shape: (batch_size, num_damage_states+1, num_components)
        t_ma_beliefs = self.get_multiagent_obs(t_beliefs, self.batch_size).to(self.device)

        return t_beliefs,t_ma_beliefs,t_actions,t_action_probs,t_next_beliefs,t_rewards,t_dones

    def save_weights(self, path, episode):
    
        for i, actor_network in enumerate(self.actor.networks):
            torch.save(actor_network.state_dict(), f'{path}/actor_{i+1}_{episode}.pth')

        torch.save(self.critic.state_dict(), f'{path}/critic_{episode}.pth')

    def load_weights(self, path, episode):

        # load actor weights
        for i, actor_network in enumerate(self.actor.networks):
            actor_network.load_state_dict(torch.load(f'{path}/actor_{i+1}_{episode}.pth', map_location=torch.device('cpu')))

        # load critic weights
        self.critic.load_state_dict(torch.load(f'{path}/critic_{episode}.pth', map_location=torch.device('cpu')))