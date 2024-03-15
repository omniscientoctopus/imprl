import numpy as np
import torch

from imprl.agents.primitives.PG_agent import PolicyGradientAgent as PGAgent
from imprl.agents.primitives.ActorNetwork import ActorNetwork
from imprl.agents.primitives.MLP import NeuralNetwork
from imprl.agents.utils import _get_from_device, preprocess_inputs


class IndependentActorCriticParameterSharing(PGAgent):

    def __init__(self, env, config, device):

        super().__init__(env, config, device)

        ## Neural networks
        n_inputs = self.n_components + self.n_damage_states + 1 # shape: n_components (id) + damage_states + time
        n_outputs_actor = self.n_comp_actions
        n_outputs_critic = 1

        self.actor_config['architecture'] = [n_inputs] + self.actor_config['hidden_layers'] + [n_outputs_actor]
        self.critic_config['architecture'] = [n_inputs] + self.critic_config['hidden_layers'] + [n_outputs_critic] 

        # Actor
        # (decentralised: can only observe component state/belief)
        # but parameters are shared
        # actions for individual component
        self.actor = ActorNetwork(self.actor_config['architecture'],
                                initialization='orthogonal',
                                optimizer=self.actor_config['optimizer'],
                                learning_rate=self.actor_config['lr'],
                                lr_scheduler=self.actor_config['lr_scheduler']).to(device)

        # Critic 
        # decentralised: can only observe component state/belief
        # but parameters are shared
        self.critic = NeuralNetwork(self.critic_config['architecture'],
                                    initialization='orthogonal',
                                    optimizer=self.critic_config['optimizer'],
                                    learning_rate=self.critic_config['lr'],
                                    lr_scheduler=self.critic_config['lr_scheduler']).to(device) 

    def get_greedy_action(self, observation, training):

        # convert to tensor
        t_observation = preprocess_inputs(observation, 1).to(self.device)
        t_ma_obs = self.get_multiagent_obs_with_idx(t_observation).to(self.device)

        action_dist = self.actor.forward(t_ma_obs, training)
        t_action = action_dist.sample()
        action = _get_from_device(t_action)

        if training:
            log_prob = action_dist.log_prob(t_action)
            action_prob = torch.prod(torch.exp(log_prob), dim=-1) # joint action prob
            return action, t_action, action_prob
        else:
            return action

    def get_future_values(self, t_next_beliefs):

        # get future values
        # shape: (batch_size, num_components)
        future_values = self.critic.forward(t_next_beliefs).squeeze()

        return future_values

    def compute_log_prob(self, t_ma_beliefs, t_actions):

        # get actions dists from each actor
        # logits shape: (batch_size, num_components, num_actions)
        action_dists = self.actor.forward(t_ma_beliefs)

        # compute log prob of each action under current policy
        # shape: (batch_size, num_components)
        _log_probs = action_dists.log_prob(t_actions)

        return _log_probs

    def compute_sample_weight(self, joint_log_probs, joint_action_probs):

        new_probs = torch.exp(joint_log_probs)

        # true dist / proposal dist
        weights = new_probs / joint_action_probs

        # truncate weights to reduce variance
        weights = torch.clamp(weights, max=2)

        return weights.detach().reshape(-1, 1)

    def compute_loss(self, *args):

        # preprocess inputs
        (t_ma_beliefs, t_actions, t_action_probs, 
         t_ma_next_beliefs, t_rewards, t_dones) = self._preprocess_inputs(*args)
                
        # Value function update
        current_values = self.critic.forward(t_ma_beliefs).squeeze() # shape: (batch_size, num_components)
        td_targets = self.compute_td_target(t_ma_next_beliefs, t_rewards, t_dones) # shape: (batch_size, num_components)
        
        advantage = (td_targets - current_values).detach() # shape: (batch_size, num_components)

        # compute log_prob actions
        # shape: (batch_size, n_components)
        t_log_probs = self.compute_log_prob(t_ma_beliefs, t_actions)

        # compute joint probs
        # sum over all actions for each component
        # shape: (batch_size)
        t_joint_log_probs = torch.sum(t_log_probs, dim=-1)

        # shape: (batch_size, 1)
        weights = self.compute_sample_weight(t_joint_log_probs.detach(), t_action_probs)

        # L_V(theta) = E[w (TD_target - V(b))^2]
        # weights: (B, 1), current_values: (B, M), td_targets: (B, M)
        # compute the BMSE across batches
        # weights * (current_values - td_targets)^2
        # (B, 1)  * ((B, M) - (B, M))^2 => (B, M)
        # we take the mean across batches and add losses of all critics
        critic_loss = torch.mean(torch.square(current_values - td_targets) * weights, dim=0).sum()

        # t_log_probs @ advantage.T
        # (B, M) * (B, M) => (B, M)
        # torch.sum(B, M, dim=1,keepdim=True) => (B, 1)
        # torch.mean(-(B, 1) * (B, 1)) => (scalar)
        actor_loss = torch.mean(-torch.sum(t_log_probs * advantage, dim=1, keepdim=True) * weights)

        return actor_loss, critic_loss

    def _preprocess_inputs(self, beliefs, actions, action_probs, next_beliefs, rewards, dones):
        
        t_beliefs = preprocess_inputs(beliefs, self.batch_size).to(self.device)
        t_next_beliefs = preprocess_inputs(next_beliefs, self.batch_size).to(self.device)
        t_dones = torch.tensor(np.asarray(dones).astype(int)).double().reshape(-1, 1).to(self.device)
        t_rewards = torch.tensor(rewards).reshape(-1, 1).to(self.device)
        t_actions = torch.stack(actions).to(self.device)
        t_action_probs = torch.tensor(action_probs).to(self.device)

        # output shape: (batch_size, num_components, num_components+num_damage_states+1)
        t_ma_beliefs = self.get_multiagent_obs_with_idx(t_beliefs, self.batch_size).to(self.device)
        t_ma_next_beliefs = self.get_multiagent_obs_with_idx(t_next_beliefs, self.batch_size).to(self.device)

        return t_ma_beliefs, t_actions, t_action_probs, t_ma_next_beliefs, t_rewards, t_dones

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

    def evaluate_critic(self, beliefs, batch_size=1):

        # preprocess inputs
        t_beliefs = preprocess_inputs(beliefs, batch_size).to(self.device)

        # if critic is decentralized
        t_ma_beliefs = self.get_multiagent_obs_with_idx(t_beliefs, batch_size).to(self.device)
        values = self.get_future_values(t_ma_beliefs)

        return values.detach()