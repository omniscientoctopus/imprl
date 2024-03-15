import torch

from imprl.agents.primitives.PG_agent import PolicyGradientAgent as PGAgent
from imprl.agents.primitives.MLP import NeuralNetwork
from imprl.agents.primitives.MultiAgentActors import MultiAgentActors
from imprl.agents.utils import preprocess_inputs


class DeepDecentralisedMultiAgentActorCritic(PGAgent):

    def __init__(self, env, config, device):

        super().__init__(env, config, device)

        ## Neural networks
        n_inputs = self.n_damage_states * self.n_components + 1  # shape: system_states + time
        n_outputs_actor = self.n_comp_actions 
        n_outputs_critic = 1

        self.actor_config['architecture'] = [n_inputs] + self.actor_config['hidden_layers'] + [n_outputs_actor]
        self.critic_config['architecture'] = [n_inputs] + self.critic_config['hidden_layers'] + [n_outputs_critic] 

        # Actors 
        # (decentralised: can observe the entire system state/belief)
        # but unlike DCMAC, parameters are not shared
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

        return self.actor.forward(t_observation, training=training, ind_obs=False)

    def get_future_values(self, t_next_beliefs):

        # bootstrapping
        future_values = self.critic.forward(t_next_beliefs)

        return future_values

    def compute_log_prob(self, t_beliefs, t_actions):

        _log_probs = torch.ones((self.batch_size, self.n_components)).to(self.device)

        # get actions from each actor network
        for k, actor_network in enumerate(self.actor.networks):
            action_dists = actor_network.forward(t_beliefs)

            # compute log prob of each action under current policy
            # shape: (batch_size)
            _log_probs[:, k] = action_dists.log_prob(t_actions[:, k])

        # compute joint probs
        # shape: (batch_size)
        joint_log_probs = torch.sum(_log_probs, dim=-1)

        return joint_log_probs

    def compute_sample_weight(self, joint_log_probs, t_action_probs):

        new_probs = torch.exp(joint_log_probs)

        # true dist / proposal dist
        weights = new_probs / t_action_probs

        # truncate weights to reduce variance
        weights = torch.clamp(weights, max=2)

        return weights.detach()

    def compute_loss(self, *args):

        (t_beliefs, t_next_beliefs, t_dones, 
        t_rewards, t_actions, t_action_probs) = self._preprocess_inputs(*args)

        current_values = self.critic.forward(t_beliefs)
        td_targets = self.compute_td_target(t_next_beliefs, t_rewards, t_dones)

        t_log_probs = self.compute_log_prob(t_beliefs, t_actions)

        weights = self.compute_sample_weight(t_log_probs.detach(), t_action_probs)

        # TD_target = r_t + γ V(b')
        # L_V(theta) = E[w (TD_target - V(b))^2]
        critic_loss = torch.mean(weights * torch.square(current_values - td_targets))

        advantage = (td_targets - current_values).detach().flatten()

        # max  (log_prob * advantage)
        # min -(log_prob * advantage)
        actor_loss = torch.mean(-t_log_probs * advantage * weights)

        return actor_loss, critic_loss

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