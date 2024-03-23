import numpy as np
import torch
torch.set_default_dtype(torch.float64)

from imprl.agents.primitives.agent import Agent
from imprl.agents.utils import preprocess_inputs

class ValueAgent(Agent):

    def __init__(self, env, config, device):

        super().__init__(env, config, device)

        self.network_config = config['NETWORK_CONFIG']

        self.learning_log = {"TD_loss": None,
                             "learning_rate": self.network_config["lr"]}

    def reset_episode(self, training=True):

        super().reset_episode(training)

        # if training and sufficient samples are available
        if training and self.total_time > 10 * self.batch_size:

            # set weights equal
            if self.episode % self.target_network_reset == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())

            # update learning rate
            self.q_network.lr_scheduler.step()

            # logging
            self.learning_log["learning_rate"] = self.q_network.lr_scheduler.get_last_lr()[0]

    def process_experience(self, belief, idx_action, next_belief, reward, done):

        super().process_rewards(reward)

        # store experience in replay memory
        self.replay_memory.store_experience(
            belief, idx_action, next_belief, reward, done
        )

        # start batch learning once sufficient samples are available
        if self.total_time > 10 * self.batch_size:
            sample_batch = self.replay_memory.sample_batch(self.batch_size)
            self.train(*sample_batch)

        if done:
            self.logger['episode'] = self.episode
            self.logger['episode_cost'] = -self.episode_return

    def train(self, *args):

        """
        Train Q-network using Q-learning.   

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

        """

        loss = self.compute_loss(*args)

        # Zero gradients, perform a backward pass, and update the weights.
        self.q_network.optimizer.zero_grad()
        loss.backward()
        self.q_network.optimizer.step()

        # logging value update
        self.logger["TD_loss"] = loss.detach()


    def _preprocess_inputs(self, beliefs, idx_actions, next_beliefs, rewards, dones):
        
        t_beliefs = preprocess_inputs(beliefs, batch_size=self.batch_size).to(self.device)
        t_idx_actions = torch.tensor(idx_actions).reshape(-1, 1).to(self.device)
        t_next_beliefs = preprocess_inputs(next_beliefs, batch_size=self.batch_size).to(self.device)
        t_dones = torch.tensor(np.asarray(dones).astype(int)).reshape(-1, 1).to(self.device)
        t_rewards = torch.tensor(rewards).reshape(-1, 1).to(self.device)

        return t_beliefs, t_idx_actions, t_next_beliefs, t_rewards, t_dones
