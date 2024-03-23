import numpy as np
import torch

torch.set_default_dtype(torch.float64)

from imprl.agents.primitives.agent import Agent
from imprl.agents.utils import preprocess_inputs


class PolicyGradientAgent(Agent):

    def __init__(self, env, config, device):

        super().__init__(env, config, device)

        self.actor_config = config["ACTOR_CONFIG"]
        self.critic_config = config["CRITIC_CONFIG"]

        # logger
        self.logger = {
            "critic_loss": None,
            "actor_loss": None,
            "lr_critic": self.critic_config["lr"],
            "lr_actor": self.actor_config["lr"],
        }

    def get_random_action(self):

        action = self.env.action_space.sample()
        t_action = torch.tensor(action).to(self.device)
        action_prob = torch.prod(torch.ones(self.n_components) / self.n_comp_actions)

        return action, t_action, action_prob

    def reset_episode(self, training=True):

        super().reset_episode(training)

        # if training and sufficient samples are available
        if training and self.total_time > 10 * self.batch_size:

            # update learning rate
            self.actor.lr_scheduler.step()
            self.critic.lr_scheduler.step()

            # logging
            self.logger["lr_actor"] = self.actor.lr_scheduler.get_last_lr()[0]
            self.logger["lr_critic"] = self.critic.lr_scheduler.get_last_lr()[0]

    def process_experience(
        self, belief, action, action_prob, next_belief, reward, done
    ):

        super().process_rewards(reward)

        # store experience in replay memory
        self.replay_memory.store_experience(
            belief, action, action_prob, next_belief, reward, done
        )

        # start batch learning once sufficient samples are available
        if self.total_time > 10 * self.batch_size:

            # sample batch of experiences from replay memory
            sample_batch = self.replay_memory.sample_batch(self.batch_size)

            # train actor and critic networks
            self.train(*sample_batch)

        if done:
            self.logger["episode"] = self.episode
            self.logger["episode_cost"] = -self.episode_return

    def train(self, beliefs, actions, action_probs, next_beliefs, rewards, dones):

        actor_loss, critic_loss = self.compute_loss(
            beliefs, actions, action_probs, next_beliefs, rewards, dones
        )

        ## Actor network
        # Zero gradients, perform a backward pass, and update the weights.
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        ## Critic network
        # Zero gradients, perform a backward pass, and update the weights.
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        # logging value update
        self.logger["actor_loss"] = actor_loss.detach()
        self.logger["critic_loss"] = critic_loss.detach()

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
        t_action_probs = torch.tensor(action_probs).to(self.device)
        t_actions = torch.stack(actions).to(self.device)

        return t_beliefs, t_next_beliefs, t_dones, t_rewards, t_actions, t_action_probs

    def evaluate_critic(self, beliefs, batch_size=1):

        # preprocess inputs
        t_beliefs = preprocess_inputs(beliefs, batch_size).to(self.device)

        # assumes critic is centralized
        values = self.get_future_values(t_beliefs)

        return values.detach()
