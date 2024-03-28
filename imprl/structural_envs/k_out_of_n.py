import numpy as np
import gym


class KOutOfN:

    def __init__(
        self,
        env_config: dict,
        baselines: dict = None,
    ) -> None:

        self.k = env_config["k"]
        self.time_horizon = env_config["time_horizon"]
        self.discount_factor = env_config["discount_factor"]
        self.FAILURE_PENALTY_FACTOR = env_config["failure_penalty_factor"]
        self.n_components = env_config["n_components"]
        self.n_damage_states = env_config["n_damage_states"]
        self.n_comp_actions = env_config["n_comp_actions"]

        # Transition model

        # shape: (n_components, n_damage_states, n_damage_states)
        self.deterioration_table = np.array(env_config["transition_model"])

        self.replacement_table = np.zeros(
            (self.n_components, self.n_damage_states, self.n_damage_states)
        )

        for c in range(self.n_components):
            r = env_config["replacement_accuracies"][c]
            self.replacement_table[c] = np.array(
                [[1, 0, 0, 0], [r, 1 - r, 0, 0], [r, 0, 1 - r, 0], [r, 0, 0, 1 - r]]
            )

        self.transition_model = np.zeros(
            (
                self.n_components,
                self.n_comp_actions,
                self.n_damage_states,
                self.n_damage_states,
            )
        )

        for c in range(self.n_components):

            # do nothing: deterioration
            self.transition_model[c, 0, :, :] = self.deterioration_table[c, :, :]

            # replacement: replace instantly + deterioration
            # D^T @ R^T @ belief ==> (R @ D)^T @ belief
            self.transition_model[c, 1, :, :] = (
                self.replacement_table[c] @ self.deterioration_table[c, :, :]
            )

            # inspect: deterioration
            self.transition_model[c, 2, :, :] = self.deterioration_table[c, :, :]

        # Reward model
        self.rewards_table = np.zeros(
            (self.n_components, self.n_damage_states, self.n_comp_actions)
        )

        self.rewards_table[:, :, 1] = np.array(
            [env_config["replacement_rewards"]] * self.n_damage_states
        ).T
        self.rewards_table[:, :, 2] = np.array(
            [env_config["inspection_rewards"]] * self.n_damage_states
        ).T

        self.system_replacement_reward = sum(env_config["replacement_rewards"])

        # Observation model
        self.inspection_model = np.zeros(
            (self.n_components, self.n_damage_states, self.n_damage_states)
        )

        for c in range(self.n_components):
            p = env_config["obs_accuracies"][c]
            self.inspection_model[c] = np.array(
                [
                    [p, 1 - p, 0.0, 0.0],
                    [(1 - p) / 2, p, (1 - p) / 2, 0.0],
                    [0.0, 1 - p, p, 0.0],
                    [0.0, 0.0, 0.0, 1],
                ]
            )

        self.failure_obs_model = np.array(
            [
                [1 / 3, 1 / 3, 1 / 3, 0.0],
                [1 / 3, 1 / 3, 1 / 3, 0.0],
                [1 / 3, 1 / 3, 1 / 3, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        self.observation_model = np.zeros(
            (
                self.n_components,
                self.n_comp_actions,
                self.n_damage_states,
                self.n_damage_states,
            )
        )

        for c in range(self.n_components):

            # do nothing: only failure observation
            self.observation_model[c, 0, :, :] = self.failure_obs_model

            # replacement: only failure observation
            self.observation_model[c, 1, :, :] = self.failure_obs_model

            # inspection: inspection
            self.observation_model[c, 2, :, :] = self.inspection_model[c]

        # Gym Spaces
        self.observation_space = gym.spaces.Tuple(
            (
                # normalized time
                gym.spaces.Box(0, 1, shape=(1,)),
                # belief over damage states
                gym.spaces.Box(
                    low=0, high=1, shape=(self.n_damage_states, self.n_components)
                ),
            )
        )

        self.action_space = gym.spaces.MultiDiscrete(
            np.ones(self.n_components, dtype=int) * self.n_comp_actions
        )

        self.state = self.reset()

        self.baselines = baselines

    def get_reward(
        self, state: list, action: list
    ) -> tuple[float, float, float, float]:

        reward_penalty = 0
        reward_replacement = 0
        reward_inspection = 0

        for c in range(self.n_components):

            if action[c] == 1:
                reward_replacement += self.rewards_table[c, state[c], action[c]]

            elif action[c] == 2:
                reward_inspection += self.rewards_table[c, state[c], action[c]]

        # check number of failed components
        _temp = state // (self.n_damage_states - 1)  # 0 if working, 1 if failed
        n_working = self.n_components - np.sum(_temp)  # number of working components

        functional = True if n_working >= self.k else False

        if not functional:
            reward_penalty = (
                self.FAILURE_PENALTY_FACTOR * self.system_replacement_reward
            )

        # discounted reward
        _discount_factor = self.discount_factor ** self.time
        reward_replacement *= _discount_factor
        reward_inspection *= _discount_factor
        reward_penalty *= _discount_factor

        reward = reward_replacement + reward_inspection + reward_penalty

        return reward, reward_replacement, reward_inspection, reward_penalty

    def get_next_state(self, state: np.array, action: list) -> np.array:

        _next_states = np.zeros(self.n_components, dtype=int)

        for c in range(self.n_components):

            next_damage_state = np.random.choice(
                np.arange(self.n_damage_states),
                p=self.transition_model[c, action[c], state[c], :],
            )

            _next_states[c] = next_damage_state

        return _next_states

    def get_observation(self, nextstate: list, action: list) -> np.array:

        _observations = np.zeros(self.n_components, dtype=int)

        for c in range(self.n_components):

            obs = np.random.choice(
                np.arange(self.n_damage_states),
                p=self.observation_model[c, action[c], nextstate[c], :],
            )

            _observations[c] = obs

        return _observations

    def belief_update(
        self, belief: np.array, action: list, observation: list
    ) -> np.array:

        next_belief = np.empty((self.n_damage_states, self.n_components))

        for c in range(self.n_components):

            belief_c = belief[:, c]

            # transition model
            belief_c = self.transition_model[c, action[c]].T @ belief_c

            # observation model
            state_probs = self.observation_model[c, action[c], :, observation[c]]
            belief_c = state_probs * belief_c

            # normalise
            belief_c = belief_c / np.sum(belief_c)

            next_belief[:, c] = belief_c

        return next_belief

    def step(self, action: list) -> tuple[np.array, float, bool, dict]:

        # collect reward: R(s,a)
        reward, reward_replacement, reward_inspection, reward_penalty = self.get_reward(
            self.damage_state, action
        )

        # compute next damage state
        next_state = self.get_next_state(self.damage_state, action)
        self.damage_state = next_state

        # compute observation
        self.observation = self.get_observation(next_state, action)

        # update belief
        self.belief = self.belief_update(self.belief, action, self.observation)

        # update time
        self.time += 1

        # check if terminal state
        done = True if self.time == self.time_horizon else False

        # update info dict
        self.info["system_failure"] = reward_penalty < 0
        self.info["reward_replacement"] = reward_replacement
        self.info["reward_inspection"] = reward_inspection
        self.info["reward_penalty"] = reward_penalty
        self.info["observation"] = self.observation

        return self._get_observation(), reward, done, self.info

    def reset(self) -> tuple[np.array, np.array]:

        # In the initial state, all components are undamaged
        initial_damage = 0
        self.damage_state = np.array([initial_damage] * self.n_components, dtype=int)
        self.observation = np.array([initial_damage] * self.n_components, dtype=int)
        belief = np.zeros((self.n_damage_states, self.n_components))
        belief[initial_damage, :] = 1
        self.belief = belief

        # reset the time
        self.time = 0

        self.info = {
            "system_failure": False,
            "reward_replacement": 0,
            "reward_inspection": 0,
            "reward_penalty": 0,
            "observation": self.observation,
        }

        return self._get_observation()

    def _get_observation(self) -> tuple[np.array, np.array]:

        # compute normalised time [0, 1]
        norm_time = self.time / self.time_horizon

        return (np.array([norm_time]), self.belief)
