class Policy:

    def __init__(self, inspection_interval, policy) -> None:
        """

        Policy: [dt, a(s1), a(s2), .... , a(sn)],
                n = |S|, dt = inspection interval
                a(sn) is action in state n

        """

        self.inspection_interval = inspection_interval
        # action for each observation
        self.policy = policy

    def __call__(self, time, observation):

        num_components = len(observation)
        action = [0] * num_components

        inspected_components = []

        for c in range(num_components):

            # check for failed components
            # replace if failed
            if observation[c] == 3:
                action[c] = 1

            # take action when inspecting
            elif time > 0 and time % self.inspection_interval == 0:

                # store inspected component to compute cost
                inspected_components.append(c)

                action[c] = self.policy[observation[c]]

        return action, inspected_components


class TimePeriodicInspectionConditionBasedMaintenance:

    def __init__(self, env) -> None:
        self.env = env
        self.policy_space = self.get_policy_space(env.time_horizon)

    @staticmethod
    def get_policy_space(time_horizon):

        policy_space = []

        action_state1 = 0  # actions in damage state 1 --> do-nothing
        action_state4 = 1  # actions in damage state 3 --> repair

        for inspection_interval in range(1, time_horizon + 1):
            for action_state2 in [
                0,
                1,
            ]:  # actions in damage state 2 --> [do-nothing, repair]
                for action_state3 in range(
                    action_state2, 2
                ):  # actions in damage state 2 --> atleast as much as state2

                    # define the policy by inspection_interval and actions to take in various states
                    _policy = Policy(
                        inspection_interval,
                        [action_state1, action_state2, action_state3, action_state4],
                    )

                    # add to policy space
                    policy_space.append(_policy)

        num_policies = len(policy_space)

        print(f"Number of policies: {num_policies}")

        return policy_space

    @staticmethod
    def rollout(env, policy):

        _ = env.reset()

        # initial observation
        observation = env.info["observation"]

        time = 0
        done = False
        episode_reward = 0

        while not done:

            # compute actions using policy
            action, inspected_components = policy(time, observation)

            # step in the environment
            _, reward, done, info = env.step(action)

            # if inspection took place
            if inspected_components:
                inspection_reward = (
                    env.discount_factor**time
                    * env.rewards_table[inspected_components, 0, 2].sum()
                )
            else:
                inspection_reward = 0

            observation = info["observation"]

            episode_reward += reward + inspection_reward

            time += 1

        return -episode_reward
