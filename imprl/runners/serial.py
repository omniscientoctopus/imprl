def training_rollout(env, agent):

    done = False
    obs = env.reset()
    state = env.info["state"]
    agent.reset_episode()

    while not done:

        # select action
        # _args are additional values (such as action_prob) to be stored
        # in the replay buffer
        action, *_args = agent.select_action(obs, training=True)

        # step in the environment
        next_obs, reward, done, info = env.step(action)

        # store experience in replay buffer
        if hasattr(agent, 'collect_state_info'):
            next_state = info["state"]
            agent.process_experience(
                obs, state, *_args, next_obs, next_state, reward, done
            )
            # overwrite state
            state = next_state
        else:
            agent.process_experience(obs, *_args, next_obs, reward, done)

        # overwrite obs
        obs = next_obs

    # Total life cycle cost
    return -agent.episode_return


def evaluate_agent(env, agent):

    done = False
    obs = env.reset()
    agent.reset_episode(training=False)

    while not done:

        # select action
        action = agent.select_action(obs, training=False)

        # step in the environment
        next_obs, reward, done, _ = env.step(action)

        # process rewards
        agent.process_rewards(reward)

        # overwrite obs
        obs = next_obs

    # Total life cycle cost
    return -agent.episode_return


def evaluate_heuristic(env, heuristic):

    done = False
    _ = env.reset()
    obs = env.info["observation"]
    episode_return = 0
    time = 0

    while not done:

        # select action
        action = heuristic.policy(obs)

        # step in the environment
        _, reward, done, info = env.step(action)

        episode_return += reward

        # overwrite obs
        obs = info["observation"]
        time += 1

    # Total life cycle cost
    return -episode_return
