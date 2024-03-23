class Random:

    def __init__(self, env) -> None:

        self.env = env

    def policy(self, observation):

        return self.env.action_space.sample()
