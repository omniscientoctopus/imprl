class FailureReplace():

    def __init__(self, env) -> None:
        self.env = env
        self.n_damage_states = env.n_damage_states

    def policy(self, observation):

        # replace if in last damage state
        action = [1 if obs == self.n_damage_states-1 else 0 for obs in observation]

        return action
