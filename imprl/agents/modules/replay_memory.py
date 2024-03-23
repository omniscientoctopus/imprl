import random
from collections import deque


class ReplayMemory:

    # ReplayMemory should store the last "size" experiences
    # and be able to return a randomly sampled batch of experiences
    def __init__(self, size):
        self.size = size
        self.memory = deque(maxlen=size)

    # Store experience in memory
    def store_experience(self, observation, action, next_observation, reward, done):
        self.memory.append((observation, action, next_observation, reward, done))

    # Randomly sample "batch_size" experiences from the memory and return them
    def sample_batch(self, batch_size):

        samples = random.sample(self.memory, batch_size)

        _obs = []
        _actions = []
        _next_obs = []
        _rewards = []
        _dones = []

        for sample in samples:
            _obs.append(sample[0])
            _actions.append(sample[1])
            _next_obs.append(sample[2])
            _rewards.append(sample[3])
            _dones.append(sample[4])

        return (_obs, _actions, _next_obs, _rewards, _dones)


class AbstractReplayMemory(ReplayMemory):
    def __init__(self, size):
        super().__init__(size)

    # store experience in memory
    def store_experience(self, *args):
        self.memory.append(args)

    # Randomly sample "batch_size" experiences from the memory and return them
    def sample_batch(self, batch_size):

        # sample "batch_size" experiences from the memory
        experiences = random.sample(self.memory, batch_size)

        num_elements = len(experiences[0])

        # create an empty list for each element of an experience
        # for example: obs, action, reward, next_obs, done
        lists = [[] for _ in range(num_elements)]

        # loop over all experiences
        for experience in experiences:
            # loop over element of an experience
            for element in range(num_elements):
                # store element in corresponding list
                lists[element].append(experience[element])

        return lists