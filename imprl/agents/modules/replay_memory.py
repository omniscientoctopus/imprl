import random
from collections import deque


class AbstractReplayMemory:
    def __init__(self, size):
        self.size = size
        self.memory = deque(maxlen=size)

    def store_experience(self, *args):
        self.memory.append(args)

    def sample_batch(self, batch_size):
        """Randomly sample batch_size experiences from the memory"""

        experiences = random.sample(self.memory, batch_size)

        return self._create_lists(experiences)

    def _create_lists(self, experiences):

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


class EpisodicReplayMemory(AbstractReplayMemory):

    def __init__(self, size):
        super().__init__(size)

        self.experience = []

    def store_experience(self, *args):

        self.experience.append(args)

        # check if the episode is complete
        if args[-1]:  # done
            self.memory.append(self.experience)
            self.experience = []

    def sample_batch(self, batch_size, num_splits=1):

        all_episodes = random.sample(self.memory, batch_size)

        split_size = len(all_episodes) // num_splits
        experiences = [[] for _ in range(num_splits)]

        for i in range(num_splits):
            start = i * split_size
            end = (
                (i + 1) * split_size if i != num_splits - 1 else None
            )  # Extend to the end for the last split
            for episode in all_episodes[start:end]:
                experiences[i].extend(episode)

        return [self._create_lists(experience) for experience in experiences]
