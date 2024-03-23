from collections import deque

import numpy as np

class RunningStats():

    def __init__(self, max_size):
        
        self.memory = deque(maxlen=max_size)

    def get_summary(self, value):

        self.memory.append(value)

        summary = {'mean_cost': np.mean(self.memory),
                   'std_cost': np.std(self.memory),
                   'max_cost': np.max(self.memory),
                   'min_cost': np.min(self.memory)}

        return summary