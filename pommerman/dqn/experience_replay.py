import random

import numpy as np
from pommerman.dqn import utils


class Memory:

    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []

    def clear(self):
        self.buffer.clear()

    def push(self, experience):
        if len(self.buffer) + 1 >= self.max_size:
            self.buffer[0:(1 + len(self.buffer)) - self.max_size] = []
        self.buffer.append(experience)


class RecurrentMemory:
    # Stores episodes
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []
        self.local_memory = Memory(max_size=max_size)

    def add_mem_to_buffer(self):
        if len(self.buffer) + 1 >= self.max_size:
            self.buffer[0:(1 + len(self.buffer)) - self.max_size] = []
        self.buffer.append(list(self.local_memory.buffer))
        self.local_memory.clear()

    def sample_recurrent(self, batch_size, time_step):
        tmp_buffer = [episode for episode in self.buffer if len(episode) + 1 > time_step]
        sampled_episodes = random.sample(tmp_buffer, batch_size)
        batch = []
        for episode in sampled_episodes:
            point = np.random.randint(0, len(episode) + 1 - time_step)
            batch.append(episode[point:point + time_step])

        return utils.split_sample(batch)

    def __len__(self):
        return len(self.buffer)
