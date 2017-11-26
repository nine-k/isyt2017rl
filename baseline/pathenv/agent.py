import collections
import random

import numpy

MemoryRecord = collections.namedtuple('MemoryRecord',
                                      'observation action reward next_observation done')


class BaseAgent(object):
    def __init__(self,
                 input_shape=None,
                 number_of_actions=1,
                 max_memory_size=250):
        self.input_shape = input_shape
        self.number_of_actions = number_of_actions
        self.max_memory_size = max_memory_size

        self.goal = None
        self.memory = []

        self._build_model()

    def __repr__(self):
        return self.__class__.__name__

    def _build_model(self):
        pass

    def new_episode(self, goal):
        self.memory.append([])
        self.memory = self.memory[-self.max_memory_size:]
        self.goal = goal

    def act(self, observation):
        action = numpy.random.choice(self.number_of_actions)
        return action

    def train_on_memory(self):
        # print("TRAINING HARD")
        pass

    def update_memory(self, observation, action, reward, next_observation, done):
        self.memory[-1].append(MemoryRecord(observation, action, reward, next_observation, done))
