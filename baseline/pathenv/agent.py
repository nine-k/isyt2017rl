import collections
import random

import numpy
import hashlib
import cPickle

MemoryRecord = collections.namedtuple('MemoryRecord',
                                      'observation action reward next_observation done')


GAMMA = 0.7
ALPHA = 0.2

RANDOM_ACTION_CHANCE = 0.1
MANUAL_CONTROL = False

class BaseAgent(object):
    def __init__(self,
                 input_shape=None,
                 number_of_actions=1,
                 max_memory_size=250,
                 load_table=None):
        self.input_shape = input_shape
        self.number_of_actions = number_of_actions
        self.max_memory_size = max_memory_size
        self.qTable = dict()
        if (load_table is not None):
            self.qTable = cPickle.load(open(load_table, "rb"))

        self.goal = None
        self.memory = []
        self.prev_result = None

        self._build_model()

    def __repr__(self):
        return self.__class__.__name__

    def _build_model(self):
        pass

    def new_episode(self, goal):
        #self.memory.append([])
        #self.memory = self.memory[-self.max_memory_size:]
        self.memory = list()
        self.goal = goal

    def get_best_action(self, observation):
        if (RANDOM_ACTION_CHANCE > random.uniform(0, 1)):
            action = numpy.random.choice(self.number_of_actions)
        else:
            observation_code = observation.tostring()
            if (observation_code not in self.qTable):
                self.qTable[observation_code] = [0] * 8
                action = numpy.random.choice(self.number_of_actions)
                #print("IDK BOUT THIS ONE")
            else:
                #print("OH BOY I KNOW THIS ONE")
                if all(v == 0 for v in self.qTable[observation_code]):
                    action = numpy.random.choice(self.number_of_actions)
                else:
                    action = numpy.argmax(self.qTable[observation_code])
        return action


    def act(self, observation):
        if (MANUAL_CONTROL):
            a = raw_input()[0]
            if a == 'w':
                action = 0
            elif a == 'e':
                action = 1
            elif a == 'd':
                action = 2
            elif a == 'c':
                action = 3
            elif a == 'x':
                action = 4
            elif a == 'z':
                action = 5
            elif a == 'a':
                action = 6
            elif a == 'q':
                action = 7
        # action = numpy.random.choice(self.number_of_actions)
        action = self.get_best_action(observation)
        # print(observation)
        return action

    def train_on_memory(self):
        # print("TRAINING HARD")
        for record in reversed(self.memory):
            reward = record.reward
            action = record.action
            state = record.observation.tostring()
            res_state = record.next_observation.tostring()
            if (state not in self.qTable):
                self.qTable[state] = [0] * 8;
            if (res_state not in self.qTable):
                self.qTable[res_state] = [0] * 8
            self.qTable[state][action] += ALPHA * (reward + GAMMA * max(self.qTable[res_state]) - self.qTable[state][action])

    def recalc_table(self):
        observation_code = self.prev_result.observation.tostring()
        next_observation_code = self.prev_result.next_observation.tostring()
        if (observation_code not in self.qTable):
            self.qTable[observation_code] = [0] * 8;
        if (next_observation_code not in self.qTable):
            self.qTable[next_observation_code] = [0] * 8
        self.qTable[observation_code][self.prev_result.action] += GAMMA * max(self.qTable[next_observation_code]) + self.prev_result.reward

    def update_memory(self, observation, action, reward, next_observation, done):
        self.memory.append(MemoryRecord(observation, action, reward, next_observation, done))
        #self.memory[-1].append(MemoryRecord(observation, action, reward, next_observation, done))
        #self.prev_result = MemoryRecord(observation, action, reward, next_observation, done)
        #self.recalc_table()

    def save_table(self, name):
        f = open(name, 'wb')
        cPickle.dump(self.qTable, f, 2)
