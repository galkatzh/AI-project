from . import BaseAgent
from util import flip_coin
import numpy as np


class NewAgent(BaseAgent):

    def __init__(self, *args, **kwargs):
        super(SimpleAgent, self).__init__(*args, **kwargs)
        self.last_action = 0
        self.last_state = None
        self.q_values = dict()
        self.epsilon = 0.2
        self.discount = 1

    def extract_state(self):
        return obs

    def update_q_value(self, state, action, next_state, reward):
        pass

    def act(self, obs, action_space):
        cur_state = self.extract_state(obs)
        self.update_q_value(self.last_state, self.last_action, cur_state, reward)
        self.last_state = cur_state
        if flip_coin(epsilon):
            self.last_action = np.argmax(q_values[extract_state(obs)])
        else:
            self.last_action = action_space.sample()
        return self.last_action
