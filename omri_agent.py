from . import BaseAgent
from util import flip_coin
import numpy as np


class NewAgent(BaseAgent):

    def __init__(self, *args, **kwargs):
        super(SimpleAgent, self).__init__(*args, **kwargs)
        _q_values = dict()
        _epsilon = 0.2

    def extract_state(self):
        pass

    def update_q_value(self, state, action, reward):
        pass

    def act(self, obs, action_space):
        if flip_coin(_epsilon):
            return np.argmax(_q_values[extract_state(obs)])
        else:
            return action_space.sample()
