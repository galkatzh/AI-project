from . import BaseAgent
import numpy as np
import random

def flip_coin(p):
    r = random.random()
    return r < p

class NewAgent(BaseAgent):

    #TODO:
    # - update cur_state at game beginning
    # - stop computing after losing (because game keep running)

    def __init__(self, *args, **kwargs):
        super(SimpleAgent, self).__init__(*args, **kwargs)
        self.last_action = 0
        self.last_state = 0
        self.q_values = dict()
        self.epsilon = 0.8
        self.discount = 1

    def extract_state(self, obs):
        def is_bomb_adjacent(board, pos):
            danger = [0] * 4
            x, y = pos
            adj = [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]
            for i, (x,y) in enumerate(adj):
                if x>=0 and x<=10 and y>=0 and y<=10:
                    if board[x,y] == 3
                        danger[i] = 1
            return danger
        return is_bomb_adjacent(obs["board"], obs["position"])

    def update_q_value(self, state, action, next_state, reward):
        if next_state not in self.q_values:
            self.q_values[next_state] = [0] * 6
        if state not in self.q_values:
            self.q_values[state] = [0] * 6
        best_next_action = np.argmax(self.q_values[next_state])    
        td_target = reward + discount_factor * Q[next_state][best_next_action]
        td_delta = td_target - Q[state][action]
        self.q_values[state][action] += alpha * td_delta

    def act(self, obs, action_space):
        cur_state = self.extract_state(obs)
        self.update_q_value(self.last_state, self.last_action, cur_state, reward)
        self.last_state = cur_state
        if flip_coin(epsilon):
            self.last_action = np.argmax(q_values[extract_state(obs)])
        else:
            self.last_action = action_space.sample()
        return self.last_action

    def episode_end(self, reward):
        cur_state = 'done'
        self.update_q_value(self.last_state, self.last_action, cur_state, reward)

