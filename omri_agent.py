from pommerman.agents import BaseAgent
import numpy as np
import random

def flip_coin(p):
    r = random.random()
    return r > p

class NewAgent(BaseAgent):

    #TODO:
    # - stop computing after losing (because game keep running)
    # - when is reward returned, how?
    # in train script, call update_q_values. not in act()
    # - store q values in file
    # - how to get start state?

    def __init__(self, *args, **kwargs):
        super(NewAgent, self).__init__(*args, **kwargs)
        self.last_action = 0
        self.new_action = 0
        self.cur_state = (0,)
        self.last_state = (0,)
        self.q_values = dict()
        self.epsilon = 0.8
        self.discount = 1
        self.alpha = 1
        print('sdfsdfsdfs')

    def extract_state(self, obs):
        def is_bomb_adjacent(board, pos):
            x, y = pos
            adj = [(x+1,y),(x-1,y),(x,y+1),(x,y-1),(x,y)]
            danger = [0] * len(adj)
            for i, (x,y) in enumerate(adj):
                if x>=0 and x<=10 and y>=0 and y<=10:
                    if board[x,y] == 3:
                        danger[i] = 1
            return danger
        return is_bomb_adjacent(obs["board"], obs["position"])

    def update_q_value(self, reward):
        if tuple(self.cur_state) not in self.q_values:
            self.q_values[tuple(self.cur_state)] = [0] * 6
        if tuple(self.last_state) not in self.q_values:
            self.q_values[tuple(self.last_state)] = [0] * 6
        best_next_action = np.argmax(self.q_values[tuple(self.cur_state)])    
        td_target = reward + self.discount * self.q_values[tuple(self.cur_state)][best_next_action]
        td_delta = td_target - self.q_values[tuple(self.last_state)][self.last_action]
        self.q_values[tuple(self.last_state)][self.last_action] += self.alpha * td_delta
        

    def act(self, obs, action_space):
        #import IPython
        #IPython.embed()
        self.last_action = self.new_action
        self.last_state = self.cur_state
        self.cur_state = self.extract_state(obs)
        #self.update_q_value(self.last_state, self.last_action, cur_state)
        if tuple(self.extract_state(obs)) in self.q_values and flip_coin(self.epsilon):
            self.new_action = np.argmax(self.q_values[tuple(self.extract_state(obs))])
        else:
            self.new_action = action_space.sample()
        return self.new_action

    #def episode_end(self, reward):
    #    cur_state = 'done'
    #    self.update_q_value(self.last_state, self.last_action, cur_state, reward)

